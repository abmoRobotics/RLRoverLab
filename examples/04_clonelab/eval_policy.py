import argparse
import contextlib
import json
import os
import random
import sys
import traceback
from pathlib import Path

# Keep BLAS single-threaded before Isaac Lab imports. This mirrors the main
# RLRoverLab entrypoints and avoids atfork crashes in non-GUI runs.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from isaaclab.app import AppLauncher


DEFAULT_TASK = "AAURoverEnvRGBDRaw-v0"


parser = argparse.ArgumentParser("Evaluate a CloneLab actor in an RLRoverLab Isaac Lab task.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings.")
parser.add_argument("--video_dir", type=str, default="logs/clonelab_eval", help="Directory for video output.")
parser.add_argument(
    "--rover_rgb_video_out",
    type=str,
    default=None,
    help="Optional MP4 path for the rover RGB observation consumed by the policy.",
)
parser.add_argument(
    "--rover_rgb_video_fps",
    type=float,
    default=5.0,
    help="Frame rate for --rover_rgb_video_out.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Name of the Isaac Lab task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--terrain",
    type=str,
    default=None,
    help="Terrain type: mars, debug, random, or a registered terrain.",
)
parser.add_argument("--terrain-seed", type=int, default=None, help="Seed for random terrain generation.")
parser.add_argument(
    "--keep-terrain",
    action="store_true",
    default=False,
    help="Keep generated random terrain after use.",
)
parser.add_argument(
    "--list-terrains",
    action="store_true",
    default=False,
    help="List available terrain types and exit.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="CloneLab checkpoint file or checkpoint directory.")
parser.add_argument(
    "--checkpoint_name",
    type=str,
    default="best_model.pt",
    help="Checkpoint filename inside a directory.",
)
parser.add_argument(
    "--policy_factory",
    type=str,
    default=None,
    help="Import path for the CloneLab actor factory. Defaults to checkpoint export_config.json.",
)
parser.add_argument("--policy_config", type=str, default=None, help="Optional JSON file with actor keyword arguments.")
parser.add_argument(
    "--torch_device",
    type=str,
    default=None,
    help="Torch device. Defaults to cuda:0 unless --cpu is set.",
)
parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps.")
parser.add_argument(
    "--target_episodes",
    type=int,
    default=None,
    help="Optional number of completed episodes to collect before stopping.",
)
parser.add_argument("--warmup_steps", type=int, default=1, help="Initial zero-action steps before policy actions.")
parser.add_argument("--action_dim", type=int, default=2, help="Action dimension used for warmup actions.")
parser.add_argument("--stochastic", action="store_true", default=False, help="Sample from stochastic policies.")
parser.add_argument(
    "--recurrent",
    action="store_true",
    default=False,
    help="Reset recurrent policy state on episode end.",
)
parser.add_argument("--metrics_out", type=str, default=None, help="Optional path to write evaluation metrics as JSON.")
parser.add_argument("--episodes_out", type=str, default=None, help="Optional path to write per-episode metrics as JSONL.")
parser.add_argument("--d_ref", type=float, default=5.0)
parser.add_argument("--clearance_exponent", type=float, default=2.5)
parser.add_argument(
    "--clip_actions",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Clamp CloneLab actions to [-1, 1] before stepping the environment.",
)
parser.add_argument(
    "--log_interval",
    type=int,
    default=100,
    help="Update tqdm metrics every N steps. Use 0 to disable the progress bar.",
)

if "--list-terrains" in sys.argv:
    from rover_envs.assets.terrains import get_terrain, list_terrains

    print("\nAvailable terrains:")
    for name in list_terrains():
        terrain = get_terrain(name)
        print(f"  - {name}: {terrain.description}")
    sys.exit(0)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video or args_cli.rover_rgb_video_out:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from skrl.utils import set_seed  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

import rover_envs  # noqa: E402, F401
import rover_envs.envs.navigation.robots  # noqa: E402, F401
from rover_envs.integrations.clonelab import (  # noqa: E402
    CloneLabActorPolicy,
    CloneLabObservationConfig,
    RoverToCloneLabObservation,
    load_policy_config,
)
from rover_envs.integrations.clonelab.risk_evaluation_metrics import (  # noqa: E402
    RiskEvaluationMetrics,
    current_clearance,
    current_goal_distances,
    current_robot_xy,
    make_risk_evaluation_recorder_cfg,
    post_step_risk_state,
    termination_masks,
)
from rover_envs.utils.logging_utils import video_record  # noqa: E402
from rover_envs.utils.terrain_utils import cleanup_terrain, handle_terrain_config  # noqa: E402

simulation_app = app_launcher.app


def _as_done_tensor(terminated, truncated, device: str) -> torch.Tensor:
    terminated = torch.as_tensor(terminated, device=device).bool()
    truncated = torch.as_tensor(truncated, device=device).bool()
    return torch.logical_or(terminated, truncated).reshape(-1)


def _base_env(env):
    return getattr(env, "unwrapped", env)


def _get_termination_manager(env):
    base_env = _base_env(env)
    return getattr(base_env, "termination_manager", None)


def _get_termination_terms(env) -> tuple[str, ...]:
    termination_manager = _get_termination_manager(env)
    if termination_manager is None:
        return ()
    return tuple(getattr(termination_manager, "active_terms", ()))


def _get_termination_term(env, name: str, device: str) -> torch.Tensor | None:
    termination_manager = _get_termination_manager(env)
    if termination_manager is None or name not in getattr(termination_manager, "active_terms", ()):
        return None
    return torch.as_tensor(termination_manager.get_term(name), device=device).bool().reshape(-1)


class RoverRGBVideoWriter:
    """Write the first rover's policy RGB observation to an MP4 file."""

    def __init__(self, path: str, fps: float) -> None:
        if fps <= 0:
            raise ValueError(f"Rover RGB video FPS must be positive, got {fps}.")

        import cv2

        self._cv2 = cv2
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fps = float(fps)
        self._writer = None

    def write(self, observations) -> None:
        policy_observations = observations.get("policy", observations)
        frame = torch.as_tensor(policy_observations["rgb_image"])[0].detach().cpu()
        if frame.ndim != 3:
            raise ValueError(f"Expected rover RGB frame [H, W, C], got {tuple(frame.shape)}.")
        if frame.shape[-1] not in (3, 4) and frame.shape[0] in (3, 4):
            frame = frame.permute(1, 2, 0)
        if frame.shape[-1] not in (3, 4):
            raise ValueError(f"Expected rover RGB frame with 3 or 4 channels, got {tuple(frame.shape)}.")

        frame = frame[..., :3]
        if frame.dtype != torch.uint8:
            frame = frame.to(dtype=torch.float32)
            if frame.max().item() <= 1.0:
                frame = frame * 255.0
            frame = frame.clamp(0.0, 255.0).to(dtype=torch.uint8)

        frame_bgr = self._cv2.cvtColor(frame.numpy(), self._cv2.COLOR_RGB2BGR)
        if self._writer is None:
            frame_size = (frame_bgr.shape[1], frame_bgr.shape[0])
            self._writer = self._cv2.VideoWriter(
                str(self._path),
                self._cv2.VideoWriter_fourcc(*"mp4v"),
                self._fps,
                frame_size,
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"Failed to open rover RGB video writer: {self._path}")
        self._writer.write(frame_bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()


def _write_episode_records(path: str | Path, records) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def main() -> None:
    if args_cli.checkpoint is None:
        raise ValueError("Set --checkpoint to a CloneLab actor checkpoint file or checkpoint directory.")
    if args_cli.target_episodes is not None and args_cli.target_episodes <= 0:
        raise ValueError("--target_episodes must be positive when provided.")

    device = args_cli.torch_device or ("cpu" if args_cli.cpu else "cuda:0")
    seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 100000000)
    set_seed(seed)

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.recorders = make_risk_evaluation_recorder_cfg()
    print("[INFO] Parsed RLRoverLab environment config.", flush=True)
    terrain_name, terrain_cleanup_path = handle_terrain_config(
        terrain_arg=args_cli.terrain,
        terrain_seed=args_cli.terrain_seed,
        keep_terrain=args_cli.keep_terrain,
    )
    if terrain_name is not None:
        env_cfg.scene.set_terrain(terrain_name)

    env = None
    rover_rgb_video_writer = None
    try:
        try:
            if args_cli.video:
                env = gym.make(args_cli.task, cfg=env_cfg, viewport=True, render_mode="rgb_array")
            else:
                env = gym.make(args_cli.task, cfg=env_cfg)
        except BaseException as exc:
            print(f"[ERROR] gym.make exited with {type(exc).__name__}: {exc!r}", flush=True)
            raise
        print("[INFO] Created RLRoverLab environment.", flush=True)
        env = video_record(
            env,
            os.path.abspath(args_cli.video_dir),
            args_cli.video,
            args_cli.video_length,
            args_cli.video_interval,
        )
        print("[INFO] Wrapped RLRoverLab environment.", flush=True)

        model_config = load_policy_config(args_cli.policy_config) if args_cli.policy_config else None
        try:
            policy = CloneLabActorPolicy.from_checkpoint(
                factory_spec=args_cli.policy_factory,
                checkpoint=args_cli.checkpoint,
                checkpoint_name=args_cli.checkpoint_name,
                model_config=model_config,
                device=device,
            )
        except BaseException as exc:
            print(f"[ERROR] CloneLab policy setup failed with {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            raise
        print("[INFO] Loaded CloneLab policy.", flush=True)
        observation_adapter = RoverToCloneLabObservation(
            CloneLabObservationConfig(
                device=device,
                proprioceptive_keys=tuple(getattr(policy, "proprioceptive_keys", ("angle_diff", "distance", "heading"))),
            )
        )

        obs, _ = env.reset()
        print("[INFO] Reset RLRoverLab environment.", flush=True)
        if args_cli.rover_rgb_video_out:
            rover_rgb_video_writer = RoverRGBVideoWriter(
                args_cli.rover_rgb_video_out,
                args_cli.rover_rgb_video_fps,
            )
            print(f"[INFO] Recording rover RGB observations to: {args_cli.rover_rgb_video_out}", flush=True)
        num_envs = observation_adapter.num_envs(obs)
        if args_cli.recurrent:
            policy.reset(num_envs)
        base_env = _base_env(env)
        risk_metrics = RiskEvaluationMetrics(
            num_envs=num_envs,
            step_dt=float(base_env.step_dt),
            device=device,
            d_ref=args_cli.d_ref,
            clearance_exponent=args_cli.clearance_exponent,
            target_episodes=args_cli.target_episodes,
        )
        risk_metrics.start_episodes(
            current_robot_xy(env),
            current_goal_distances(env),
        )

        actions = torch.zeros((num_envs, args_cli.action_dim), device=device)
        episode_returns = torch.zeros(num_envs, device=device)
        episode_lengths = torch.zeros(num_envs, device=device)
        done_for_policy = torch.zeros(num_envs, dtype=torch.bool, device=device)
        completed_returns = []
        completed_lengths = []
        completed_episodes = 0
        termination_terms = _get_termination_terms(env)
        termination_counts = {name: 0 for name in termination_terms}
        total_reward = 0.0
        total_steps = 0
        recent_reward = 0.0
        recent_steps = 0

        print(
            f"[INFO] Evaluating CloneLab policy for {args_cli.steps} steps "
            f"on {args_cli.task} with {num_envs} envs.",
            flush=True,
        )

        progress = tqdm(
            range(args_cli.steps),
            total=args_cli.steps,
            desc="Evaluating",
            unit="step",
            dynamic_ncols=True,
            disable=args_cli.log_interval == 0,
        )
        for step in progress:
            if step >= args_cli.warmup_steps:
                if args_cli.recurrent:
                    policy.reset_done(done_for_policy)
                state = observation_adapter.to_state(obs)
                actions = policy.act(state, deterministic=not args_cli.stochastic)
                if args_cli.clip_actions:
                    actions = torch.clamp(actions, -1.0, 1.0)

            decision_clearance = current_clearance(env)
            obs, rewards, terminated, truncated, _ = env.step(actions)
            if rover_rgb_video_writer is not None:
                rover_rgb_video_writer.write(obs)
            rewards = torch.as_tensor(rewards, device=device, dtype=torch.float32).reshape(-1)
            done = _as_done_tensor(terminated, truncated, device)
            final_positions_xy, final_clearance = post_step_risk_state(env)
            risk_metrics.update(
                final_positions_xy=final_positions_xy,
                decision_clearance=decision_clearance,
                final_clearance=final_clearance,
                rewards=rewards,
                done=done,
                termination_masks=termination_masks(env, device),
            )
            for term_name in termination_terms:
                term_done = _get_termination_term(env, term_name, device)
                if term_done is not None:
                    termination_counts[term_name] += int(term_done.sum().item())
            episode_returns += rewards
            episode_lengths += 1
            reward_sum = float(rewards.sum().item())
            reward_count = rewards.numel()
            total_reward += reward_sum
            total_steps += reward_count
            recent_reward += reward_sum
            recent_steps += reward_count

            if done.any():
                completed_returns.extend(episode_returns[done].detach().cpu().tolist())
                completed_lengths.extend(episode_lengths[done].detach().cpu().tolist())
                episode_returns[done] = 0.0
                episode_lengths[done] = 0.0

            completed_episodes += int(done.sum().item())
            done_for_policy = done
            if done.any() and not risk_metrics.target_reached:
                risk_metrics.start_episodes(
                    current_robot_xy(env),
                    current_goal_distances(env),
                    env_mask=done,
                )

            should_log = args_cli.log_interval > 0 and (
                (step + 1) % args_cli.log_interval == 0 or (step + 1) == args_cli.steps
                or risk_metrics.target_reached
            )
            if should_log:
                mean_step_reward = total_reward / total_steps if total_steps else 0.0
                recent_mean_reward = recent_reward / recent_steps if recent_steps else 0.0
                mean_completed_return = (
                    sum(completed_returns) / len(completed_returns) if completed_returns else None
                )
                mean_completed_length = (
                    sum(completed_lengths) / len(completed_lengths) if completed_lengths else None
                )
                completed_return_text = (
                    "n/a" if mean_completed_return is None else f"{mean_completed_return:.4f}"
                )
                completed_length_text = (
                    "n/a" if mean_completed_length is None else f"{mean_completed_length:.1f}"
                )
                progress.set_postfix(
                    recent_reward=f"{recent_mean_reward:.4f}",
                    mean_reward=f"{mean_step_reward:.4f}",
                    partial_return=f"{episode_returns.mean().item():.4f}",
                    episodes=completed_episodes,
                    success_rate=(
                        "n/a"
                        if completed_episodes == 0
                        else f"{termination_counts.get('is_success', 0) / completed_episodes:.3f}"
                    ),
                    episode_return=completed_return_text,
                    episode_length=completed_length_text,
                    refresh=True,
                )
                recent_reward = 0.0
                recent_steps = 0
            if risk_metrics.target_reached:
                break

        success_count = termination_counts.get("is_success", 0)
        success_rate = success_count / completed_episodes if completed_episodes else None
        risk_summary = risk_metrics.summary()
        metrics = {
            "task": args_cli.task,
            "steps": args_cli.steps,
            "num_envs": num_envs,
            "target_episodes": args_cli.target_episodes,
            "target_episodes_reached": risk_metrics.target_reached,
            "d_ref": args_cli.d_ref,
            "clearance_exponent": args_cli.clearance_exponent,
            "completed_episodes": completed_episodes,
            "successes": success_count,
            "success_rate": success_rate,
            "termination_counts": termination_counts,
            "mean_step_reward": total_reward / total_steps if total_steps else 0.0,
            "mean_completed_return": sum(completed_returns) / len(completed_returns) if completed_returns else None,
            "mean_completed_length": sum(completed_lengths) / len(completed_lengths) if completed_lengths else None,
            "mean_partial_return": float(episode_returns.mean().item()),
            **risk_summary,
        }
        print(f"[INFO] Evaluation metrics: {metrics}", flush=True)

        if args_cli.metrics_out:
            metrics_path = Path(args_cli.metrics_out)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            print(f"[INFO] Wrote evaluation metrics: {metrics_path}", flush=True)
        if args_cli.episodes_out:
            _write_episode_records(args_cli.episodes_out, risk_metrics.records)
            print(f"[INFO] Wrote episode metrics: {args_cli.episodes_out}", flush=True)
        if args_cli.target_episodes is not None and not risk_metrics.target_reached:
            print(
                f"[ERROR] Reached --steps={args_cli.steps} before "
                f"--target_episodes={args_cli.target_episodes}.",
                flush=True,
            )
            raise SystemExit(2)

    finally:
        if rover_rgb_video_writer is not None:
            with contextlib.suppress(Exception):
                rover_rgb_video_writer.close()
        if env is not None:
            env.close()
        cleanup_terrain(terrain_cleanup_path if "terrain_cleanup_path" in locals() else None)
        simulation_app.close()


if __name__ == "__main__":
    main()
