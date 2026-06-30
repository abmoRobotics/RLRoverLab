#!/usr/bin/env python3
"""Evaluate a recurrent risk-conditioned CloneLab policy in RLRoverLab."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    "Evaluate a risk-conditioned recurrent CloneLab actor in an RLRoverLab task."
)
parser.add_argument("--checkpoint", required=True, help="Actor checkpoint file or checkpoint directory.")
parser.add_argument("--checkpoint_name", default="final_model.pt", help="Filename when --checkpoint is a directory.")
parser.add_argument("--risk_preference", type=float, required=True, help="Fixed alpha in [0, 1].")
parser.add_argument("--policy_factory", default=None, help="Optional CloneLab model factory override.")
parser.add_argument("--policy_config", default=None, help="Optional JSON model configuration override.")
parser.add_argument("--task", default="AAURoverEnvRGBDRawHD720-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--terrain", default=None)
parser.add_argument("--terrain-seed", type=int, default=None)
parser.add_argument("--keep-terrain", action="store_true", default=False)
parser.add_argument("--target_episodes", type=int, default=100)
parser.add_argument("--max_steps", type=int, default=50000)
parser.add_argument("--torch_device", default=None)
parser.add_argument("--stochastic", action="store_true", default=False)
parser.add_argument("--no_clip_actions", action="store_true", default=False)
parser.add_argument("--d_ref", type=float, default=5.0)
parser.add_argument("--clearance_exponent", type=float, default=2.5)
parser.add_argument("--metrics_out", required=True)
parser.add_argument("--episodes_out", required=True)
parser.add_argument(
    "--log_interval",
    type=int,
    default=100,
    help="Update tqdm metrics every N steps and on episode completion. Use 0 to disable it.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
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
from rover_envs.integrations.clonelab.observations import (  # noqa: E402
    CloneLabObservationConfig,
    RoverToCloneLabObservation,
)
from rover_envs.integrations.clonelab.policy import load_policy_config  # noqa: E402
from rover_envs.integrations.clonelab.risk_conditioned_policy import (  # noqa: E402
    RiskConditionedCloneLabActorPolicy,
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
from rover_envs.utils.terrain_utils import cleanup_terrain, handle_terrain_config  # noqa: E402


simulation_app = app_launcher.app


class ActionSpeedMetrics:
    def __init__(self) -> None:
        self.action_rows = 0
        self.forward_sum = 0.0
        self.near_1p5_rows = 0
        self.near_1p5_forward_sum = 0.0
        self.near_2p0_rows = 0
        self.near_2p0_forward_sum = 0.0
        self.high_speed_near_2p0_rows = 0

    def update(self, actions: torch.Tensor, clearance: torch.Tensor) -> None:
        actions = torch.as_tensor(actions).detach()
        clearance = torch.as_tensor(clearance, device=actions.device).detach().reshape(-1)
        forward = actions[:, 0].float().clamp(min=0.0)

        self.action_rows += int(forward.numel())
        self.forward_sum += float(forward.sum().item())

        near_1p5 = clearance < 1.5
        self.near_1p5_rows += int(near_1p5.sum().item())
        self.near_1p5_forward_sum += float(forward[near_1p5].sum().item())

        near_2p0 = clearance < 2.0
        self.near_2p0_rows += int(near_2p0.sum().item())
        self.near_2p0_forward_sum += float(forward[near_2p0].sum().item())
        self.high_speed_near_2p0_rows += int(((forward > 0.66) & near_2p0).sum().item())

    def summary(self) -> dict:
        return {
            "action_rows": self.action_rows,
            "mean_forward_speed": self._ratio(self.forward_sum, self.action_rows),
            "near_1p5_action_rows": self.near_1p5_rows,
            "mean_forward_speed_near_1p5m": self._ratio(
                self.near_1p5_forward_sum,
                self.near_1p5_rows,
            ),
            "near_2p0_action_rows": self.near_2p0_rows,
            "mean_forward_speed_near_2p0m": self._ratio(
                self.near_2p0_forward_sum,
                self.near_2p0_rows,
            ),
            "high_speed_near_2p0_fraction": self._ratio(
                self.high_speed_near_2p0_rows,
                self.near_2p0_rows,
            ),
        }

    @staticmethod
    def _ratio(numerator: float | int, denominator: float | int) -> float | None:
        return float(numerator) / float(denominator) if denominator else None


def _done_tensor(terminated, truncated, device: str) -> torch.Tensor:
    terminated = torch.as_tensor(terminated, device=device).bool()
    truncated = torch.as_tensor(truncated, device=device).bool()
    return torch.logical_or(terminated, truncated).reshape(-1)


def _write_episode_records(path: str | Path, records) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def main() -> int:
    if not 0.0 <= args_cli.risk_preference <= 1.0:
        raise ValueError(f"--risk_preference must lie in [0, 1], got {args_cli.risk_preference}.")
    if args_cli.target_episodes <= 0:
        raise ValueError("--target_episodes must be positive.")
    if args_cli.max_steps <= 0:
        raise ValueError("--max_steps must be positive.")

    device = args_cli.torch_device or ("cpu" if args_cli.cpu else "cuda:0")
    seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 100000000)
    set_seed(seed)

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.recorders = make_risk_evaluation_recorder_cfg()
    terrain_name, terrain_cleanup_path = handle_terrain_config(
        terrain_arg=args_cli.terrain,
        terrain_seed=args_cli.terrain_seed,
        keep_terrain=args_cli.keep_terrain,
    )
    if terrain_name is not None:
        env_cfg.scene.set_terrain(terrain_name)

    env = None
    exit_code = 0
    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
        model_config = load_policy_config(args_cli.policy_config) if args_cli.policy_config else None
        policy = RiskConditionedCloneLabActorPolicy.from_checkpoint(
            checkpoint=args_cli.checkpoint,
            checkpoint_name=args_cli.checkpoint_name,
            risk_preference=args_cli.risk_preference,
            factory_spec=args_cli.policy_factory,
            model_config=model_config,
            device=device,
        )
        observation_adapter = RoverToCloneLabObservation(
            CloneLabObservationConfig(
                device=device,
                proprioceptive_keys=policy.proprioceptive_keys,
            )
        )

        observations, _ = env.reset()
        num_envs = observation_adapter.num_envs(observations)
        policy.reset(num_envs)
        base_env = getattr(env, "unwrapped", env)
        metrics = RiskEvaluationMetrics(
            num_envs=num_envs,
            step_dt=float(base_env.step_dt),
            device=device,
            d_ref=args_cli.d_ref,
            clearance_exponent=args_cli.clearance_exponent,
            target_episodes=args_cli.target_episodes,
        )
        metrics.start_episodes(
            current_robot_xy(env),
            current_goal_distances(env),
        )
        speed_metrics = ActionSpeedMetrics()
        done_for_policy = torch.zeros(num_envs, dtype=torch.bool, device=device)

        progress = tqdm(
            range(args_cli.max_steps),
            total=args_cli.max_steps,
            desc=f"RC-RIQL alpha={args_cli.risk_preference:.3f}",
            unit="step",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=args_cli.log_interval == 0,
        )
        progress.set_postfix(
            episodes=f"0/{args_cli.target_episodes}",
            steps_s="n/a",
            success="n/a",
            collision="n/a",
            min_clearance="n/a",
            refresh=False,
        )
        evaluation_start_time = time.perf_counter()
        simulation_steps = 0
        for step in progress:
            policy.reset_done(done_for_policy)
            policy_state = observation_adapter.to_state(observations)
            actions = policy.act(policy_state, deterministic=not args_cli.stochastic)
            if not torch.isfinite(actions).all():
                raise RuntimeError("Policy produced non-finite actions.")
            if not args_cli.no_clip_actions:
                actions = actions.clamp(-1.0, 1.0)

            decision_clearance = current_clearance(env)
            speed_metrics.update(actions, decision_clearance)
            observations, rewards, terminated, truncated, _ = env.step(actions)
            simulation_steps = step + 1
            done = _done_tensor(terminated, truncated, device)
            final_positions_xy, final_clearance = post_step_risk_state(env)
            completed = metrics.update(
                final_positions_xy=final_positions_xy,
                decision_clearance=decision_clearance,
                final_clearance=final_clearance,
                rewards=torch.as_tensor(rewards, device=device, dtype=torch.float32).reshape(-1),
                done=done,
                termination_masks=termination_masks(env, device),
            )
            done_for_policy = done

            if done.any() and not metrics.target_reached:
                metrics.start_episodes(
                    current_robot_xy(env),
                    current_goal_distances(env),
                    env_mask=done,
                )

            should_log = args_cli.log_interval > 0 and (
                completed
                or (step + 1) % args_cli.log_interval == 0
                or metrics.target_reached
            )
            if should_log:
                current_summary = metrics.summary()
                elapsed_seconds = max(time.perf_counter() - evaluation_start_time, 1e-9)
                progress.set_postfix(
                    episodes=(
                        f"{current_summary['completed_episodes']}/"
                        f"{args_cli.target_episodes}"
                    ),
                    steps_s=f"{simulation_steps / elapsed_seconds:.2f}",
                    success=(
                        "n/a"
                        if current_summary["success_rate"] is None
                        else f"{current_summary['success_rate']:.3f}"
                    ),
                    collision=(
                        "n/a"
                        if current_summary["collision_rate"] is None
                        else f"{current_summary['collision_rate']:.3f}"
                    ),
                    min_clearance=(
                        "n/a"
                        if current_summary["mean_episode_minimum_clearance_m"] is None
                        else f"{current_summary['mean_episode_minimum_clearance_m']:.3f}"
                    ),
                    refresh=True,
                )
            if metrics.target_reached:
                break

        summary = {
            "task": args_cli.task,
            "checkpoint": str(Path(args_cli.checkpoint)),
            "risk_preference": float(args_cli.risk_preference),
            "seed": int(seed),
            "terrain": terrain_name or args_cli.terrain,
            "terrain_seed": args_cli.terrain_seed,
            "num_envs": num_envs,
            "simulation_steps": simulation_steps,
            "target_episodes": args_cli.target_episodes,
            "target_episodes_reached": metrics.target_reached,
            "d_ref": args_cli.d_ref,
            "clearance_exponent": args_cli.clearance_exponent,
            "speed": speed_metrics.summary(),
            **metrics.summary(),
        }
        metrics_path = Path(args_cli.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_episode_records(args_cli.episodes_out, metrics.records)
        print(f"[INFO] Evaluation metrics: {json.dumps(summary, sort_keys=True)}", flush=True)
        print(f"[INFO] Wrote aggregate metrics: {metrics_path}", flush=True)
        print(f"[INFO] Wrote episode metrics: {args_cli.episodes_out}", flush=True)

        if not metrics.target_reached:
            print(
                f"[ERROR] Reached --max_steps={args_cli.max_steps} before "
                f"--target_episodes={args_cli.target_episodes}.",
                flush=True,
            )
            exit_code = 2
    except BaseException as exc:
        print(f"[ERROR] Risk-conditioned evaluation failed with {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        exit_code = 1
    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        cleanup_terrain(terrain_cleanup_path if "terrain_cleanup_path" in locals() else None)
        simulation_app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
