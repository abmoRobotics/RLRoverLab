"""Evaluate driving policies on the BEV distillation task (Phase 4): the height-map teacher, the RGB student and the
BEV students, all on the same task and seeds.

Every finished episode's outcome is recorded: success, collision, leaving the area, or the time limit. So are its
duration, the driven path length and the straight-line distance between its first and last pose, and for BEV students
the episode's drift severity, which the policy never sees. Writes episodes.npz and metrics.json to logs/bev_policy/.

    python examples/03_inference/eval_bev_policy.py --policy bev \
        --checkpoint logs/rsl_rl/<experiment>/<run>/model_<iteration>.pt
"""

import argparse
import json
import math
import os
import sys
import traceback
from datetime import datetime
from importlib.metadata import version

# Keep BLAS single-threaded before Isaac Lab starts Kit in headless mode.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Evaluate driving policies on the BEV distillation task.")
parser.add_argument("--task", default="AAURoverEnvSimpleBEVDistillation-v0")
parser.add_argument("--policy", choices=("teacher", "rgb", "bev"), required=True)
parser.add_argument(
    "--checkpoint", default=None,
    help="PPO checkpoint for the teacher (default: the task's), distillation checkpoint for the students.",
)
parser.add_argument("--variant", choices=("full", "persistent", "instantaneous"), default=None,
                    help="BEV student variant; defaults to the task's.")
parser.add_argument("--severity", default="0.25:4", help="Drift severity range drawn per episode, e.g. 0.25:4, or 0.")
parser.add_argument("--terrain", default="mars")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1500, help="Policy steps (0.2 s each); only finished episodes count.")
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--name", default=None, help="Suffix of the output folder; defaults to the policy.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rover_envs.utils.launcher import configure_camera_launcher_args  # noqa: E402

configure_camera_launcher_args(args)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.utils import resolve_callable  # noqa: E402

import rover_envs.envs.navigation.robots  # noqa: E402, F401
from rover_envs.envs.navigation.learning.bev.pipeline import PersistentBev  # noqa: E402
from rover_envs.envs.navigation.learning.rsl_rl.bev_env import PersistentBevVecEnv  # noqa: E402
from rover_envs.envs.navigation.learning.rsl_rl.distillation_cfg import (RoverBevDistillationRunnerCfg,  # noqa: E402
                                                                         RoverDistillationRunnerCfg)
from rover_envs.envs.navigation.learning.rsl_rl.ppo_cfg import RoverPPORunnerCfg  # noqa: E402

# Termination terms, in the order that decides an episode's outcome when several fire at once.
OUTCOMES = ("is_success", "collision", "far_from_target", "time_limit")
SEVERITY_EDGES = (0.5, 1.0, 2.0)


def severity(text: str) -> float | tuple[float, float]:
    low, _, high = text.partition(":")
    return (float(low), float(high)) if high else float(low)


def load_policy(env, kind: str, checkpoint: str, device: str) -> torch.nn.Module:
    """The actor of the PPO teacher, or a distillation checkpoint's student, as it acts on ``env``'s observations.

    The BEV variants share one student configuration; they differ in the channels of the ``bev`` group only.
    """
    rsl_rl = version("rsl-rl-lib")
    if kind == "teacher":
        model_cfg, groups, key = handle_deprecated_rsl_rl_cfg(RoverPPORunnerCfg(), rsl_rl).actor, ["policy"], "actor"
    else:
        runner_cfg = RoverDistillationRunnerCfg() if kind == "rgb" else RoverBevDistillationRunnerCfg()
        runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, rsl_rl)
        model_cfg, groups, key = runner_cfg.student, runner_cfg.obs_groups["student"], "student"
    model_cfg = model_cfg.to_dict()
    model_class = resolve_callable(model_cfg.pop("class_name"))
    model = model_class(env.get_observations(), {key: groups}, key, env.num_actions, **model_cfg)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get(f"{key}_state_dict") or state["teacher_state_dict"])
    return model.to(device).eval()


def summarize(episodes: dict[str, np.ndarray], by_severity: bool) -> dict:
    def rates(rows: np.ndarray) -> dict:
        count = int(rows.sum())
        result = {"episodes": count}
        for outcome in OUTCOMES:
            share = (episodes["outcome"][rows] == OUTCOMES.index(outcome)).mean() if count else math.nan
            result[outcome] = float(share)
        success = rows & (episodes["outcome"] == 0)
        # Path length over the straight-line distance, and time, of successful episodes.
        ratio = episodes["path"][success] / np.maximum(episodes["straight"][success], 1e-6)
        result["path_ratio_median"] = float(np.median(ratio)) if success.any() else math.nan
        result["success_time_median_s"] = float(np.median(episodes["duration"][success])) if success.any() else math.nan
        return result

    summary = {"pooled": rates(np.ones(len(episodes["outcome"]), dtype=bool))}
    if by_severity:
        edges = (0.0, *SEVERITY_EDGES, math.inf)
        summary["by_severity"] = [
            {"severity": [low, high], **rates((episodes["severity"] >= low) & (episodes["severity"] < high))}
            for low, high in zip(edges[:-1], edges[1:])
        ]
    return summary


def evaluate() -> int:
    env = None
    exit_code = 0
    try:
        device = args.device
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        env_cfg.scene.set_terrain(args.terrain)
        env_cfg.seed = args.seed
        variant = args.variant or env_cfg.policy_bev.variant
        folder = os.path.abspath(os.path.join(
            "logs", "bev_policy", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.name or args.policy}"
        ))
        os.makedirs(folder)
        gym_env = gym.make(args.task, cfg=env_cfg)
        dt = gym_env.unwrapped.step_dt
        env = RslRlVecEnvWrapper(gym_env)
        pipeline = None
        if args.policy == "bev":
            env_cfg.policy_bev.variant = variant
            drift = severity(args.severity)
            env_cfg.policy_bev.severity = drift if isinstance(drift, tuple) else (drift, drift)
            pipeline = PersistentBev.from_cfg(env_cfg.policy_bev, env_cfg.bev, env.num_envs, dt, args.seed, device)
            env = PersistentBevVecEnv(env, pipeline, keep=("policy", "proprio", "pose"))
        checkpoint = args.checkpoint or gym.spec(args.task).kwargs["rsl_rl_teacher_model_path"]
        policy = load_policy(env, args.policy, checkpoint, device)
        terminations = env.unwrapped.termination_manager

        episodes = {key: [] for key in ("outcome", "duration", "path", "straight", "severity")}
        duration = torch.zeros(env.num_envs, device=device)
        path = torch.zeros(env.num_envs, device=device)
        obs = env.get_observations()
        start = obs["pose"][:, :2].clone()
        restarted = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        with torch.inference_mode():
            for _ in range(args.steps):
                if pipeline is not None and pipeline.odometry is not None:
                    episode_severity = pipeline.odometry.severity.clone()
                else:
                    episode_severity = torch.zeros(env.num_envs, device=device)
                position = obs["pose"][:, :2].clone()
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                done = dones.bool()
                after = obs["pose"][:, :2]
                duration += dt
                # Poses around a reset may be stale or teleported: neither the reset step nor the next one counts,
                # and a new episode starts at its first pose after the reset.
                path += torch.where(done | restarted, 0.0, (after - position).norm(dim=1))
                start = torch.where(restarted.unsqueeze(1), after, start)
                if done.any():
                    flags = torch.stack([terminations.get_term(name) for name in OUTCOMES], dim=1)
                    outcome = torch.where(flags.any(dim=1), flags.float().argmax(dim=1), len(OUTCOMES))
                    straight = (position - start).norm(dim=1)
                    for key, values in (("outcome", outcome), ("duration", duration), ("path", path),
                                        ("straight", straight), ("severity", episode_severity)):
                        episodes[key].append(values[done].cpu())
                    duration[done], path[done] = 0.0, 0.0
                restarted = done

        episodes = {key: torch.cat(values).numpy() for key, values in episodes.items()}
        summary = summarize(episodes, by_severity=args.policy == "bev")
        summary["config"] = {**vars(args), "variant": variant, "checkpoint": checkpoint, "dt": dt}
        np.savez(os.path.join(folder, "episodes.npz"), **episodes)
        with open(os.path.join(folder, "metrics.json"), "w") as file:
            json.dump(summary, file, indent=2, default=str)
        rows = [("pooled", summary["pooled"])]
        rows += [("{:g}-{:g}x".format(*row["severity"]), row) for row in summary.get("by_severity", [])]
        for label, row in rows:
            print(f"{label:10s} {row['episodes']:5d} episodes | success {row['is_success']:.3f} collision "
                  f"{row['collision']:.3f} far {row['far_from_target']:.3f} timeout {row['time_limit']:.3f} | "
                  f"path ratio {row['path_ratio_median']:.2f} time {row['success_time_median_s']:.1f} s")
        print(f"Wrote {folder}")
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        if env is not None:
            env.close()
        simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(evaluate())
