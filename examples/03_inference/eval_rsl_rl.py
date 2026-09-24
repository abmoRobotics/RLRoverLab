"""Evaluate a rover policy trained with RSL-RL (PPO or teacher-student distillation)."""

import argparse
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

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a rover policy trained with RSL-RL.")
parser.add_argument("--task", default="AAURoverEnvSimple-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point", help="Registered RSL-RL agent configuration key.")
parser.add_argument("--terrain", default=None, help="Registered terrain name, e.g. mars or debug.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", default=None, help="Checkpoint to evaluate; defaults to the task's best model.")
parser.add_argument("--steps", type=int, default=1000000, help="Number of evaluation steps to run.")
parser.add_argument("--video", action="store_true", help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rover_envs.utils.launcher import configure_camera_launcher_args  # noqa: E402

configure_camera_launcher_args(args)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import rover_envs.envs.navigation.robots  # noqa: E402, F401
from rover_envs.utils.logging_utils import video_record  # noqa: E402


def evaluate() -> int:
    env = None
    exit_code = 0
    try:
        checkpoint = args.checkpoint or gym.spec(args.task).kwargs.get("rsl_rl_best_model_path")
        if checkpoint is None:
            raise ValueError(f"Task '{args.task}' has no RSL-RL best model; pass --checkpoint.")
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        agent_cfg = load_cfg_from_registry(args.task, args.agent)
        if args.terrain:
            env_cfg.scene.set_terrain(args.terrain)
        if args.seed is not None:
            agent_cfg.seed = args.seed
        agent_cfg.device = args.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
        env_cfg.seed = agent_cfg.seed

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, f"eval_{timestamp}"))
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        env = video_record(env, log_dir, args.video, args.video_length, args.video_interval)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_class = DistillationRunner if agent_cfg.class_name == "DistillationRunner" else OnPolicyRunner
        runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint, map_location=agent_cfg.device)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(args.steps):
                if not simulation_app.is_running():
                    break
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
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
