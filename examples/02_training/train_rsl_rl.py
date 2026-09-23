"""Train a rover height-map policy with RSL-RL and export it to ONNX."""

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

parser = argparse.ArgumentParser(description="Train a rover with RSL-RL PPO.")
parser.add_argument("--task", default="AAURoverEnvSimple-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point", help="Registered RSL-RL agent configuration key.")
parser.add_argument("--terrain", default=None, help="Registered terrain name, e.g. mars or debug.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", default=None, help="Resume from an RSL-RL checkpoint.")
parser.add_argument("--export-only", action="store_true", help="Export --checkpoint to ONNX without training.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.export_only and not args.checkpoint:
    parser.error("--export-only requires --checkpoint")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import rover_envs.envs.navigation.robots  # noqa: E402, F401
from rover_envs.utils.rsl_rl_wandb import patch_rsl_rl_logger_for_training_progress  # noqa: E402


def train() -> int:
    env = None
    exit_code = 0
    try:
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        agent_cfg = load_cfg_from_registry(args.task, args.agent)
        if args.terrain:
            env_cfg.scene.set_terrain(args.terrain)
        if args.seed is not None:
            agent_cfg.seed = args.seed
        if args.max_iterations is not None:
            agent_cfg.max_iterations = args.max_iterations
        agent_cfg.device = args.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
        env_cfg.seed = agent_cfg.seed

        log_dir = os.path.abspath(
            os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        )
        env_cfg.log_dir = log_dir
        gym_env = gym.make(args.task, cfg=env_cfg)
        step_dt = gym_env.unwrapped.step_dt
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        patch_rsl_rl_logger_for_training_progress(step_dt=step_dt)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        if args.checkpoint:
            runner.load(args.checkpoint, map_location=agent_cfg.device)
        if not args.export_only:
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        runner.export_policy_to_onnx(path=os.path.join(log_dir, "exported"), filename="policy.onnx")
        print(f"Exported ONNX policy to {log_dir}/exported/policy.onnx")
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        if env is not None:
            env.close()
        simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(train())
