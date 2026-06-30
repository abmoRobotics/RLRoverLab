import argparse
import math
import os
import random
import sys
from datetime import datetime

# Temporary work around for --viz=none
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Isaac Lab: Omniverse Robotics Environments!")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="AAURoverEnvSimple-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--agent", type=str, default="PPO", help="Name of the agent.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--experiment-dir", type=str, default=None, help="Experiment root under logs/skrl.")
parser.add_argument("--experiment-name", type=str, default=None, help="Run name inside the experiment directory.")
parser.add_argument("--max-steps", type=int, default=None, help="Override trainer timesteps.")
parser.add_argument("--policy-std", type=float, default=None, help="Override Gaussian policy std after loading a checkpoint.")
parser.add_argument("--wandb", action="store_true", default=True, help="Enable Weights & Biases logging during training.")
parser.add_argument("--terrain", type=str, default=None, help="Terrain type: 'mars', 'debug', 'random', or other registered terrain.")
parser.add_argument("--terrain-seed", type=int, default=None, help="Seed for random terrain generation (only used with --terrain random).")
parser.add_argument("--keep-terrain", action="store_true", default=False, help="Keep generated random terrain after use.")
parser.add_argument("--list-terrains", action="store_true", default=False, help="List available terrain types and exit.")

# Handle --list-terrains before AppLauncher to avoid starting simulation
if "--list-terrains" in sys.argv:
    from rover_envs.assets.terrains import list_terrains, get_terrain
    print("\nAvailable terrains:")
    for name in list_terrains():
        terrain = get_terrain(name)
        print(f"  - {name}: {terrain.description}")
    sys.exit(0)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# app_launcher = AppLauncher(launcher_args=args_cli, experience=app_experience)

app_launcher = AppLauncher(args_cli)

import gymnasium as gym  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

simulation_app = app_launcher.app

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from skrl.agents.torch.base import Agent  # noqa: E402
from skrl.trainers.torch import SequentialTrainer  # noqa: E402
from skrl.utils import set_seed  # noqa: E402, F401

import rover_envs.envs.navigation.robots  # noqa: E402, F401
# Import the general agent factory
from rover_envs.learning.agents import create_agent  # noqa: E402
from rover_envs.utils.config import parse_skrl_cfg  # noqa: E402
from rover_envs.utils.logging_utils import log_setup, video_record  # noqa: E402
from rover_envs.utils.skrl_wandb import patch_skrl_summary_writer_for_wandb  # noqa: E402
from rover_envs.utils.terrain_utils import handle_terrain_config, cleanup_terrain  # noqa: E402


def override_policy_std(agent: Agent, policy_std: float) -> None:
    if policy_std <= 0.0:
        raise ValueError(f"--policy-std must be positive, got {policy_std}")

    log_std = math.log(policy_std)
    updated_models = []
    for model_name, model in getattr(agent, "models", {}).items():
        log_std_parameter = getattr(model, "log_std_parameter", None)
        if log_std_parameter is None:
            continue
        log_std_parameter.data.fill_(log_std)
        updated_models.append(model_name)

    if not updated_models:
        raise RuntimeError("Could not find a log_std_parameter on any agent model.")

    print(f"[INFO] Set policy std to {policy_std} for models: {updated_models}")


def train():
    args_cli_seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 100000000)
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0" if not args_cli.cpu else "cpu", num_envs=args_cli.num_envs)
    
    # Handle terrain configuration (including random generation)
    terrain_name, terrain_cleanup_path = handle_terrain_config(
        terrain_arg=args_cli.terrain,
        terrain_seed=args_cli.terrain_seed,
        keep_terrain=args_cli.keep_terrain,
    )
    if terrain_name is not None:
        env_cfg.scene.set_terrain(terrain_name)
    
    # key = agent name, value = path to config file
    experiment_cfg_file = gym.spec(args_cli.task).kwargs.get("skrl_cfgs")[args_cli.agent.upper()]
    experiment_cfg = parse_skrl_cfg(experiment_cfg_file)
    experiment_settings = experiment_cfg.setdefault("agent", {}).setdefault("experiment", {})
    experiment_settings["wandb"] = bool(args_cli.wandb)
    if args_cli.experiment_dir is not None:
        experiment_settings["directory"] = args_cli.experiment_dir
    if args_cli.experiment_name is not None:
        experiment_settings["experiment_name"] = args_cli.experiment_name
    if args_cli.max_steps is not None:
        experiment_cfg.setdefault("trainer", {})["timesteps"] = args_cli.max_steps
    if args_cli.wandb:
        patch_skrl_summary_writer_for_wandb()

    log_dir = log_setup(experiment_cfg, env_cfg, args_cli.agent)

    # Create the environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, viewport=args_cli.video, render_mode=render_mode)
    # Check if video recording is enabled
    env = video_record(env, log_dir, args_cli.video, args_cli.video_length, args_cli.video_interval)
    # Wrap the environment
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

    # Get the observation and action spaces
    trainer_cfg = experiment_cfg["trainer"]

    agent: Agent = create_agent(args_cli.agent, env, experiment_cfg)
    if args_cli.checkpoint is not None:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        agent.load(args_cli.checkpoint)
    if args_cli.policy_std is not None:
        override_policy_std(agent, args_cli.policy_std)
    trainer = SequentialTrainer(cfg=trainer_cfg, agents=agent, env=env)
    trainer.train()

    env.close()
    simulation_app.close()
    
    # Cleanup temporary terrain if needed
    cleanup_terrain(terrain_cleanup_path)


if __name__ == "__main__":
    train()
