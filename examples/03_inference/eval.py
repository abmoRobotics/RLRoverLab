import argparse
import math
import os
import random
import sys
from datetime import datetime

# Temporary work around for --viz=none
# NumPy's OpenBLAS runtime is already loaded. Keep BLAS single-threaded before
# any Isaac Lab imports to avoid OpenBLAS atfork crashes in --viz none.
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
parser.add_argument("--task", type=str, default="AAURoverEnv-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--agent", type=str, default="PPO", help="Name of the agent.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--steps", type=int, default=1000000, help="Number of evaluation steps to run.")
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Optional episode length override in seconds. Useful for bounding legacy recorder export memory.",
)
parser.add_argument("--dataset_dir", type=str, default="./datasets", help="Path to the dataset directory.")
parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset.")
parser.add_argument(
    "--dataset_type",
    type=str,
    default="RL",
    choices=["IL", "RL", "RL_COMPRESSED"],
    help="Type of dataset to use. Options: IL, RL, or RL_COMPRESSED.",
)
parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging during evaluation.")
parser.add_argument("--terrain", type=str, default=None, help="Registered terrain name, e.g. 'mars' or 'debug'.")
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

from rover_envs.utils.launcher import configure_camera_launcher_args

configure_camera_launcher_args(args_cli)

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# app_launcher = AppLauncher(launcher_args=args_cli, experience=app_experience)

app_launcher = AppLauncher(args_cli)

import gymnasium as gym  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from skrl.agents.torch.base import Agent  # noqa: E402
from skrl.trainers.torch import SequentialTrainer  # noqa: E402
from skrl.utils import set_seed  # noqa: E402, F401

import rover_envs  # noqa: E402
import rover_envs.envs.navigation.robots  # noqa: E402, F401
# Import the general agent factory
from rover_envs.learning.agents import create_agent  # noqa: E402
# Import to ensure navigation agents are registered
#import rover_envs.envs.navigation.learning.skrl.agents  # noqa: E402, F401
from rover_envs.utils.config import parse_skrl_cfg  # noqa: E402
from rover_envs.utils.logging_utils import configure_datarecorder, log_setup, video_record  # noqa: E402
from rover_envs.utils.skrl_wandb import patch_skrl_summary_writer_for_wandb  # noqa: E402
from rover_envs.utils.terrain_utils import handle_terrain_config  # noqa: E402


def main():
    args_cli_seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 100000000)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s

    # Handle terrain configuration.
    terrain_name = handle_terrain_config(args_cli.terrain)
    if terrain_name is not None:
        env_cfg.scene.set_terrain(terrain_name)

    if args_cli.dataset_name is not None:
        env_cfg = configure_datarecorder(env_cfg, args_cli.dataset_dir, args_cli.dataset_name, args_cli.dataset_type)


    # key = agent name, value = path to config file
    experiment_cfg_file = gym.spec(args_cli.task).kwargs.get("skrl_cfgs")[args_cli.agent.upper()]
    experiment_cfg = parse_skrl_cfg(experiment_cfg_file)
    # Evaluation does not train, so keep SKRL memory small. This matters for HD images in dict observations.
    experiment_cfg["agent"]["rollouts"] = 1
    experiment_cfg.setdefault("agent", {}).setdefault("experiment", {})["wandb"] = bool(args_cli.wandb)
    if args_cli.wandb:
        patch_skrl_summary_writer_for_wandb()

    log_dir = log_setup(experiment_cfg, env_cfg, args_cli.agent)

    # Create the environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    # Check if video recording is enabled
    env = video_record(env, log_dir, args_cli.video, args_cli.video_length, args_cli.video_interval)
    # Wrap the environment
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

    # Get the observation and action spaces
    num_obs = env.observation_manager.group_obs_dim["policy"][0]
    num_actions = env.action_manager.action_term_dim[0]
    #observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
    #action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))

    trainer_cfg = experiment_cfg["trainer"]
    trainer_cfg["timesteps"] = args_cli.steps

    agent: Agent = create_agent(args_cli.agent, env, experiment_cfg)

    # Get the checkpoint path from the experiment configuration
    print(f'args_cli.task: {args_cli.task}')
    agent_policy_path = args_cli.checkpoint or gym.spec(args_cli.task).kwargs.get("best_model_path")
    if agent_policy_path is None:
        raise ValueError(f"Task '{args_cli.task}' has no default checkpoint; pass --checkpoint explicitly.")

    agent.load(agent_policy_path)
    trainer_cfg = experiment_cfg["trainer"]
    print(trainer_cfg)

    trainer = SequentialTrainer(cfg=trainer_cfg, agents=agent, env=env)
    trainer.eval()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
