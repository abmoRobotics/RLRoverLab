import argparse
import math
import os
import random
import sys
from datetime import datetime

import gymnasium as gym
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
parser.add_argument("--dataset_dir", type=str, default="./datasets", help="Path to the dataset directory.")
parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset.")
parser.add_argument("--dataset_type", type=str, default="RL", choices=["IL", "RL"], help="Type of dataset to use. Options: IL or RL.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# app_launcher = AppLauncher(launcher_args=args_cli, experience=app_experience)

app_launcher = AppLauncher(args_cli)

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402


from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from skrl.trainers.torch import SequentialTrainer  # noqa: E402
from skrl.utils import set_seed  # noqa: E402, F401

import rover_envs  # noqa: E402
import rover_envs.envs.navigation.robots  # noqa: E402, F401
from rover_envs.utils.logging_utils import video_record, log_setup, configure_datarecorder  # noqa: E402
# Import agents
from rover_envs.envs.navigation.learning.skrl import get_agent  # noqa: E402
from rover_envs.utils.config import parse_skrl_cfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402

def main():
    args_cli_seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 100000000)
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0" if not args_cli.cpu else "cpu", num_envs=args_cli.num_envs)

    if args_cli.dataset_name is not None:
        env_cfg = configure_datarecorder(env_cfg, args_cli.dataset_dir, args_cli.dataset_name, args_cli.dataset_type)


    # key = agent name, value = path to config file
    experiment_cfg_file = gym.spec(args_cli.task).kwargs.get("skrl_cfgs")[args_cli.agent.upper()]
    experiment_cfg = parse_skrl_cfg(experiment_cfg_file)

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
    num_obs = env.observation_manager.group_obs_dim["policy"][0]
    num_actions = env.action_manager.action_term_dim[0]
    observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))

    trainer_cfg = experiment_cfg["trainer"]
    trainer_cfg["timesteps"] = 1000000

    agent = get_agent(args_cli.agent, env, observation_space, action_space, experiment_cfg, conv=True)

    # Get the checkpoint path from the experiment configuration
    print(f'args_cli.task: {args_cli.task}')
    agent_policy_path = gym.spec(args_cli.task).kwargs.pop("best_model_path")

    agent.load(agent_policy_path)
    trainer_cfg = experiment_cfg["trainer"]
    print(trainer_cfg)

    trainer = SequentialTrainer(cfg=trainer_cfg, agents=agent, env=env)
    trainer.eval()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
