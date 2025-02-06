

import argparse
import traceback

import carb
import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Empty Scene")

parser.add_argument("--cpu", default=False, action="store_true", help="Run on CPU")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create")
parser.add_argument("--task", type=str, default="AAURoverEnv-v0", help="Task name")
parser.add_argument("--robot", type=str, default="aau_rover", help="Robot name")

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import parse_env_cfg  # noqa: F401, E402

import rover_envs.envs.navigation.robots  # noqa: F401, E402

# Import agents


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device="cuda:0" if not args_cli.cpu else "cpu", num_envs=args_cli.num_envs
    )

    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f'[INFO]: Created environment with task: {args_cli.task}')
    print(f'[INFO]: Number of environments: {args_cli.num_envs}')
    print(f'[INFO]: Using GPU: {not args_cli.cpu}')
    print(f'[INFO]: Using Fabric: {not args_cli.disable_fabric}')
    print(f'[INFO]: Robot: {args_cli.robot}')

    print(f'[INFO]: Environment observation space: {env.observation_space}')
    print(f'[INFO]: Environment action space: {env.action_space}')

    env.reset()

    while simulation_app.is_running:
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Error in main: {e}")
        carb.log_error(traceback.format_exc())
    finally:
        simulation_app.close()
