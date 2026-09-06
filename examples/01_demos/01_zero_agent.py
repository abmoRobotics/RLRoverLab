import argparse
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero-action rover demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create")
parser.add_argument("--task", type=str, default="AAURoverEnv-v0", help="Task name")
parser.add_argument("--robot", type=str, default="aau_rover", help="Robot name")
parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="Stop after this many environment steps (default: run until the app closes)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

from rover_envs.utils.launcher import configure_camera_launcher_args

configure_camera_launcher_args(args_cli)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import rover_envs.envs.navigation.robots  # noqa: E402, F401


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f'[INFO]: Created environment with task: {args_cli.task}')
    print(f'[INFO]: Number of environments: {args_cli.num_envs}')
    print(f'[INFO]: Using GPU: {str(args_cli.device).startswith("cuda")}')
    print(f'[INFO]: Using Fabric: {not getattr(args_cli, "disable_fabric", False)}')
    print(f'[INFO]: Robot: {args_cli.robot}')

    print(f'[INFO]: Environment observation space: {env.observation_space}')
    print(f'[INFO]: Environment action space: {env.action_space}')

    env.reset()

    action_shape = env.action_space.shape
    step_count = 0
    while simulation_app.is_running() and (args_cli.steps is None or step_count < args_cli.steps):
        with torch.inference_mode():
            actions = torch.zeros(action_shape, device=env.unwrapped.device)
            actions[..., 0] = 1.0  # Move forward
            if actions.shape[-1] > 1:
                actions[..., 1] = 0.0
            env.step(actions)
            step_count += 1

    print(f"[INFO]: Completed {step_count} environment steps")
    env.close()


if __name__ == "__main__":
    error = None
    try:
        main()
    except Exception as e:
        error = e
        print(f"Error in main: {e}")
        traceback.print_exc()
    finally:
        try:
            simulation_app.close()
        except SystemExit:
            if error is None:
                raise
    if error is not None:
        raise SystemExit(1) from error
