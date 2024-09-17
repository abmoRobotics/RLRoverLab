# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

from __future__ import annotations

import argparse
import copy
import os
import random

from omni.isaac.lab.app import AppLauncher

import wandb

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--agent", type=str, default="PPO", help="Name of the agent.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
# load cheaper kit config in headless
if args_cli.headless:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
else:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
# launch omniverse app
app_launcher = AppLauncher(args_cli, experience=app_experience)
simulation_app = app_launcher.app


import traceback  # noqa: F401, E402
from datetime import datetime  # noqa: F401, E402

import carb  # noqa: F401, E402
import gymnasium as gym  # noqa: F401, E402
from omni.isaac.lab.utils.dict import print_dict  # noqa: F401, E402
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml  # noqa: F401, E402
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: F401, E402
from omni.isaac.lab_tasks.utils.wrappers.skrl import process_skrl_cfg  # noqa: F401, E402
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG  # noqa: F401, E402
from skrl.memories.torch import RandomMemory  # noqa: F401, E402
from skrl.utils import set_seed  # noqa: F401, E402
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model  # noqa: F401, E402

import rover_envs.envs.manipulation.config.franka.agents  # noqa: F401, E402
import rover_envs.envs.navigation.robots  # noqa: E402, F401
from rover_envs.utils.config import parse_skrl_cfg  # noqa: E402
from rover_envs.utils.skrl_utils import SkrlSequentialLogTrainer  # noqa: E402, E402
from rover_envs.utils.skrl_utils import SkrlVecEnvWrapper  # noqa: E402, E402

# import rover_envs.envs  # noqa: F401


def main(env, tags, sweep=False):
    """Train with skrl agent."""
    # read the seed from command line
    args_cli_seed = args_cli.seed
    CONV = True
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    experiment_cfg = parse_skrl_cfg(args_cli.task + f"_{args_cli.agent}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{experiment_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    experiment_cfg["agent"]["experiment"]["directory"] = log_root_path
    experiment_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), experiment_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), experiment_cfg)

    # create isaac environment

    # wrap for video recording

    # wrap around environment for skrl

    experiment_cfg["seed"] = random.randint(0, 1000000)
    # set seed for the experiment (override from command line)
    seed = args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"]
    set_seed(seed)

    # configure and instantiate PPO agent
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    experiment_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    agent_cfg.update(process_skrl_cfg(experiment_cfg["agent"]))
    trainer_cfg = experiment_cfg["trainer"]
    agent_cfg["seed"] = seed
    agent_cfg["task"] = args_cli.task
    agent_cfg["num_obs"] = num_obs = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
    agent_cfg["conv"] = CONV
    config = {**agent_cfg, **trainer_cfg}
    # set default values
    wandb_kwargs = copy.deepcopy(agent_cfg.get("experiment", {}).get("wandb_kwargs", {}))
    wandb_kwargs.setdefault("name", os.path.split(log_dir)[-1])
    wandb_kwargs.setdefault("sync_tensorboard", True)
    wandb_kwargs.setdefault("config", {})
    wandb_kwargs["config"].update(config)
    # init Weights & Biases
    wandb.init(project='RLRoverLab-AAURoverEnv-v0', tags=tags, **wandb_kwargs)
    if sweep:
        # agent_cfg["mini_batches"] = wandb.config.rollouts // wandb.config.mini_batch_factor
        agent_cfg["rollouts"] = wandb.config.rollouts
        agent_cfg["mini_batches"] = wandb.config.rollouts // wandb.config.mini_batch_factor
        agent_cfg["learning_rate"] = wandb.config.learning_rate
        agent_cfg["discount_factor"] = wandb.config.discount_factor
        agent_cfg["lambda"] = wandb.config.lambda_
        agent_cfg["value_loss_scale"] = wandb.config.value_loss_scale
        agent_cfg["entropy_loss_scale"] = wandb.config.entropy_loss_scale
        agent_cfg["kl_threshold"] = wandb.config.kl_threshold
        agent_cfg["learning_epochs"] = wandb.config.learning_epochs
        wandb.config.update({'mini_batches': agent_cfg["mini_batches"]}, allow_val_change=True)

    agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})
    # if sweep:
    #     agent_cfg["mini_batches"] = wandb.config.rollouts / wandb.config.
    import math  # noqa: E402

    from rover_envs.learning.train import get_agent  # noqa: E402

    # Get the observation and action spaces
    num_obs = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
    num_actions = env.unwrapped.action_manager.action_term_dim[0]
    observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))
    agent = get_agent(args_cli.agent, env, observation_space, action_space, experiment_cfg, conv=CONV)
    # agent.load("best_agent.pt")
    # configure and instantiate a custom RL trainer for logging episode events
    # https://skrl.readthedocs.io/en/latest/modules/skrl.trainers.base_class.html

    trainer = SkrlSequentialLogTrainer(cfg=trainer_cfg, env=env, agents=agent)
    # train the agent
    trainer.train()

    # # close the simulator
    # env.close()
    wandb.finish()


def start_env():
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    return SkrlVecEnvWrapper(env)


def sweep(env):
    sweep_configuration = {
        'method': 'bayes',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'Reward / Total reward (mean)'},
        'parameters':
        {
            'mini_batch_factor': {'values': [1, 2, 4]},
            'learning_epochs': {'values': [2, 4, 6, 8, 10]},
            'learning_rate': {'max': 0.003, 'min': 0.00003},
            'rollouts': {'values': [60, 50, 40, 30, 20, 10, 6]},
            'discount_factor': {'values': [0.99, 0.98, 0.95, 0.9, 0.85, 0.8]},
            'lambda_': {'values': [0.99, 0.98, 0.95, 0.9, 0.85, 0.8]},
            'value_loss_scale': {'values': [0.5, 1, 2]},
            'entropy_loss_scale': {'max': 0.1, 'min': 0.0001},
            'kl_threshold': {'values': [0.02, 0.01, 0.008, 0.004, 0.002]},
        }
    }

    sweep_id = wandb.sweep(sweep_configuration, project=f'RLRoverLab-{args_cli.task}')
    wandb.agent(sweep_id, function=lambda: main(env, tags=[args_cli.task, "experiment:sweep"], sweep=True))
    # wandb.agent("ox5aways", project="RLRoverLab-FrankaCubeLift-v0",
    #             function=lambda: main(env, tags=[args_cli.task, "experiment:sweep"], sweep=True))


if __name__ == "__main__":
    try:
        # run the main execution
        env = start_env()
        # main(env, tags=[args_cli.task, "experiment:train"])
        for i in range(5):
            main(env, tags=[args_cli.task, "experiment:observation_space"])
#        sweep(env)
        env.close()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
