import os

import gymnasium as gym

from ...learning.skrl import get_agent
from . import env_cfg

gym.register(
    id="AAURoverEnv-v0",
    entry_point='rover_envs.envs.navigation.entrypoints:RoverEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.AAURoverEnvCfg,
        "best_model_path": f"{os.path.dirname(__file__)}/policies/best_agent.pt",
        "get_agent_fn": get_agent,
    }
)

gym.register(
    id="AAURoverEnvCamera-v0",
    entry_point='rover_envs.envs.navigation.entrypoints:RoverEnvCamera',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.AAURoverEnvCfg,
        "best_model_path": f"{os.path.dirname(__file__)}/policies/best_agent.pt",
        "get_agent_fn": get_agent,
    }
)
