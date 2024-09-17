import os

import gymnasium as gym

from . import env_cfg

gym.register(
    id="Exomy-v0",
    entry_point='rover_envs.envs.navigation.entrypoints:RoverEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.ExoMyEnvCfg,
        "best_model_path": f"{os.path.dirname(__file__)}/policies/best_agent.pt",
    }
)
