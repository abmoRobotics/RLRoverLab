import os

import gymnasium as gym

from . import env_cfg


gym.register(
    id="Sawppy-v0",
    entry_point="rover_envs.envs.navigation.entrypoints:RoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.SawppyEnvCfg,
        "rsl_rl_cfg_entry_point": "rover_envs.envs.navigation.learning.rsl_rl.ppo_cfg:RoverPPORunnerCfg",
        "skrl_cfgs": {
            "PPO": f"{os.path.dirname(__file__)}/../../learning/skrl/configs/rover_ppo.yaml",
            "TRPO": f"{os.path.dirname(__file__)}/../../learning/skrl/configs/rover_trpo.yaml",
            "TD3": f"{os.path.dirname(__file__)}/../../learning/skrl/configs/rover_td3.yaml",
            "SAC": f"{os.path.dirname(__file__)}/../../learning/skrl/configs/rover_sac.yaml",
            "RPO": f"{os.path.dirname(__file__)}/../../learning/skrl/configs/rover_rpo.yaml",
        },
    },
)
