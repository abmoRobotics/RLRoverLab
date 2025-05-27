import os

import gymnasium as gym

from . import env_cfg

gym.register(
    id="AAURoverEnv-v0",
    entry_point='rover_envs.envs.navigation.entrypoints:RoverEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.AAURoverEnvCfg,
        "best_model_path": f"{os.path.dirname(__file__)}/policies/best_agent2.pt",
        "skrl_cfgs": {
            "PPO": f"{os.path.dirname(__file__)}/../../learning/skrl/configs/rover_ppo.yaml",
            "TRPO": f"{os.path.dirname(__file__)}/../../learning/skrl/configs/rover_trpo.yaml",
            "TD3": f"{os.path.dirname(__file__)}/../../learning/skrl/configs/rover_td3.yaml",
            "SAC": f"{os.path.dirname(__file__)}/../../learning/skrl/configs/rover_sac.yaml",
            "RPO": f"{os.path.dirname(__file__)}/../../learning/skrl/configs/rover_rpo.yaml",
        },
    }
)

gym.register(
    id="AAURoverEnvSimple-v0",
    entry_point='rover_envs.envs.navigation.entrypoints:RoverEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.AAURoverEnvCfgSimple,
        "best_model_path": f"{os.path.dirname(__file__)}/policies/best_agent2.pt",
        "skrl_cfgs": {
            "PPO": f"{os.path.dirname(__file__)}/../../learning/skrl/configs/rover_ppo.yaml",
            "TRPO": f"{os.path.dirname(__file__)}/../../learning/skrl/configs/rover_trpo.yaml",
            "TD3": f"{os.path.dirname(__file__)}/../../learning/skrl/configs/rover_td3.yaml",
            "SAC": f"{os.path.dirname(__file__)}/../../learning/skrl/configs/rover_sac.yaml",
            "RPO": f"{os.path.dirname(__file__)}/../../learning/skrl/configs/rover_rpo.yaml",
        },
    }
)

gym.register(
    id="AAURoverEnvCamera-v0",
    entry_point='rover_envs.envs.navigation.entrypoints:RoverEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.AAURoverRGBResnetEnvCfg,
        "skrl_cfgs": {
            "PPO": f"{os.path.dirname(__file__)}/../../learning/skrl/configs/rover_ppo_camera_resnet.yaml",
        },
        "best_model_path": f"{os.path.dirname(__file__)}/policies/best_agent_camera_resnet.pt",
    }
)
