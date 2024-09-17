import gymnasium as gym

from . import agents, ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg

gym.register(
    id="FrankaCubeLift-v0",
    entry_point="rover_envs.entrypoints.ManagerBasedRLEnv:ManagerBasedRLEnvLab",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.FrankaCubeLiftEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="FrankaCubeLift_REL-v0",
    entry_point="rover_envs.entrypoints.ManagerBasedRLEnv:ManagerBasedRLEnvLab",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.FrankaCubeLiftEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="FrankaCubeLift_ABS-v0",
    entry_point="rover_envs.entrypoints.ManagerBasedRLEnv:ManagerBasedRLEnvLab",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.FrankaCubeLiftEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
