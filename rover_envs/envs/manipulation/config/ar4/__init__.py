import gymnasium as gym

from . import agents, ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg  # noqa: F401

gym.register(
    id="AR4CubeLift-v0",
    entry_point="rover_envs.entrypoints.RLTaskEnv:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.FrankaCubeLiftEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
