import os

# import gymnasium as gym

ORBIT_CUSTOM_ENVS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

ORBIT_CUSTOM_ENVS_DATA_DIR = os.path.join(ORBIT_CUSTOM_ENVS_EXT_DIR, "learning")


# gym.register(
#     id='Rover-v0',
#     # entry_point='omni.isaac.lab.envs:ManagerBasedRLEnv',
#     entry_point='rover_envs.envs.navigation:RoverEnv',
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": "rover_envs.envs.navigation:RoverEnvCfg",
#     }
# )

# gym.register(
#     id='RoverNoObstacles-v0',
#     entry_point='rover_envs.envs.navigation:RoverEnv',
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": "rover_envs.envs.navigation:RoverEnvNoObstaclesCfg",
#     }
# )

# gym.register(
#     id='Rover_Manipulator-v0',
#     entry_point='omni.isaac.lab.envs:ManagerBasedRLEnv',
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": "rover_envs.envs.manipulation:ManipulatorEnvCfg",
#     }
# )

# gym.register(
#     id="RoverCamera-v0",
#     entry_point="envs.rover:RoverEnvCamera",
#     kwargs={"cfg_entry_point": "envs.rover:RoverEnvCfg"},
# )
