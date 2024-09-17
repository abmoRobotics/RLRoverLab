from __future__ import annotations

from omni.isaac.lab.utils import configclass

import rover_envs.mdp as mdp
from rover_envs.assets.robots.exomy import EXOMY_CFG
from rover_envs.envs.navigation.rover_env_cfg import RoverEnvCfg


@configclass
class ExoMyEnvCfg(RoverEnvCfg):
    """Configuration for the ExoyMy rover environment."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = EXOMY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Define kinematics for the robot
        self.actions.actions = mdp.AckermannActionCfg(
            asset_name="robot",
            wheelbase_length=0.29778,
            middle_wheel_distance=0.1548,
            rear_and_front_wheel_distance=0.1548,
            wheel_radius=0.1,
            min_steering_radius=0.4,
            steering_joint_names=["FL_Steer_Joint", "RL_Steer_Joint", "RR_Steer_Joint", "FR_Steer_Joint"],
            drive_joint_names=[".*Drive_Joint"],
        )
