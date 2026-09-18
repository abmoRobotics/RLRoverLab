from isaaclab.utils import configclass

import rover_envs.mdp as mdp
from rover_envs.assets.robots.sawppy import SAWPPY_CFG
from rover_envs.envs.navigation.rover_env_cfg import RoverEnvCfg


@configclass
class SawppyEnvCfg(RoverEnvCfg):
    """Configuration for the Sawppy rover heightmap environment."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SAWPPY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/SAWPPY/SAWPPY/Body_Box"
        self.scene.contact_sensor.prim_path = (
            "{ENV_REGEX_NS}/Robot/SAWPPY/SAWPPY/"
            "(Body_Box|Bogie|Bogiev2_Mirrored|Steerable_Wheel(_Mirrored)?(_01)?|Wheel(_Mirrored)?(_0[12])?)"
        )
        self.actions.actions = mdp.AckermannActionCfg(
            asset_name="robot",
            wheelbase_length=0.542,
            middle_wheel_distance=0.528,
            rear_and_front_wheel_distance=0.458,
            wheel_radius=0.062,
            min_steering_radius=0.3,
            steering_joint_names=[".*_Steer"],
            drive_joint_names=[".*_Drive"],
            offset=-0.0136,
        )
        self.actions.actions.drive_order = ["FL", "FR", "ML", "MR", "RL", "RR"]
