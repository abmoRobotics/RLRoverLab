from __future__ import annotations

from isaaclab.utils import configclass
import rover_envs.mdp as mdp
from rover_envs.assets.robots.aau_rover import AAU_ROVER_CFG
from rover_envs.assets.robots.aau_rover_simple import AAU_ROVER_SIMPLE_CFG
from rover_envs.envs.navigation.rover_env_cfg import RoverEnvCfg
from rover_envs.envs.navigation.rover_env_camera_cfg import RoverRGBResnetEnvCfg, RoverCosmosEnvCfg, RoverRGBDRawEnvCfg, RoverRGBDRawTempEnvCfg
from rover_envs.envs.navigation.rover_env_cfg import RoverEnvDictCfg
@configclass
class AAURoverEnvCfgSimple(RoverEnvCfg):
    """Configuration for the AAU rover environment (simple version)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = AAU_ROVER_SIMPLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.actions = mdp.AckermannActionCfg(
            asset_name="robot",
            wheelbase_length=0.849,
            middle_wheel_distance=0.894,
            rear_and_front_wheel_distance=0.77,
            wheel_radius=0.1,
            min_steering_radius=0.8,
            steering_joint_names=[".*Steer_Revolute"],
            drive_joint_names=[".*Drive_Continuous"],
            offset=-0.0135
        )

@configclass
class AAURoverEnvCfg(RoverEnvCfg):
    """Configuration for the AAU rover environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = AAU_ROVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.actions = mdp.AckermannActionCfg(
            asset_name="robot",
            wheelbase_length=0.849,
            middle_wheel_distance=0.894,
            rear_and_front_wheel_distance=0.77,
            wheel_radius=0.1,
            min_steering_radius=0.8,
            steering_joint_names=[".*Steer_Revolute"],
            drive_joint_names=[".*Drive_Continuous"],
            offset=-0.0135
        )
        self.sim.dt = 1 / 30
        self.decimation = 6
        self.sim.physx.solver_type = 0  # 0: PGS, 1: TGS

@configclass
class AAURoverEnvDictCfg(RoverEnvDictCfg):
    """Configuration for the AAU rover environment with dictionary-based observation and action spaces."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = AAU_ROVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.actions = mdp.AckermannActionCfg(
            asset_name="robot",
            wheelbase_length=0.849,
            middle_wheel_distance=0.894,
            rear_and_front_wheel_distance=0.77,
            wheel_radius=0.1,
            min_steering_radius=0.8,
            steering_joint_names=[".*Steer_Revolute"],
            drive_joint_names=[".*Drive_Continuous"],
            offset=-0.0135
        )
        self.sim.dt = 1 / 30
        self.decimation = 6
        self.sim.physx.solver_type = 0  # 0: PGS, 1: TGS

@configclass
class AAURoverRGBResnetEnvCfg(RoverRGBResnetEnvCfg):
    """Configuration for the AAU rover environment with RGB ResNet."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = AAU_ROVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.actions = mdp.AckermannActionCfg(
            asset_name="robot",
            wheelbase_length=0.849,
            middle_wheel_distance=0.894,
            rear_and_front_wheel_distance=0.77,
            wheel_radius=0.1,
            min_steering_radius=0.8,
            steering_joint_names=[".*Steer_Revolute"],
            drive_joint_names=[".*Drive_Continuous"],
            offset=-0.0135
        )

@configclass
class AAURoverRGBCosmosEnvCfg(RoverCosmosEnvCfg):
    """Configuration for the AAU rover environment with RGB Cosmos tokenizer."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = AAU_ROVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.actions = mdp.AckermannActionCfg(
            asset_name="robot",
            wheelbase_length=0.849,
            middle_wheel_distance=0.894,
            rear_and_front_wheel_distance=0.77,
            wheel_radius=0.1,
            min_steering_radius=0.8,
            steering_joint_names=[".*Steer_Revolute"],
            drive_joint_names=[".*Drive_Continuous"],
            offset=-0.0135
        )

@configclass
class AAURoverRGBDRawEnvCfg(RoverRGBDRawEnvCfg):
    """Configuration for the AAU rover environment with RGB-D raw observations, will be used for learning by cheating."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = AAU_ROVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.actions = mdp.AckermannActionCfg(
            asset_name="robot",
            wheelbase_length=0.849,
            middle_wheel_distance=0.894,
            rear_and_front_wheel_distance=0.77,
            wheel_radius=0.1,
            min_steering_radius=0.8,
            steering_joint_names=[".*Steer_Revolute"],
            drive_joint_names=[".*Drive_Continuous"],
            offset=-0.0135
        )

@configclass
class AAURoverRGBDRawTempEnvCfg(RoverRGBDRawTempEnvCfg):
    """Temporary configuration for the AAU rover environment with RGB-D raw observations, will be used for learning by cheating."""
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = AAU_ROVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.actions = mdp.AckermannActionCfg(
            asset_name="robot",
            wheelbase_length=0.849,
            middle_wheel_distance=0.894,
            rear_and_front_wheel_distance=0.77,
            wheel_radius=0.1,
            min_steering_radius=0.8,
            steering_joint_names=[".*Steer_Revolute"],
            drive_joint_names=[".*Drive_Continuous"],
            offset=-0.0135
        )
        