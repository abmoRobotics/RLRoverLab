from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import rover_envs.mdp as mdp
import rover_envs.envs.navigation.mdp as navigation_mdp
from rover_envs.assets.robots.aau_rover import AAU_ROVER_CFG
from rover_envs.assets.robots.aau_rover_simple import AAU_ROVER_SIMPLE_CFG
from rover_envs.envs.navigation.rover_env_cfg import RewardsCfg, RoverEnvCfg
from rover_envs.envs.navigation.utils.terrains.risk_map import ObstacleRiskMapCfg
from rover_envs.envs.navigation.rover_env_camera_cfg import (
    RoverCosmosEnvCfg,
    RoverRGBDRawEnvCfg,
    RoverRGBDRawHD1080EnvCfg,
    RoverRGBDRawHD720EnvCfg,
    RoverRGBDRawTempEnvCfg,
    RoverRGBDRawWVGAEnvCfg,
    RoverRGBResnetEnvCfg,
)
from rover_envs.envs.navigation.rover_env_cfg import RoverEnvDictCfg


@configclass
class ConservativeTeacherRewardsCfg(RewardsCfg):
    """Base navigation reward plus a smooth obstacle-risk penalty."""

    obstacle_risk = RewTerm(
        func=navigation_mdp.obstacle_risk_cost,
        weight=-20.0,
        params={"asset_cfg": SceneEntityCfg(name="robot")},
    )


@configclass
class VeryConservativeTeacherRewardsCfg(RewardsCfg):
    """Base navigation reward plus a wider, stronger obstacle-risk penalty."""

    obstacle_risk = RewTerm(
        func=navigation_mdp.obstacle_risk_cost,
        weight=-40.0,
        params={"asset_cfg": SceneEntityCfg(name="robot")},
    )


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
class AAURoverEnvSimpleRiskTeacherCfg(AAURoverEnvCfgSimple):
    """Simple AAU rover environment for moderately conservative teacher PPO training."""

    rewards: ConservativeTeacherRewardsCfg = ConservativeTeacherRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.obstacle_risk_cfg = ObstacleRiskMapCfg(decay_distance_m=1.5)


@configclass
class AAURoverEnvSimpleVeryRiskTeacherCfg(AAURoverEnvCfgSimple):
    """Simple AAU rover environment for very conservative teacher PPO training."""

    rewards: VeryConservativeTeacherRewardsCfg = VeryConservativeTeacherRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.obstacle_risk_cfg = ObstacleRiskMapCfg(decay_distance_m=2.5)


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
        self.sim.physics.solver_type = 0  # 0: PGS, 1: TGS
        self.sim.physics.enable_external_forces_every_iteration = 0

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
        self.sim.physics.solver_type = 0  # 0: PGS, 1: TGS
        self.sim.physics.enable_external_forces_every_iteration = 0

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
        # Make robot invisible to its own sensors and other robots
        robot_cfg = AAU_ROVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot_cfg.spawn.visible = True
        self.scene.robot = robot_cfg
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
        self.sim.dt = 1 / 45
        self.decimation = 9

@configclass
class AAURoverRGBDRawHD720EnvCfg(RoverRGBDRawHD720EnvCfg):
    """Configuration for the AAU rover with ZED2i HD720 RGB-D raw observations."""

    def __post_init__(self):
        super().__post_init__()
        robot_cfg = AAU_ROVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot_cfg.spawn.visible = True
        self.scene.robot = robot_cfg
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
        self.sim.dt = 1 / 45
        self.decimation = 9

@configclass
class AAURoverRGBDRawHD1080EnvCfg(RoverRGBDRawHD1080EnvCfg):
    """Configuration for the AAU rover with ZED2i HD1080 RGB-D raw observations."""

    def __post_init__(self):
        super().__post_init__()
        robot_cfg = AAU_ROVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot_cfg.spawn.visible = True
        self.scene.robot = robot_cfg
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
        self.sim.dt = 1 / 45
        self.decimation = 9

@configclass
class AAURoverRGBDRawWVGAEnvCfg(RoverRGBDRawWVGAEnvCfg):
    """Configuration for the AAU rover with ZED2i WVGA RGB-D raw observations."""

    def __post_init__(self):
        super().__post_init__()
        robot_cfg = AAU_ROVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot_cfg.spawn.visible = True
        self.scene.robot = robot_cfg
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
        self.sim.dt = 1 / 45
        self.decimation = 9

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
        
