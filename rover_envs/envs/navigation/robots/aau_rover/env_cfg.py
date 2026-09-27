from __future__ import annotations

import os

from isaaclab.utils import configclass
import rover_envs.mdp as mdp
from rover_envs.assets.robots.aau_rover import AAU_ROVER_CFG
from rover_envs.assets.robots.aau_rover_simple import AAU_ROVER_SIMPLE_CFG
from rover_envs.envs.navigation.rover_env_cfg import RoverEnvCfg
from rover_envs.envs.navigation.rover_env_camera_cfg import (
    PolicyBevCfg,
    RoverBevDistillationObservationsCfg,
    RoverBevEnvCfg,
    RoverDistillationEnvCfg,
    RoverRGBDRawEnvCfg,
    RoverRGBDRawHD1080EnvCfg,
    RoverRGBDRawHD720EnvCfg,
    RoverRGBDRawTempEnvCfg,
    RoverRGBDRawWVGAEnvCfg,
    RoverRGBResnetEnvCfg,
    StereoDepthCfg,
)
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
class AAURoverSimpleDistillationEnvCfg(RoverDistillationEnvCfg):
    """RGB student environment for distilling an AAURoverEnvSimple-v0 height-map teacher."""

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
class AAURoverSimpleBevEnvCfg(RoverBevEnvCfg):
    """BEV perception environment driven by an AAURoverEnvSimple-v0 height-map policy."""

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
class AAURoverSimpleBevNoisyDepthEnvCfg(AAURoverSimpleBevEnvCfg):
    """AAURoverEnvSimpleBEV-v0 where the model sees ZED-like noisy depth; the BEV targets stay exact."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.depth = StereoDepthCfg()

POLICIES = os.path.join(os.path.dirname(__file__), "policies")

@configclass
class AAURoverSimpleBevDistillationEnvCfg(AAURoverSimpleBevNoisyDepthEnvCfg):
    """Height-map teacher and a student that sees the persistent BEV it builds from noisy depth (Phase 4)."""

    policy_bev: PolicyBevCfg = PolicyBevCfg(
        model_path=os.path.join(POLICIES, "bev_perception_noisy_depth.pt"),
        reliability_path=os.path.join(POLICIES, "bev_reliability_map.npz"),
    )

    def __post_init__(self):
        # The BEV task's observations plus the student's goal; the parent then switches the depth to noisy depth.
        self.observations = RoverBevDistillationObservationsCfg()
        super().__post_init__()

@configclass
class AAURoverSimpleBevDistillationPersistentEnvCfg(AAURoverSimpleBevDistillationEnvCfg):
    """Ablation: the persistent BEV without provenance and reliability."""

    def __post_init__(self):
        super().__post_init__()
        self.policy_bev.variant = "persistent"

@configclass
class AAURoverSimpleBevDistillationInstantaneousEnvCfg(AAURoverSimpleBevDistillationEnvCfg):
    """Ablation: the current frame's BEV only, without memory."""

    def __post_init__(self):
        super().__post_init__()
        self.policy_bev.variant = "instantaneous"

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
        self.sim.physics.enable_external_forces_every_iteration = False

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
        self.sim.physics.enable_external_forces_every_iteration = False

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
        
