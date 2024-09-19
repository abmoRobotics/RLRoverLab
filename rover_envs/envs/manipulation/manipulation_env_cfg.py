from __future__ import annotations

import os
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ActionTermCfg as ActionTerm  # noqa: F401
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm  # noqa: F401
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm  # noqa: F401
from omni.isaac.lab.managers import RandomizationTermCfg as RandTerm  # noqa: F401
from omni.isaac.lab.managers import RewardTermCfg as RewTerm  # noqa: F401
from omni.isaac.lab.managers import SceneEntityCfg  # noqa: F401
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm  # noqa: F401
from omni.isaac.lab.scene import InteractiveSceneCfg  # noqa: F401
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns  # noqa: F401
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sim import PhysxCfg
from omni.isaac.lab.sim import SimulationCfg as SimCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from omni.isaac.lab.terrains import TerrainImporter, TerrainImporterCfg  # noqa: F401
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: F401
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # noqa: F401

import rover_envs
import rover_envs.envs.manipulation.mdp as mdp  # noqa: F401
# from rover_envs.assets.terrains.debug import DebugTerrainSceneCfg  # noqa: F401
# from rover_envs.assets.terrains.mars import MarsTerrainSceneCfg  # noqa: F401
from rover_envs.envs.navigation.utils.terrains.terrain_importer import RoverTerrainImporter  # noqa: F401

# from rover_envs.envs.navigation.utils.terrains.terrain_importer import TerrainBasedPositionCommandCustom

##
# Scene Description
##


@configclass
class ManipulatorSceneCfg(InteractiveSceneCfg):
    """
    Rover Scene Configuration

    Note:
        Terrains can be changed by changing the parent class e.g.
        RoverSceneCfg(MarsTerrainSceneCfg) -> RoverSceneCfg(DebugTerrainSceneCfg)

    """

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color_temperature=4500.0,
            intensity=100,
            enable_color_temperature=True,
            texture_file=os.path.join(
                os.path.dirname(os.path.abspath(rover_envs.__path__[0])),
                "rover_envs",
                "assets",
                "textures",
                "background.png",
            ),
            texture_format="latlong",
        ),
    )

    sphere_light = AssetBaseCfg(
        prim_path="/World/SphereLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=30000.0, radius=50, color_temperature=5500, enable_color_temperature=True
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -180.0, 80.0)),
    )

    # Robot
    robot: ArticulationCfg = MISSING

    # End effector frame
    ee_frame: FrameTransformerCfg = MISSING

    # Target object
    object: RigidObjectCfg = MISSING

    # Table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    # )

    # Plane
    # plane = AssetBaseCfg(
    #     prim_path="/World/GroundPlane",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0.00]),
    #     spawn=GroundPlaneCfg(),
    # )

    # Plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0.0]),
        spawn=GroundPlaneCfg(),
    )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_(link1|link2|link3|link4|link5|link6|link7|hand)",
    )
    # contact_sensor = None


@configclass
class ActionsCfg:
    """Action"""
    body_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_joint_pos: mdp.JointPositionActionCfg = MISSING


@configclass
class ObservationCfg:
    """Observation configuration for the task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_pose = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.06}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.06, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.06, "command_name": "object_pose"},
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)

    # ee_closed_near_object = RewTerm(
    #     func=mdp.ee_closed_near_object,
    #     params={"std": 0.01},
    #     weight=0.5)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination conditions for the task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.base_height, params={
            "minimum_height": -0.05,
            "asset_cfg": SceneEntityCfg("object")
        }
    )

    table_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "threshold": 0.01,
            "sensor_cfg": SceneEntityCfg("contact_sensor")
        },
    )


# "mdp.illegal_contact
@configclass
class CommandsCfg:
    """Command configuration for the task."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # Will be filled in the environment cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
            # pos_x=(0.3, 0.7), pos_y=(-0.5, 0.0), pos_z=(0.15, 0.3), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class EventCfg:
    """Configuration for randomization of the task."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="rock"),
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 30000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-3, "num_steps": 30000}
    )


@configclass
class ManipulatorEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the rover environment."""

    # Create scene
    scene: ManipulatorSceneCfg = ManipulatorSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Setup PhysX Settings
    sim: SimCfg = SimCfg(
        physx=PhysxCfg(
            enable_stabilization=True,
            gpu_max_rigid_contact_count=8388608,
            gpu_max_rigid_patch_count=262144,
            gpu_found_lost_pairs_capacity=2**21,
            gpu_found_lost_aggregate_pairs_capacity=2**25,  # 2**21,
            gpu_total_aggregate_pairs_capacity=2**21,   # 2**13,
            gpu_max_soft_body_contacts=1048576,
            gpu_max_particle_contacts=1048576,
            gpu_heap_capacity=67108864,
            gpu_temp_buffer_capacity=16777216,
            gpu_max_num_partitions=8,
            gpu_collision_stack_size=2**28,
            friction_correlation_distance=0.025,
            friction_offset_threshold=0.04,
            bounce_threshold_velocity=2.0,
        )
    )

    # Basic Settings
    observations: ObservationCfg = ObservationCfg()
    actions: ActionsCfg = ActionsCfg()
    # randomization: RandomizationCfg = RandomizationCfg()
    events: EventCfg = EventCfg()

    # MDP Settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.sim.dt = 1 / 100.0  # 100 Hz
        self.decimation = 2
        self.episode_length_s = 5
        self.viewer.eye = (-6.0, -6.0, 3.5)

        if self.scene.contact_sensor is not None:
            self.scene.contact_sensor.update_period = self.sim.dt * self.decimation
