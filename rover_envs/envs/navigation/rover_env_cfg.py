from __future__ import annotations

import math
import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm  # noqa: F401
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg  # noqa: F401
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg
from isaaclab.sim import SimulationCfg as SimCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg  # noqa: F401
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # noqa: F401

##
# Scene Description
##
import rover_envs
import rover_envs.envs.navigation.mdp as mdp
from rover_envs.assets.terrains.debug.debug_terrains import DebugTerrainSceneCfg  # noqa: F401
from rover_envs.assets.terrains.mars import MarsTerrainSceneCfg  # noqa: F401
from rover_envs.envs.navigation.utils.terrains.commands_cfg import TerrainBasedPositionCommandCfg  # noqa: F401
# from rover_envs.envs.navigation.utils.terrains.terrain_importer import TerrainBasedPositionCommandCustom  # noqa: F401
from rover_envs.envs.navigation.utils.terrains.terrain_importer import RoverTerrainImporter  # noqa: F401
from rover_envs.envs.navigation.utils.terrains.terrain_importer import TerrainBasedPositionCommand  # noqa: F401


@configclass
class RoverSceneCfg(MarsTerrainSceneCfg):
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

    robot: ArticulationCfg = MISSING
    # AAU_ROVER_SIMPLE_CFG.replace(
    #     prim_path="{ENV_REGEX_NS}/Robot")

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_(Drive|Steer|Boogie|Body|Rocker)",
        filter_prim_paths_expr=["/World/terrain/obstacles/obstacles"],
    )
    # contact_sensor = None

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body",
        offset=RayCasterCfg.OffsetCfg(pos=[0.0, 0.0, 10.0]),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[5.0, 5.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/terrain/hidden_terrain"],
        max_distance=100.0,
    )


@configclass
class ActionsCfg:
    """Action"""

    # We define the action space for the rover
    actions: ActionTerm = MISSING


@configclass
class ObservationCfg:
    """Observation configuration for the task."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        distance = ObsTerm(func=mdp.distance_to_target_euclidean, params={
                           "command_name": "target_pose"}, scale=0.11)
        heading = ObsTerm(
            func=mdp.angle_to_target_observation,
            params={
                "command_name": "target_pose",
            },
            scale=1 / math.pi,
        )
        angle_diff = ObsTerm(
            func=mdp.angle_diff,
            params={"command_name": "target_pose"},
            scale=1 / math.pi
        )
        height_scan = ObsTerm(
            func=mdp.height_scan_rover,
            scale=1,
            params={"sensor_cfg": SceneEntityCfg(name="height_scanner")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    distance_to_target = RewTerm(
        func=mdp.distance_to_target_reward,
        weight=5.0,
        params={"command_name": "target_pose"},
    )
    reached_target = RewTerm(
        func=mdp.reached_target,
        weight=5.0,
        params={"command_name": "target_pose", "threshold": 0.18},
    )
    oscillation = RewTerm(
        func=mdp.oscillation_penalty,
        weight=-0.05,
        params={},
    )
    angle_to_target = RewTerm(
        func=mdp.angle_to_target_penalty,
        weight=-1.5,
        params={"command_name": "target_pose"},
    )
    heading_soft_contraint = RewTerm(
        func=mdp.heading_soft_contraint,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg(name="robot")},
    )
    collision = RewTerm(
        func=mdp.collision_penalty,
        weight=-3.0,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_sensor"), "threshold": 1.0},
    )
    far_from_target = RewTerm(
        func=mdp.far_from_target_reward,
        weight=-2.0,
        params={"command_name": "target_pose", "threshold": 11.0},
    )
    angle_diff = RewTerm(
        func=mdp.angle_to_goal_reward,
        weight=5.0,
        params={"command_name": "target_pose"},
    )


@configclass
class TerminationsCfg:
    """Termination conditions for the task."""

    time_limit = DoneTerm(func=mdp.time_out, time_out=True)
    is_success = DoneTerm(
        func=mdp.is_success,
        params={"command_name": "target_pose", "threshold": 0.18},
    )
    far_from_target = DoneTerm(
        func=mdp.far_from_target,
        params={"command_name": "target_pose", "threshold": 11.0},
    )
    collision = DoneTerm(
        func=mdp.collision_with_obstacles,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_sensor"), "threshold": 1.0},
    )


# "mdp.illegal_contact
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    target_pose = TerrainBasedPositionCommandCfg(
        class_type=TerrainBasedPositionCommand,  # TerrainBasedPositionCommandCustom,
        asset_name="robot",
        rel_standing_envs=0.0,
        simple_heading=False,
        resampling_time_range=(150.0, 150.0),
        ranges=TerrainBasedPositionCommandCfg.Ranges(
            heading=(-math.pi, math.pi)),
        debug_vis=True,
    )


@configclass
class EventCfg:
    """Randomization configuration for the task."""
    # startup_state = RandTerm(
    #     func=mdp.reset_root_state_rover,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )
    reset_state = EventTerm(
        func=mdp.reset_root_state_rover,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )


# @configclass
# class CurriculumCfg:
#     """ Curriculum configuration for the task. """
#     target_distance = CurrTerm(func=mdp.goal_distance_curriculum)


@configclass
class RoverEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the rover environment."""

    # Create scene
    scene: RoverSceneCfg = RoverSceneCfg(
        num_envs=128, env_spacing=4.0, replicate_physics=False)

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
    events: EventCfg = EventCfg()

    # MDP Settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.sim.dt = 1 / 30.0
        self.decimation = 6
        self.episode_length_s = 150
        self.viewer.eye = (-6.0, -6.0, 3.5)

        # update sensor periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        if self.scene.contact_sensor is not None:
            self.scene.contact_sensor.update_period = self.sim.dt * self.decimation
