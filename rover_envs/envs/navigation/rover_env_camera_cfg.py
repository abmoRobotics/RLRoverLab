from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
import isaaclab.sim as sim_utils
from rover_envs.envs.navigation.rover_env_cfg import RoverEnvCfg, RoverSceneCfg
# import
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
import rover_envs.envs.navigation.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
import math
from ...mdp.observations import extended_image_features as extended_image_features

# ZED2i Camera Aperture Specifications:
# Resolution   Size         Pixel Size  H_Aperture  V_Aperture
# HD2K         2208×1242    0.002       4.416       2.484
# HD1080       1920×1080    0.002       3.840       2.160
# HD720        1280×720     0.004       5.120       2.880
# WVGA         672×376      0.008       5.376       3.008

@configclass
class RoverResNetObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        distance_to_target = ObsTerm(
            func=mdp.distance_to_target_euclidean,
            params={"command_name": "target_pose"},
            scale=0.11
        )
        heading_to_target = ObsTerm(
            func=mdp.angle_to_target_observation,
            params={"command_name": "target_pose"},
            scale=1 / math.pi
        )
        angle_difference_to_target = ObsTerm(
            func=mdp.angle_diff,
            params={"command_name": "target_pose"},
            scale=1 / math.pi
        )
        image_resnet_features = ObsTerm(
            func=extended_image_features,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",  # Can use any of the Cosmos models
            },
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()


@configclass
class RoverCosmosObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        distance_to_target = ObsTerm(
            func=mdp.distance_to_target_euclidean,
            params={"command_name": "target_pose"},
            scale=0.11
        )
        heading_to_target = ObsTerm(
            func=mdp.angle_to_target_observation,
            params={"command_name": "target_pose"},
            scale=1 / math.pi
        )
        angle_difference_to_target = ObsTerm(
            func=mdp.angle_diff,
            params={"command_name": "target_pose"},
            scale=1 / math.pi
        )
        image_cosmos_features = ObsTerm(
            func=extended_image_features,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "rgb",
                "model_name": "Cosmos-0.1-Tokenizer-CI8x8",  # Can use any of the Cosmos models
            },
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class RoverRGBDRawObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        distance = ObsTerm(
            func=mdp.distance_to_target_euclidean,
            params={"command_name": "target_pose"},
            scale=0.11
        )
        heading = ObsTerm(
            func=mdp.angle_to_target_observation,
            params={"command_name": "target_pose"},
            scale=1 / math.pi
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
        rgb_image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb"})
        depth_image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "depth"})
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False


    policy: PolicyCfg = PolicyCfg()

@configclass
class RoverCameraSceneCfg(RoverSceneCfg):

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.151, 0, 0.73428), rot=(0.57923, 0.40558, -0.40558, -0.57923),convention="opengl"),
        data_types=["rgb",  "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            horizontal_aperture=4.416,
            vertical_aperture=2.484,
            clipping_range=(0.1, 100),
        ),
        width=224,
        height=224,
    )

@configclass
class RoverZed2iWVGAEnvCfg(RoverSceneCfg):
    #belly_
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body/Zed2iWVGA_camera",
        #offset=TiledCameraCfg.OffsetCfg(pos=(0.27, -0.26, 0.4), rot=(0.9896, -0.13028, 0.06053, -0.00797),convention="opengl"),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.26294, -0.20045, 0.40189), rot=(0.58622, 0.4639, -0.39541, -0.53366),convention="opengl"),
        data_types=["rgb",  "depth"],
        # Zed 2i - WVGA - 672x376
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.12,
            horizontal_aperture=5.376, #4.416,
            vertical_aperture=3.008, #2.484,
            clipping_range=(0.1, 100),
        ),
        width=224,
        height=224,
    )

@configclass
class RoverZed2iWVGAEnvCfgTEMP(RoverZed2iWVGAEnvCfg):
    """Temporary configuration for the Rover environment with ZED2i WVGA camera, will be used for learning by cheating"""

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.151, 0, 0.73428),
            rot=(0.64086, 0.29884, -0.29884, -0.64086),
            convention="opengl",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.12,
            horizontal_aperture=5.376, 
            vertical_aperture=3.008,
            clipping_range=(0.01, 1000000.0),
        ),
        width=160,
        height=90,
    )


from rover_envs.envs.navigation.utils.terrains.terrain_importer import TerrainBasedPositionCommand  # noqa: F401
from rover_envs.envs.navigation.utils.terrains.commands_cfg import TerrainBasedPositionCommandCfg  # noqa: F401

@configclass
class CommandsNoVizCfg:
    """Command terms for the MDP."""

    target_pose = TerrainBasedPositionCommandCfg(
        class_type=TerrainBasedPositionCommand,  # TerrainBasedPositionCommandCustom,
        asset_name="robot",
        rel_standing_envs=0.0,
        simple_heading=False,
        resampling_time_range=(150.0, 150.0),
        ranges=TerrainBasedPositionCommandCfg.Ranges(
            heading=(-math.pi, math.pi)),
        debug_vis=False,
    )

@configclass
class RoverRGBResnetEnvCfg(RoverEnvCfg):

    observations: RoverResNetObservationsCfg = RoverResNetObservationsCfg()
    scene: RoverCameraSceneCfg = RoverCameraSceneCfg(num_envs=8, env_spacing=4.0, replicate_physics=False)
    scene.height_scanner = None

@configclass
class RoverCosmosEnvCfg(RoverEnvCfg):

    observations: RoverCosmosObservationsCfg = RoverCosmosObservationsCfg()
    scene: RoverCameraSceneCfg = RoverCameraSceneCfg(num_envs=8, env_spacing=4.0, replicate_physics=False)
    scene.height_scanner = None

@configclass
class RoverRGBDRawEnvCfg(RoverEnvCfg):
    """Configuration for the Rover environment with RGB-D raw observations, will be used for learning by cheating"""

    observations: RoverRGBDRawObservationsCfg = RoverRGBDRawObservationsCfg()
    scene: RoverZed2iWVGAEnvCfg = RoverZed2iWVGAEnvCfg(num_envs=8, env_spacing=4.0, replicate_physics=False)
    commands: CommandsNoVizCfg = CommandsNoVizCfg()

@configclass
class RoverRGBDRawTempEnvCfg(RoverEnvCfg):
    """Temporary configuration for the Rover environment with RGB-D raw observations, will be used for learning by cheating"""

    observations: RoverRGBDRawObservationsCfg = RoverRGBDRawObservationsCfg()
    scene: RoverZed2iWVGAEnvCfgTEMP = RoverZed2iWVGAEnvCfgTEMP(num_envs=8, env_spacing=4.0, replicate_physics=False)
    commands: CommandsNoVizCfg = CommandsNoVizCfg()



