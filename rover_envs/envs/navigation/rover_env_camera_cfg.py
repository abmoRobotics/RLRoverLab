from isaaclab.sensors import MultiMeshRayCasterCfg, patterns
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
import isaaclab.sim as sim_utils
from rover_envs.assets.terrains import get_terrain
from rover_envs.envs.navigation.rover_env_cfg import ObservationCfg, RoverEnvCfg, RoverSceneCfg
# import
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
import rover_envs.envs.navigation.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.observations import image_features
import math

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
            func=image_features,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
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
        # Includes a 180-degree optical-axis roll so rendered frames are upright.
        offset=TiledCameraCfg.OffsetCfg(pos=(0.26294, -0.20045, 0.40189), rot=(0.53366, -0.39541, -0.4639, 0.58622),convention="opengl"),
        data_types=["rgb",  "depth"],
        # Zed 2i - WVGA - 672x376
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.12,
            horizontal_aperture=5.376, #4.416,
            vertical_aperture=3.008, #2.484,
            clipping_range=(0.1, 100),
        ),
        width=160,
        height=90,
    )

@configclass
class RoverZed2iWVGAFullEnvCfg(RoverSceneCfg):
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body/Zed2iWVGAFull_camera",
        # Same physical mount as the downsampled WVGA ZED2i camera.
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.26294, -0.20045, 0.40189),
            rot=(0.53366, -0.39541, -0.4639, 0.58622),
            convention="opengl",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.12,
            horizontal_aperture=5.376,
            vertical_aperture=3.008,
            clipping_range=(0.1, 100),
        ),
        width=672,
        height=376,
    )

@configclass
class RoverZed2iHD720EnvCfg(RoverSceneCfg):
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body/Zed2iHD720_camera",
        # Same physical mount as the WVGA ZED2i camera, rendered with HD720 intrinsics.
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.26294, -0.20045, 0.40189),
            rot=(0.53366, -0.39541, -0.4639, 0.58622),
            convention="opengl",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.12,
            horizontal_aperture=5.120,
            vertical_aperture=2.880,
            clipping_range=(0.1, 100),
        ),
        width=1280,
        height=720,
    )

@configclass
class RoverZed2iHD1080EnvCfg(RoverSceneCfg):
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body/Zed2iHD1080_camera",
        # Same physical mount as the WVGA ZED2i camera, rendered with HD1080 intrinsics.
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.26294, -0.20045, 0.40189),
            rot=(0.53366, -0.39541, -0.4639, 0.58622),
            convention="opengl",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.12,
            horizontal_aperture=3.840,
            vertical_aperture=2.160,
            clipping_range=(0.1, 100),
        ),
        width=1920,
        height=1080,
    )

@configclass
class RoverZed2iWVGAEnvCfgTEMP(RoverZed2iWVGAEnvCfg):
    """Temporary configuration for the Rover environment with ZED2i WVGA camera, will be used for learning by cheating"""

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.151, 0, 0.73428),
            # Includes a 180-degree optical-axis roll so rendered frames are upright.
            rot=(0.64086, -0.29884, -0.29884, 0.64086),
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
class RoverRGBDRawEnvCfg(RoverEnvCfg):
    """Configuration for the Rover environment with RGB-D raw observations, will be used for learning by cheating"""

    observations: RoverRGBDRawObservationsCfg = RoverRGBDRawObservationsCfg()
    scene: RoverZed2iWVGAEnvCfg = RoverZed2iWVGAEnvCfg(num_envs=8, env_spacing=4.0, replicate_physics=False)
    commands: CommandsNoVizCfg = CommandsNoVizCfg()

@configclass
class RoverRGBDRawWVGAEnvCfg(RoverEnvCfg):
    """Rover RGB-D raw observation task with ZED2i WVGA camera parameters."""

    observations: RoverRGBDRawObservationsCfg = RoverRGBDRawObservationsCfg()
    scene: RoverZed2iWVGAFullEnvCfg = RoverZed2iWVGAFullEnvCfg(num_envs=8, env_spacing=4.0, replicate_physics=False)
    commands: CommandsNoVizCfg = CommandsNoVizCfg()

@configclass
class RoverRGBDRawHD720EnvCfg(RoverEnvCfg):
    """Rover RGB-D raw observation task with ZED2i HD720 camera parameters."""

    observations: RoverRGBDRawObservationsCfg = RoverRGBDRawObservationsCfg()
    scene: RoverZed2iHD720EnvCfg = RoverZed2iHD720EnvCfg(num_envs=8, env_spacing=4.0, replicate_physics=False)
    commands: CommandsNoVizCfg = CommandsNoVizCfg()

@configclass
class RoverRGBDRawHD1080EnvCfg(RoverEnvCfg):
    """Rover RGB-D raw observation task with ZED2i HD1080 camera parameters."""

    observations: RoverRGBDRawObservationsCfg = RoverRGBDRawObservationsCfg()
    scene: RoverZed2iHD1080EnvCfg = RoverZed2iHD1080EnvCfg(num_envs=8, env_spacing=4.0, replicate_physics=False)
    commands: CommandsNoVizCfg = CommandsNoVizCfg()

@configclass
class RoverRGBDRawTempEnvCfg(RoverEnvCfg):
    """Temporary configuration for the Rover environment with RGB-D raw observations, will be used for learning by cheating"""

    observations: RoverRGBDRawObservationsCfg = RoverRGBDRawObservationsCfg()
    scene: RoverZed2iWVGAEnvCfgTEMP = RoverZed2iWVGAEnvCfgTEMP(num_envs=8, env_spacing=4.0, replicate_physics=False)
    commands: CommandsNoVizCfg = CommandsNoVizCfg()

@configclass
class RoverDistillationObservationsCfg(ObservationCfg):
    """Teacher height-map observations (`policy`) plus the student's goal and camera observations."""

    @configclass
    class ProprioCfg(ObsGroup):
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

    @configclass
    class RGBCfg(ObsGroup):
        rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb", "normalize": False},
        )

    proprio: ProprioCfg = ProprioCfg()
    rgb: RGBCfg = RGBCfg()

@configclass
class RoverDistillationEnvCfg(RoverEnvCfg):
    """Height-map teacher and ZED2i WVGA RGB student observations for teacher-student distillation."""

    observations: RoverDistillationObservationsCfg = RoverDistillationObservationsCfg()
    scene: RoverZed2iWVGAFullEnvCfg = RoverZed2iWVGAFullEnvCfg(num_envs=32, env_spacing=4.0, replicate_physics=False)
    commands: CommandsNoVizCfg = CommandsNoVizCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.tiled_camera.data_types = ["rgb"]
        # Render once per policy step instead of every physics step.
        self.sim.render_interval = self.decimation

@configclass
class BevCfg:
    """Ego BEV of the BEV perception task.

    The BEV frame is the yaw-aligned rover Body frame: origin at the Body origin, +x forward, +y left, +z up, metres.
    Maps have the layout [x, y]: cell (i, j) covers x in
    [x_range[0] + i * resolution, x_range[0] + (i + 1) * resolution) and likewise along y.
    """

    x_range: tuple[float, float] = (-2.0, 4.0)
    y_range: tuple[float, float] = (-3.0, 3.0)
    resolution: float = 0.05
    classes: tuple[str, ...] = ("ground", "rock")
    """Semantic classes, in the order of the BEV scanner's ray-cast targets (see RoverBevSceneCfg.set_terrain)."""

@configclass
class RoverBevSceneCfg(RoverZed2iWVGAFullEnvCfg):
    """ZED2i WVGA RGB-D camera plus one downward ray per BEV cell that labels its elevation and semantic class."""

    # The grid pattern and offset are set from BevCfg by RoverBevEnvCfg, the targets per terrain by set_terrain.
    bev_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body",
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(1.0, 1.0), ordering="yx"),
        mesh_prim_paths=[],
        update_mesh_ids=True,
        max_distance=100.0,
    )

    def set_terrain(self, terrain_name: str) -> None:
        super().set_terrain(terrain_name)
        # The hit target index is the semantic class: ground is the terrain mesh, rock the merged obstacle mesh.
        # TODO: take the class targets from the terrain registry (docs/specs/terrain_semantics.md).
        self.bev_scanner.mesh_prim_paths = ["/World/terrain/terrain", get_terrain(terrain_name).obstacle_mesh_prim_path]

@configclass
class RoverBevObservationsCfg(ObservationCfg):
    """Height-map observations for the driving policy (`policy`) plus the BEV model inputs and targets."""

    @configclass
    class RGBCfg(ObsGroup):
        rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb", "normalize": False},
        )

    @configclass
    class DepthCfg(ObsGroup):
        depth = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "depth", "normalize": False},
        )

    @configclass
    class CameraCfg(ObsGroup):
        """Intrinsics [9] followed by the camera pose in the BEV frame [12]."""

        intrinsics = ObsTerm(func=mdp.camera_intrinsics, params={"sensor_cfg": SceneEntityCfg("tiled_camera")})
        pose = ObsTerm(
            func=mdp.camera_pose_bev,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "frame_cfg": SceneEntityCfg("bev_scanner")},
        )

    @configclass
    class BevTargetCfg(ObsGroup):
        target = ObsTerm(
            func=mdp.bev_ground_truth,
            params={"sensor_cfg": SceneEntityCfg("bev_scanner"), "camera_cfg": SceneEntityCfg("tiled_camera")},
        )

    @configclass
    class PoseCfg(ObsGroup):
        """World pose of the ego BEV frame [x, y, z, yaw]: the odometry for the temporal BEV memory."""

        pose = ObsTerm(func=mdp.bev_frame_pose, params={"sensor_cfg": SceneEntityCfg("bev_scanner")})

    rgb: RGBCfg = RGBCfg()
    depth: DepthCfg = DepthCfg()
    camera: CameraCfg = CameraCfg()
    bev_target: BevTargetCfg = BevTargetCfg()
    pose: PoseCfg = PoseCfg()

@configclass
class StereoDepthCfg(ObsGroup):
    """ZED-like stereo depth for the BEV model input; the BEV targets keep using the exact rendered depth.

    With the 12 cm ZED baseline, fx = 265 px (WVGA) and 0.2 px disparity noise, the depth error is about 0.6 cm at
    1 m, 2.5 cm at 2 m and 10 cm at 4 m, correlated over 9 x 9 pixels, with 5% of the image in holes.
    """

    depth = ObsTerm(
        func=mdp.stereo_depth,
        params={
            "sensor_cfg": SceneEntityCfg("tiled_camera"),
            "baseline": 0.12,
            "disparity_std": 0.2,
            "correlation_px": 9,
            "dropout": 0.05,
        },
    )

@configclass
class RoverBevEnvCfg(RoverEnvCfg):
    """ZED2i WVGA RGB-D frames with BEV elevation and semantic targets, driven by the height-map policy."""

    bev: BevCfg = BevCfg()
    observations: RoverBevObservationsCfg = RoverBevObservationsCfg()
    scene: RoverBevSceneCfg = RoverBevSceneCfg(num_envs=32, env_spacing=4.0, replicate_physics=False)
    commands: CommandsNoVizCfg = CommandsNoVizCfg()

    def __post_init__(self):
        super().__post_init__()
        # Render once per policy step instead of every physics step.
        self.sim.render_interval = self.decimation
        # One ray per cell centre from 10 m above the Body; the grid pattern spans -size/2 to size/2 inclusive.
        (x_min, x_max), (y_min, y_max), resolution = self.bev.x_range, self.bev.y_range, self.bev.resolution
        self.scene.bev_scanner.offset.pos = ((x_min + x_max) / 2, (y_min + y_max) / 2, 10.0)
        self.scene.bev_scanner.pattern_cfg = patterns.GridPatternCfg(
            resolution=resolution, size=(x_max - x_min - resolution, y_max - y_min - resolution), ordering="yx"
        )
        self.scene.bev_scanner.update_period = self.sim.dt * self.decimation
