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
class RoverCameraSceneCfg(RoverSceneCfg):

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.151, 0, 0.73428), rot=(0.57923, 0.40558, -0.40558, -0.57923),convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            horizontal_aperture=4.416,
            vertical_aperture=2.484,
            clipping_range=(0.1, 100),
        ),
        width=224,
        height=224,
    )

    height_scanner = None


@configclass
class RoverRGBResnetEnvCfg(RoverEnvCfg):

    observations: RoverResNetObservationsCfg = RoverResNetObservationsCfg()
    scene: RoverCameraSceneCfg = RoverCameraSceneCfg(num_envs=8, env_spacing=4.0, replicate_physics=False)

@configclass
class RoverCosmosEnvCfg(RoverEnvCfg):

    observations: RoverCosmosObservationsCfg = RoverCosmosObservationsCfg()
    scene: RoverCameraSceneCfg = RoverCameraSceneCfg(num_envs=8, env_spacing=4.0, replicate_physics=False)