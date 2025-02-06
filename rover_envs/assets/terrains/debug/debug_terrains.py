import os

from isaaclab import sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from rover_envs.envs.navigation.utils.terrains.terrain_importer import RoverTerrainImporter


@configclass
class DebugTerrainSceneCfg(InteractiveSceneCfg):

    # Hidden Terrain (merged terrain of ground and obstacles) for raycaster.
    # This is done because the raycaster doesn't work with multiple meshes
    hidden_terrain = AssetBaseCfg(
        prim_path="/World/terrain/hidden_terrain",
        spawn=sim_utils.UsdFileCfg(
            visible=False,
            usd_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "debug1/terrain_merged.usd",
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Ground Terrain
    terrain = TerrainImporterCfg(
        class_type=RoverTerrainImporter,
        prim_path="/World/terrain",
        terrain_type="usd",
        collision_group=-1,
        usd_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "debug1/terrain_only.usd",
        ),
    )

    # Obstacles
    obstacles = AssetBaseCfg(
        prim_path="/World/terrain/obstacles",
        spawn=sim_utils.UsdFileCfg(
            visible=True,
            usd_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "debug1/rocks_merged.usd",
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
