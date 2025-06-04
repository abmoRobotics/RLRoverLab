from terrain_importer import RoverTerrainImporter
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg

# TODO: FINISH THIS FILE AND IMPLEMENT IT IN THE TERRAIN IMPORTER
@configclass
class RoverTerrainImporterCfg(TerrainImporterCfg):
    """Configuration for the rover terrain importer.

    This configuration is used to create a terrain importer that samples targets from a terrain mesh.
    """

    class_type: type = RoverTerrainImporter
    """The class to use for the terrain importer."""

    target_distance: float = 9.0
    """The distance from the environment origin to sample targets."""

    num_spawn_locations: int = 2000
    """The number of spawn locations to generate."""

    target_distance_to_boundary: float = 7
    """The minimum distance in meters from the boundary of the terrain to the spawn locations."""

    spawn_distance_to_boundary: float = 15.0
    """The distance from each spawn location to the boundary of the terrain."""

    safety_margin_to_obstacles: float = 1.5
    """The minimum distance in meters from obstacles to spawn locations."""

    gradient_analysis_resolution: float = 0.05
    """The resolution in meters for gradient analysis of the terrain."""

    gradient_analysis_threshold: float = 0.35
    """The threshold for gradient analysis to determine if is suitable for spawning."""