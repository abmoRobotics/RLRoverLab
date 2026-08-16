from __future__ import annotations

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .risk_map import ObstacleRiskMapCfg


@configclass
class RoverTerrainImporterCfg(TerrainImporterCfg):
    """Terrain importer config with rover-specific spawn analysis inputs."""

    spawn_obstacle_mesh_prim_path: str | None = None
    """Runtime mesh prim used to build obstacle masks for spawn generation."""

    obstacle_risk_cfg: ObstacleRiskMapCfg | None = None
    """Configuration for the smooth obstacle-risk reward map."""
