"""Terrain assets and registry for rover environments."""

from .terrain_registry import (
    TerrainConfig,
    TerrainFiles,
    create_lethal_collision_cfg,
    create_lighting_cfg,
    create_obstacles_cfg,
    create_terrain_importer_cfg,
    discover_generated_terrains,
    discover_lunar_lte_terrains,
    get_terrain,
    get_terrain_choices,
    list_terrains,
    register_terrain,
    register_terrain_from_folder,
)

__all__ = [
    "TerrainConfig",
    "TerrainFiles",
    "register_terrain",
    "register_terrain_from_folder",
    "discover_generated_terrains",
    "discover_lunar_lte_terrains",
    "get_terrain",
    "list_terrains",
    "get_terrain_choices",
    "create_lethal_collision_cfg",
    "create_lighting_cfg",
    "create_obstacles_cfg",
    "create_terrain_importer_cfg",
]
