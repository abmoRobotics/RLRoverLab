"""
Terrain Registry - A centralized registry for terrain configurations.

This module provides a clean way to register and retrieve terrain configurations
that can be specified at runtime via command-line arguments.

Note: IsaacLab imports are deferred to avoid import order issues with SimulationApp.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Type hints only - these won't be imported at runtime
if TYPE_CHECKING:
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.terrains import TerrainImporterCfg


@dataclass
class TerrainFiles:
    """Dataclass holding paths to terrain USD files."""
    terrain_only: str  # Ground terrain without obstacles
    terrain_merged: str  # Combined terrain for raycaster
    rocks_merged: str  # Obstacles/rocks


@dataclass
class TerrainConfig:
    """Configuration for a terrain type."""
    name: str
    files: TerrainFiles
    description: str = ""


# Global terrain registry
_TERRAIN_REGISTRY: dict[str, TerrainConfig] = {}


def register_terrain(name: str, files: TerrainFiles, description: str = "") -> None:
    """Register a terrain configuration.
    
    Args:
        name: Unique identifier for the terrain (e.g., "mars", "debug")
        files: TerrainFiles dataclass with paths to USD files
        description: Optional description of the terrain
    """
    _TERRAIN_REGISTRY[name] = TerrainConfig(name=name, files=files, description=description)


def register_terrain_from_folder(
    name: str,
    folder: str,
    description: str = "",
    terrain_only: str = "terrain_only.usd",
    terrain_merged: str = "terrain_merged.usd",
    rocks_merged: str = "rocks_merged.usd",
) -> None:
    """Register a terrain from a folder with standard file naming.
    
    This is a convenience function that simplifies terrain registration when
    USD files follow the standard naming convention.
    
    Args:
        name: Unique identifier for the terrain (e.g., "mars", "debug")
        folder: Path to the folder containing the terrain USD files (absolute or relative to terrains dir)
        description: Optional description of the terrain
        terrain_only: Filename for ground terrain (default: "terrain_only.usd")
        terrain_merged: Filename for merged terrain used by raycaster (default: "terrain_merged.usd")
        rocks_merged: Filename for obstacles/rocks (default: "rocks_merged.usd")
    """
    # If folder is not absolute, treat it as relative to the terrains base path
    if not os.path.isabs(folder):
        folder = os.path.join(_TERRAINS_BASE_PATH, folder)
    
    files = TerrainFiles(
        terrain_only=os.path.join(folder, terrain_only),
        terrain_merged=os.path.join(folder, terrain_merged),
        rocks_merged=os.path.join(folder, rocks_merged),
    )
    register_terrain(name=name, files=files, description=description)


def get_terrain(name: str) -> TerrainConfig:
    """Get a registered terrain configuration.
    
    Args:
        name: Name of the terrain to retrieve
        
    Returns:
        TerrainConfig for the requested terrain
        
    Raises:
        KeyError: If terrain is not registered
    """
    if name not in _TERRAIN_REGISTRY:
        available = list(_TERRAIN_REGISTRY.keys())
        raise KeyError(f"Terrain '{name}' not found. Available terrains: {available}")
    return _TERRAIN_REGISTRY[name]


def list_terrains() -> list[str]:
    """List all registered terrain names."""
    return list(_TERRAIN_REGISTRY.keys())


def get_terrain_choices() -> list[str]:
    """Get terrain names for argparse choices."""
    return list(_TERRAIN_REGISTRY.keys())


def discover_generated_terrains() -> None:
    """Auto-discover and register terrains in the 'generated' folder."""
    generated_dir = os.path.join(_TERRAINS_BASE_PATH, "generated")
    if not os.path.exists(generated_dir):
        return
    
    for name in os.listdir(generated_dir):
        terrain_path = os.path.join(generated_dir, name)
        if not os.path.isdir(terrain_path):
            continue
        
        # Check if required files exist
        required_files = ["terrain_only.usd", "terrain_merged.usd", "rocks_merged.usd"]
        if all(os.path.exists(os.path.join(terrain_path, f)) for f in required_files):
            if name not in _TERRAIN_REGISTRY:
                register_terrain_from_folder(
                    name=name,
                    folder=f"generated/{name}",
                    description=f"Generated terrain: {name}",
                )


# ============================================================================
# Helper functions to create scene components from terrain config
# These functions use lazy imports to avoid SimulationApp import order issues
# ============================================================================

def create_hidden_terrain_cfg(terrain_config: TerrainConfig) -> "AssetBaseCfg":
    """Create hidden terrain configuration for raycaster."""
    # Lazy import to avoid SimulationApp requirement
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    
    return AssetBaseCfg(
        prim_path="/World/terrain/hidden_terrain",
        spawn=sim_utils.UsdFileCfg(
            visible=False,
            usd_path=terrain_config.files.terrain_merged,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


def create_obstacles_cfg(terrain_config: TerrainConfig) -> "AssetBaseCfg":
    """Create obstacles configuration."""
    # Lazy import to avoid SimulationApp requirement
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    
    return AssetBaseCfg(
        prim_path="/World/terrain/obstacles",
        spawn=sim_utils.UsdFileCfg(
            visible=True,
            usd_path=terrain_config.files.rocks_merged,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


def create_terrain_importer_cfg(terrain_config: TerrainConfig) -> "TerrainImporterCfg":
    """Create terrain importer configuration."""
    # Lazy import to avoid SimulationApp requirement
    from isaaclab.terrains import TerrainImporterCfg
    from rover_envs.envs.navigation.utils.terrains.terrain_importer import RoverTerrainImporter
    
    return TerrainImporterCfg(
        class_type=RoverTerrainImporter,
        prim_path="/World/terrain",
        terrain_type="usd",
        collision_group=-1,
        usd_path=terrain_config.files.terrain_only,
    )


# ============================================================================
# Register built-in terrains
# ============================================================================

_TERRAINS_BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Mars terrain
register_terrain_from_folder(
    name="mars",
    folder="mars/terrain1",
    description="Mars-like rocky terrain with obstacles",
)

# Debug terrain
register_terrain_from_folder(
    name="debug",
    folder="debug/debug1",
    description="Simple debug terrain for testing",
)

# Auto-discover generated terrains
discover_generated_terrains()