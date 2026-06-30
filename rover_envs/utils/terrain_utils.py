"""Utility functions for terrain configuration in training/evaluation scripts."""

from typing import Optional


def handle_terrain_config(terrain_arg: Optional[str]) -> Optional[str]:
    """
    Resolve the terrain CLI argument to a registered terrain name.

    Args:
        terrain_arg: Terrain argument from CLI, or None for the default terrain.

    Returns:
        Terrain name to pass to set_terrain(), or None if using the default.
    """
    if terrain_arg is None:
        return None
    if terrain_arg.lower() == "random":
        raise ValueError(
            "Random terrain generation is deprecated. Generate/register a terrain separately, "
            "then pass its registered name with --terrain."
        )
    return terrain_arg
