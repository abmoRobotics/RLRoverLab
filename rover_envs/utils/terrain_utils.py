"""Utility functions for terrain configuration in training/evaluation scripts."""

from pathlib import Path
from typing import Optional


_UNSET = object()


def handle_terrain_config(
    terrain_arg: Optional[str] = None,
    *,
    terrain_seed: Optional[int] | object = _UNSET,
    keep_terrain: bool | object = _UNSET,
):
    """
    Resolve the terrain CLI argument to a registered terrain name.

    Args:
        terrain_arg: Terrain argument from CLI, or None for the default terrain.
        terrain_seed: Compatibility argument for callers that support temporary
            generated terrains. Registered terrain names do not need cleanup.
        keep_terrain: Compatibility argument for callers that support temporary
            generated terrains. Registered terrain names do not need cleanup.

    Returns:
        Terrain name to pass to set_terrain(), or None if using the default.
        Callers that pass ``terrain_seed`` or ``keep_terrain`` receive
        ``(terrain_name, cleanup_path)`` for newer evaluator compatibility.
    """
    return_tuple = terrain_seed is not _UNSET or keep_terrain is not _UNSET
    if terrain_arg is None:
        return (None, None) if return_tuple else None
    if terrain_arg.lower() == "random":
        raise ValueError(
            "Random terrain generation is deprecated. Generate/register a terrain separately, "
            "then pass its registered name with --terrain."
        )
    return (terrain_arg, None) if return_tuple else terrain_arg


def cleanup_terrain(path: Optional[str | Path]) -> None:
    """Compatibility cleanup hook for temporary terrains.

    Registered terrains such as ``lunar_lte11`` do not create temporary files,
    so there is nothing to remove.
    """
    _ = path
