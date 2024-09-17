from dataclasses import MISSING

from omni.isaac.lab.envs.mdp.commands.commands_cfg import CommandTermCfg
from omni.isaac.lab.utils import configclass

from .terrain_importer import TerrainBasedPositionCommand

# TODO: THIS IS A TEMPORARY FIX, since terrain based command is changed in the Orbit API


@configclass
class TerrainBasedPositionCommandCfg(CommandTermCfg):
    """Configuration for the terrain-based position command generator."""

    class_type: type = TerrainBasedPositionCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    rel_standing_envs: float = MISSING
    """Probability threshold for environments where the robots that are standing still."""
    simple_heading: bool = MISSING
    """Whether to use simple heading or not.

    If True, the heading is in the direction of the target position.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the position commands."""
