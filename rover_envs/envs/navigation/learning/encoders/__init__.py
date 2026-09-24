"""Observation encoders shared by the navigation RL backends."""

# DinoV3Encoder is imported from .dinov3 directly to keep transformers out of the heightmap import path.
from .heightmap import ConvHeightmapEncoder, HeightmapEncoder  # noqa: F401
