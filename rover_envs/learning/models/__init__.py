"""
General model utilities for the rover_envs package.

This module provides general model registration and factory utilities
that can be used across different domains (navigation, manipulation, etc.).
"""

from .base_models import get_activation
from .factory import ModelFactory
from .registry import MODEL_REGISTRY, register_model

# Import models from different domains to ensure they get registered
try:
    # Import navigation models to register them
    from rover_envs.envs.navigation.learning.skrl import models as navigation_models  # noqa: F401
except ImportError:
    # Navigation models not available
    pass

__all__ = ["MODEL_REGISTRY", "register_model", "ModelFactory", "get_activation"]
