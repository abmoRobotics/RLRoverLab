from __future__ import annotations


import importlib
import json
from pathlib import Path
from typing import Any


import torch
import logging




logger = logging.getLogger(__name__)

DEFAULT_ACTOR_CONFIG: dict[str, Any] = {
    "proprioception_channels": 3,
    "image_channels": 3,
    "depth_channels": 1,
    "action_dim": 2,
    "mlp_features": [512, 256, 128, 64],
    "image_input_dim": [160, 90],
    "image_encoder_features": [8, 16, 32, 64],
    "image_fc_features": [160, 120, 60],
    "activation": "leaky_relu",
    "dropout_rate": 0,
    "use_batch_norm": False,
}




def load_export_config(checkpoint: str | Path) -> dict[str, Any] | None:
    """Load CloneLab export_config.json next to a checkpoint directory or file."""


    path = Path(checkpoint)
    candidates = [path / "export_config.json"] if path.is_dir() else [
        path.parent / "export_config.json",
        path.parent.parent / "export_config.json",
    ]
    export_config_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if export_config_path is None:
        return None
