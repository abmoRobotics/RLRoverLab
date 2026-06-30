from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import torch


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

    with open(export_config_path, encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise TypeError(
            f"CloneLab export config must be a JSON object, got {type(config).__name__}"
        )
    return config


def load_object(spec: str):
    """Load an object from ``module:attribute`` or ``module.attribute`` syntax."""

    if ":" in spec:
        module_name, object_name = spec.split(":", 1)
    else:
        module_name, object_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def load_policy_config(path: str | None = None, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load CloneLab actor kwargs, layered on top of the local default config."""

    config = dict(defaults if defaults is not None else DEFAULT_ACTOR_CONFIG)
    if path is None:
        return config

    with open(path, encoding="utf-8") as file:
        user_config = json.load(file)
    if not isinstance(user_config, dict):
        raise TypeError(f"Policy config must be a JSON object, got {type(user_config).__name__}")
    config.update(user_config)
    return config


def build_actor(factory_spec: str, model_config: dict[str, Any], device: str | torch.device):
    factory = load_object(factory_spec)
    actor = factory(**model_config)
    return actor.to(device)


def _torch_load(path: Path, device: str | torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _checkpoint_candidates(checkpoint: str | Path, checkpoint_name: str) -> list[Path]:
    path = Path(checkpoint)
    if path.is_dir():
        return [
            path / "actor" / checkpoint_name,
            path / checkpoint_name,
        ]
    return [path]


def load_actor_state(actor: torch.nn.Module, checkpoint: str | Path, checkpoint_name: str, device: str | torch.device):
    candidates = _checkpoint_candidates(checkpoint, checkpoint_name)
    checkpoint_path = next((path for path in candidates if path.exists()), None)
    if checkpoint_path is None:
        formatted = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not find CloneLab actor checkpoint. Tried: {formatted}")

    state = _torch_load(checkpoint_path, device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    actor.load_state_dict(state)
    return checkpoint_path


class CloneLabActorPolicy:
    """Small runtime wrapper for CloneLab actor modules.

    RLRoverLab supplies the simulator and observations; CloneLab supplies the
    learned actor. This wrapper deliberately avoids depending on CloneLab
    trainers or environment launch code.
    """

    def __init__(self, actor: torch.nn.Module, device: str | torch.device):
        self.actor = actor.to(device)
        self.actor.eval()
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        factory_spec: str | None,
        checkpoint: str | Path,
        checkpoint_name: str,
        model_config: dict[str, Any] | None,
        device: str | torch.device,
    ) -> "CloneLabActorPolicy":
        export_config = load_export_config(checkpoint)
        if export_config is not None:
            factory_spec = factory_spec or export_config.get("model_factory")
            model_config = load_policy_config(
                None,
                defaults=export_config.get("model_config", {}),
            )
            checkpoint_name = export_config.get("checkpoint_name", checkpoint_name)

        if factory_spec is None:
            factory_spec = "Examples.isaaclab.models_cai:actor_gaussian_image"
        if model_config is None:
            model_config = load_policy_config()

        actor = build_actor(factory_spec, model_config, device)
        loaded_path = load_actor_state(actor, checkpoint, checkpoint_name, device)
        print(f"[INFO] Loaded CloneLab actor checkpoint: {loaded_path}")
        return cls(actor, device)

    def act(self, state: dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        state = {key: value.to(self.device) for key, value in state.items()}
        with torch.inference_mode():
            if hasattr(self.actor, "get_action"):
                actions = self.actor.get_action(state, deterministic=deterministic)
            else:
                output = self.actor(state)
                actions = self._actions_from_output(output, deterministic)
        return actions.detach()

    def reset(self, batch_size: int) -> None:
        if hasattr(self.actor, "reset_hidden"):
            self.actor.reset_hidden(batch_size)

    def reset_done(self, done: torch.Tensor) -> None:
        hidden = getattr(self.actor, "hidden_val", None)
        if hidden is None:
            return

        done = done.to(self.device).bool().reshape(-1)
        if not done.any():
            return

        if isinstance(hidden, tuple):
            self.actor.hidden_val = tuple(self._reset_hidden_tensor(value, done) for value in hidden)
        else:
            self.actor.hidden_val = self._reset_hidden_tensor(hidden, done)

    @staticmethod
    def _reset_hidden_tensor(hidden: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        hidden = hidden.clone()
        hidden[:, done, :] = 0
        return hidden

    @staticmethod
    def _actions_from_output(output, deterministic: bool) -> torch.Tensor:
        if hasattr(output, "mean") and hasattr(output, "sample"):
            return output.mean if deterministic else output.sample()
        if isinstance(output, tuple):
            first = output[0]
            if hasattr(first, "mean") and hasattr(first, "sample"):
                return first.mean if deterministic else first.sample()
            return first
        return output
