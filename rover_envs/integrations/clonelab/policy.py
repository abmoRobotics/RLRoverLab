from __future__ import annotations

import contextlib
import importlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


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

    def __init__(
        self,
        actor: torch.nn.Module,
        device: str | torch.device,
        model_config: dict[str, Any] | None = None,
    ):
        model_config = model_config or {}
        self.actor = actor.to(device)
        self.actor.eval()
        self.device = device
        self.proprioceptive_keys = tuple(model_config.get("proprioceptive_keys", ("angle_diff", "distance", "heading")))
        self.online_adapter = self._build_online_adapter(model_config)

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
        return cls(actor, device, model_config)

    def act(self, state: dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        state = {key: value.to(self.device) for key, value in state.items()}
        if self.online_adapter is not None:
            state = self.online_adapter.to_state(state)
        with torch.inference_mode():
            if hasattr(self.actor, "get_action"):
                actions = self.actor.get_action(state, deterministic=deterministic)
            else:
                output = self.actor(state)
                actions = self._actions_from_output(output, deterministic)
        return actions.detach()

    def _build_online_adapter(self, model_config: dict[str, Any]):
        if not _is_dino_da_actor(self.actor):
            return None
        return OnlineDinoDAFeatureAdapter(model_config=model_config, device=self.device)

    def reset(self, batch_size: int) -> None:
        if hasattr(self.actor, "reset_hidden"):
            self.actor.reset_hidden(batch_size)

    def reset_done(self, done: torch.Tensor) -> None:
        hidden = getattr(self.actor, "hidden_val", None)
        done = done.to(self.device).bool().reshape(-1)
        if not done.any():
            return

        if isinstance(hidden, tuple):
            self.actor.hidden_val = tuple(self._reset_hidden_tensor(value, done) for value in hidden)
        elif hidden is not None:
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


class OnlineDinoDAFeatureAdapter:
    """Convert live RLRoverLab RGB frames into cached-feature policy inputs.

    The DINO/DA student was trained on precomputed tensors:

    - dino_tokens: [B, 577, 384]
    - da_depth:    [B, 1, H, W]

    During live evaluation we compute those tensors online from RGB only. The
    simulator depth observation is deliberately ignored for this policy.
    """

    def __init__(self, model_config: dict[str, Any], device: str | torch.device):
        self.model_config = dict(model_config)
        self.device = torch.device(device)
        self.dino_model_id = self.model_config.get("dino_model", "facebook/dinov3-vits16-pretrain-lvd1689m")
        self.da3_model_id = self.model_config.get("da3_model", "depth-anything/DA3-SMALL")
        self.dino_size = (
            int(self.model_config.get("dino_input_width", 512)),
            int(self.model_config.get("dino_input_height", 288)),
        )
        self.da_depth_hw = (
            int(self.model_config.get("depth_height", 72)),
            int(self.model_config.get("depth_width", 128)),
        )
        self.da3_process_res = int(self.model_config.get("da3_process_res", 504))
        self.da3_process_res_method = self.model_config.get("da3_process_res_method", "upper_bound_resize")
        self.show_model_logs = bool(self.model_config.get("show_model_logs", False))

        self.dino = self._load_dino()
        self.da3 = self._load_da3()
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def to_state(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "image" not in state:
            raise KeyError("DINO/DA online evaluation requires state['image'] from RLRoverLab RGB observations.")

        rgb = self._rgb_to_uint8_chw(state["image"])
        dino_tokens = self._extract_dino(rgb)
        da_depth = self._extract_da3(rgb)
        return {
            "dino_tokens": dino_tokens,
            "da_depth": da_depth,
            "proprioceptive": state["proprioceptive"].to(self.device, dtype=torch.float32),
        }

    def _load_dino(self) -> torch.nn.Module:
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "DINO/DA online evaluation requires transformers. Install it in the RLRoverLab runtime with:\n"
                "  /isaac-sim/python.sh -m pip install 'transformers>=4.56' huggingface_hub safetensors"
            ) from exc

        try:
            model = AutoModel.from_pretrained(self.dino_model_id)
        except OSError as exc:
            raise RuntimeError(
                f"Could not load DINO model {self.dino_model_id!r}. If the model is gated, request access "
                "on Hugging Face and authenticate inside the RLRoverLab container with:\n"
                "  /isaac-sim/python.sh -c 'from huggingface_hub import login; login()'"
            ) from exc
        return model.to(self.device).eval()

    def _load_da3(self):
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise RuntimeError(
                "DINO/DA online evaluation requires Depth Anything 3. Install it in the RLRoverLab runtime with:\n"
                "  /isaac-sim/python.sh -m pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
            ) from exc

        model = DepthAnything3.from_pretrained(self.da3_model_id)
        return model.to(device=self.device).eval()

    def _rgb_to_uint8_chw(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.device)
        if image.ndim != 4:
            raise ValueError(f"Expected RGB image [B, C, H, W], got {tuple(image.shape)}")
        if image.shape[1] < 3:
            raise ValueError(f"Expected at least 3 RGB channels, got {tuple(image.shape)}")
        image = image[:, :3].detach()
        if image.dtype != torch.uint8:
            max_value = float(image.max().item()) if image.numel() else 0.0
            if max_value <= 1.5:
                image = image * 255.0
            image = image.clamp(0, 255).to(torch.uint8)
        return image

    @torch.inference_mode()
    def _extract_dino(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        pixels = rgb_uint8.float() / 255.0
        pixels = F.interpolate(pixels, size=(self.dino_size[1], self.dino_size[0]), mode="bicubic", align_corners=False)
        pixels = (pixels - self.mean) / self.std
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            outputs = self.dino(pixel_values=pixels)
        tokens = _strip_dino_register_tokens(outputs.last_hidden_state)
        expected = (rgb_uint8.shape[0], 577, 384)
        if tuple(tokens.shape) != expected:
            raise RuntimeError(f"Expected online DINO tokens {expected}, got {tuple(tokens.shape)}.")
        return tokens.float()

    @torch.inference_mode()
    def _extract_da3(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        rgb_hwc = rgb_uint8.permute(0, 2, 3, 1).detach().cpu().numpy()
        depths: list[torch.Tensor] = []
        for image in rgb_hwc:
            with _maybe_suppress_output(enabled=not self.show_model_logs):
                prediction = self._run_da3_inference(image)
            depth = torch.as_tensor(np.asarray(prediction.depth), dtype=torch.float32, device=self.device)
            if depth.ndim == 2:
                depth = depth.unsqueeze(0).unsqueeze(0)
            elif depth.ndim == 3:
                depth = depth.unsqueeze(1)
            else:
                raise RuntimeError(f"Expected DA3 depth [H, W] or [N, H, W], got {tuple(depth.shape)}.")
            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth = F.interpolate(depth, size=self.da_depth_hw, mode="bilinear", align_corners=False)
            depths.append(depth[0])
        return torch.stack(depths, dim=0)

    def _run_da3_inference(self, image: np.ndarray):
        kwargs = {
            "process_res": self.da3_process_res,
            "process_res_method": self.da3_process_res_method,
            "export_dir": None,
            "export_format": "mini_npz",
        }
        try:
            return self.da3.inference(image=[image], **kwargs)
        except TypeError:
            return self.da3.inference([image], **kwargs)


def _is_dino_da_actor(actor: torch.nn.Module) -> bool:
    return (
        hasattr(actor, "dino_token_count")
        and hasattr(actor, "depth_height")
        and hasattr(actor, "depth_width")
        and (
            hasattr(actor, "encode_visual")
            or hasattr(getattr(actor, "visual_encoder", None), "encode_visual")
        )
    )


def _strip_dino_register_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.shape[1] == 577:
        return tokens
    if tokens.shape[1] == 581:
        return torch.cat([tokens[:, :1], tokens[:, 5:]], dim=1)
    return tokens


@contextlib.contextmanager
def _maybe_suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
