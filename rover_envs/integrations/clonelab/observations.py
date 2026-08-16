from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch


TEMPORARY_FRONT_HEIGHT_SCAN_MODES = {
    "height_scan_front_2p5m": 2.5,
    "height_scan_front_2m": 2.0,
}


@dataclass(frozen=True)
class CloneLabObservationConfig:
    """Mapping from RLRoverLab policy observations to CloneLab state tensors."""

    policy_group: str = "policy"
    visual_mode: str = "rgbd"
    image_key: str = "rgb_image"
    depth_key: str = "depth_image"
    height_scan_key: str = "height_scan"
    proprioceptive_keys: tuple[str, ...] = ("angle_diff", "distance", "heading")
    normalize_image: bool = False
    depth_nan_value: float = 6.0
    depth_min: float = 0.0
    depth_max: float = 6.0
    device: str | torch.device | None = None


class RoverToCloneLabObservation:
    """Convert RLRoverLab observations into CloneLab's canonical policy input.

    CloneLab policies used in this project expect:

    - ``proprioceptive``: ``(num_envs, features)``
    - ``image``: RGB tensor in channel-first format
    - ``depth``: depth tensor in channel-first format, clamped to the configured range
    """

    def __init__(self, cfg: CloneLabObservationConfig | None = None):
        self.cfg = cfg or CloneLabObservationConfig()

    def to_state(self, observations: Mapping[str, object]) -> dict[str, torch.Tensor]:
        policy_obs = self._policy_observations(observations)

        proprioceptive_parts = [
            self._as_feature_column(self._required_tensor(policy_obs, key)).float()
            for key in self.cfg.proprioceptive_keys
        ]
        proprioceptive = torch.cat(proprioceptive_parts, dim=1)

        if self._is_height_scan_mode(self.cfg.visual_mode):
            return {
                "proprioceptive": proprioceptive,
                "image": self._height_scan_image(
                    self._required_tensor(policy_obs, self.cfg.height_scan_key),
                    self.cfg.visual_mode,
                ),
            }

        image = self._channel_first(self._required_tensor(policy_obs, self.cfg.image_key)).float()
        if self.cfg.normalize_image:
            image = image / 255.0

        depth = self._channel_first(self._required_tensor(policy_obs, self.cfg.depth_key)).float()
        depth = torch.nan_to_num(
            depth,
            nan=self.cfg.depth_nan_value,
            posinf=self.cfg.depth_nan_value,
            neginf=self.cfg.depth_min,
        )
        depth = torch.clamp(depth, min=self.cfg.depth_min, max=self.cfg.depth_max)

        return {
            "proprioceptive": proprioceptive,
            "image": image,
            "depth": depth,
        }

    def num_envs(self, observations: Mapping[str, object]) -> int:
        policy_obs = self._policy_observations(observations)
        visual_keys = (self.cfg.height_scan_key,) if self._is_height_scan_mode(self.cfg.visual_mode) else (
            self.cfg.depth_key,
            self.cfg.image_key,
        )
        for key in (*visual_keys, *self.cfg.proprioceptive_keys):
            value = policy_obs.get(key)
            if value is not None:
                return int(torch.as_tensor(value).shape[0])
        raise KeyError("Could not infer num_envs from the RLRoverLab observation dictionary.")

    def _policy_observations(self, observations: Mapping[str, object]) -> Mapping[str, object]:
        if self.cfg.policy_group in observations and isinstance(observations[self.cfg.policy_group], Mapping):
            return observations[self.cfg.policy_group]  # type: ignore[return-value]
        return observations

    def _required_tensor(self, observations: Mapping[str, object], key: str) -> torch.Tensor:
        if key not in observations:
            available = ", ".join(sorted(observations.keys()))
            raise KeyError(f"Missing RLRoverLab observation key '{key}'. Available keys: {available}")
        value = observations[key]
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        if self.cfg.device is not None:
            tensor = tensor.to(self.cfg.device)
        return tensor

    @staticmethod
    def _as_feature_column(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor.reshape(1, 1)
        if tensor.ndim == 1:
            return tensor.unsqueeze(1)
        return tensor.reshape(tensor.shape[0], -1)

    @staticmethod
    def _channel_first(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor.unsqueeze(0).unsqueeze(0)

        if tensor.ndim == 3:
            # RLRoverLab task observations are vectorized, so this is usually
            # batched depth shape (B, H, W).
            return tensor.unsqueeze(1)

        if tensor.ndim != 4:
            raise ValueError(f"Expected image/depth tensor with 2 to 4 dims, got shape {tuple(tensor.shape)}")

        channels_first = tensor.shape[1] in (1, 3, 4)
        channels_last = tensor.shape[-1] in (1, 3, 4)

        if channels_last and not channels_first:
            return tensor.permute(0, 3, 1, 2).contiguous()
        if channels_first:
            return tensor.contiguous()

        return tensor.contiguous()

    @staticmethod
    def _is_height_scan_mode(visual_mode: str) -> bool:
        return visual_mode.lower() in {"height_scan", "heightmap", *TEMPORARY_FRONT_HEIGHT_SCAN_MODES}

    @staticmethod
    def _height_scan_image(tensor: torch.Tensor, visual_mode: str = "height_scan") -> torch.Tensor:
        tensor = tensor.float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim == 2:
            side = int(round(math.sqrt(int(tensor.shape[1]))))
            if side * side != int(tensor.shape[1]):
                raise ValueError(f"Expected flattened square height_scan, got shape {tuple(tensor.shape)}")
            tensor = tensor.reshape(tensor.shape[0], 1, side, side)
        elif tensor.ndim == 3:
            if tensor.shape[1] == 1:
                side = int(round(math.sqrt(int(tensor.shape[2]))))
                if side * side == int(tensor.shape[2]):
                    tensor = tensor.reshape(tensor.shape[0], 1, side, side)
            else:
                tensor = tensor.unsqueeze(1)
        elif tensor.ndim == 4:
            if tensor.shape[-1] == 1 and tensor.shape[1] != 1:
                tensor = tensor.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(f"Expected height_scan tensor with 1 to 4 dims, got shape {tuple(tensor.shape)}")

        tensor = torch.nan_to_num(tensor.contiguous(), nan=0.0, posinf=5.0, neginf=-5.0)
        tensor = torch.clamp(tensor, min=-5.0, max=5.0)
        return RoverToCloneLabObservation._crop_front_height_scan(tensor, visual_mode)

    @staticmethod
    def _crop_front_height_scan(tensor: torch.Tensor, visual_mode: str) -> torch.Tensor:
        mode = visual_mode.lower()
        if mode not in TEMPORARY_FRONT_HEIGHT_SCAN_MODES:
            return tensor

        # Temporary experiment: keep x from 0m forward to the requested range.
        # Isaac's GridPatternCfg uses xy ordering, so after row-major reshape the
        # width dimension is local x and the height dimension is local y.
        forward_m = TEMPORARY_FRONT_HEIGHT_SCAN_MODES[mode]
        resolution_m = 0.05
        center_col = tensor.shape[-1] // 2
        end_col = min(tensor.shape[-1], center_col + int(round(forward_m / resolution_m)) + 1)
        return tensor[..., center_col:end_col]
