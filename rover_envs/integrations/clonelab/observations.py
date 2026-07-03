from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class CloneLabObservationConfig:
    """Mapping from RLRoverLab policy observations to CloneLab state tensors."""

    policy_group: str = "policy"
    image_key: str = "rgb_image"
    depth_key: str | None = "depth_image"
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

        image = self._channel_first(self._required_tensor(policy_obs, self.cfg.image_key)).float()
        if self.cfg.normalize_image:
            image = image / 255.0

        proprioceptive_parts = [
            self._as_feature_column(self._required_tensor(policy_obs, key)).float()
            for key in self.cfg.proprioceptive_keys
        ]
        proprioceptive = torch.cat(proprioceptive_parts, dim=1)

        state = {
            "proprioceptive": proprioceptive,
            "image": image,
        }

        if self.cfg.depth_key is not None and self.cfg.depth_key in policy_obs:
            depth = self._channel_first(self._required_tensor(policy_obs, self.cfg.depth_key)).float()
            depth = torch.nan_to_num(
                depth,
                nan=self.cfg.depth_nan_value,
                posinf=self.cfg.depth_nan_value,
                neginf=self.cfg.depth_min,
            )
            state["depth"] = torch.clamp(depth, min=self.cfg.depth_min, max=self.cfg.depth_max)

        return state

    def num_envs(self, observations: Mapping[str, object]) -> int:
        policy_obs = self._policy_observations(observations)
        keys = [self.cfg.image_key, *self.cfg.proprioceptive_keys]
        if self.cfg.depth_key is not None:
            keys.insert(0, self.cfg.depth_key)
        for key in keys:
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
