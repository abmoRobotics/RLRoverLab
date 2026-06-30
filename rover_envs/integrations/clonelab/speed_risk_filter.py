"""Policy-side speed filtering for risk-aware teacher data collection.

The filter is intentionally outside the rover task and action term
configuration.  It modifies the policy action before ``env.step()``, so the
existing action recorder sees the filtered command as the behavior action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch


@dataclass(frozen=True)
class SpeedRiskFilterProfile:
    name: str
    v_near: float
    v_far: float = 1.0
    rover_radius_m: float = 0.5
    slow_start_body_clearance_m: float = 2.5
    full_slow_body_clearance_m: float = 1.0


SPEED_RISK_FILTER_PROFILES = {
    "moderate": SpeedRiskFilterProfile(name="moderate", v_near=0.66),
    "conservative": SpeedRiskFilterProfile(name="conservative", v_near=0.33),
}


def infer_speed_risk_filter_profile(*labels: Any) -> SpeedRiskFilterProfile | None:
    """Infer the intended filter from existing teacher/checkpoint/dataset names."""

    text = " ".join(str(label or "") for label in labels).lower().replace("-", "_")
    compact = text.replace("_", "").replace(" ", "")

    if "aggressive" in text or "no_risk" in text or "norisk" in compact:
        return None
    if (
        "conservative" in text
        or "very_risk" in text
        or "risk_teacher_very" in text
        or "teacher_very" in text
        or "veryrisk" in compact
    ):
        return SPEED_RISK_FILTER_PROFILES["conservative"]
    if "moderate" in text:
        return SPEED_RISK_FILTER_PROFILES["moderate"]
    return None


def _base_env(env):
    return getattr(env, "unwrapped", env)


def _to_torch(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    import warp as wp

    return wp.to_torch(value)


def current_center_distance_to_rock(env, asset_name: str = "robot") -> torch.Tensor:
    """Return current rover-center distance to the projected rock map.

    Missing risk maps are treated as infinite clearance so the filter becomes a
    no-op on tasks without the projected-rock map.
    """

    base_env = _base_env(env)
    asset = base_env.scene[asset_name]
    xy = _to_torch(asset.data.root_link_pos_w)[:, :2]
    terrain_manager = getattr(base_env.scene.terrain, "_terrainManager", None)
    risk_map = getattr(terrain_manager, "obstacle_risk_map", None)
    if risk_map is None:
        return torch.full((xy.shape[0],), float("inf"), device=xy.device, dtype=torch.float32)

    return risk_map.distance_at_world_xy(xy).to(device=xy.device, dtype=torch.float32)


def speed_cap_from_center_distance(
    center_distance_to_rock: torch.Tensor,
    profile: SpeedRiskFilterProfile,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute normalized speed cap, body clearance, and smooth risk factor."""

    clearance = center_distance_to_rock.to(dtype=torch.float32)
    body_clearance = torch.clamp(clearance - profile.rover_radius_m, min=0.0)
    denom = profile.slow_start_body_clearance_m - profile.full_slow_body_clearance_m
    if denom <= 0:
        raise ValueError("slow_start_body_clearance_m must exceed full_slow_body_clearance_m.")

    risk = torch.clamp(
        (profile.slow_start_body_clearance_m - body_clearance) / denom,
        min=0.0,
        max=1.0,
    )
    risk = risk * risk * (3.0 - 2.0 * risk)
    cap = (1.0 - risk) * profile.v_far + risk * profile.v_near
    return cap, body_clearance, risk


def apply_speed_risk_filter(
    actions,
    center_distance_to_rock: torch.Tensor,
    profile: SpeedRiskFilterProfile | None,
):
    """Cap only positive forward speed; steering and reverse commands pass through."""

    if profile is None:
        return actions

    input_is_numpy = isinstance(actions, np.ndarray)
    action_tensor = torch.as_tensor(actions)
    original_ndim = action_tensor.ndim
    if original_ndim == 1:
        action_tensor = action_tensor.unsqueeze(0)
    if action_tensor.ndim != 2 or action_tensor.shape[-1] < 1:
        raise ValueError(f"Expected actions with shape [N, A], got {tuple(action_tensor.shape)}.")

    center_distance_to_rock = center_distance_to_rock.to(
        device=action_tensor.device,
        dtype=torch.float32,
    ).reshape(-1)
    if center_distance_to_rock.shape[0] != action_tensor.shape[0]:
        raise ValueError(
            "Expected one clearance value per environment, got "
            f"{center_distance_to_rock.shape[0]} clearances for {action_tensor.shape[0]} actions."
        )

    cap, _, _ = speed_cap_from_center_distance(center_distance_to_rock, profile)
    cap = cap.to(device=action_tensor.device, dtype=action_tensor.dtype)
    filtered = action_tensor.clone()
    filtered[:, 0] = torch.where(
        filtered[:, 0] > 0.0,
        torch.minimum(filtered[:, 0], cap),
        filtered[:, 0],
    )
    if original_ndim == 1:
        filtered = filtered.squeeze(0)
    if input_is_numpy:
        return filtered.detach().cpu().numpy()
    return filtered


@dataclass
class SpeedRiskFilterStats:
    profile: SpeedRiskFilterProfile | None
    action_rows: int = 0
    capped_action_rows: int = 0
    forward_raw_sum: float = 0.0
    forward_filtered_sum: float = 0.0
    near_1p5_rows: int = 0
    near_1p5_forward_filtered_sum: float = 0.0
    near_2p0_rows: int = 0
    near_2p0_forward_filtered_sum: float = 0.0
    high_speed_near_2p0_rows: int = 0

    def update(self, raw_actions, filtered_actions, center_distance_to_rock: torch.Tensor) -> None:
        raw = torch.as_tensor(raw_actions).detach()
        filtered = torch.as_tensor(filtered_actions).detach()
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        if filtered.ndim == 1:
            filtered = filtered.unsqueeze(0)

        center_distance_to_rock = center_distance_to_rock.detach().to(device=filtered.device).reshape(-1)
        raw_forward = raw[:, 0].to(device=filtered.device, dtype=torch.float32).clamp(min=0.0)
        filtered_forward = filtered[:, 0].to(dtype=torch.float32).clamp(min=0.0)

        self.action_rows += int(filtered_forward.numel())
        self.capped_action_rows += int((raw_forward > filtered_forward + 1e-6).sum().item())
        self.forward_raw_sum += float(raw_forward.sum().item())
        self.forward_filtered_sum += float(filtered_forward.sum().item())

        near_1p5 = center_distance_to_rock < 1.5
        self.near_1p5_rows += int(near_1p5.sum().item())
        self.near_1p5_forward_filtered_sum += float(filtered_forward[near_1p5].sum().item())

        near_2p0 = center_distance_to_rock < 2.0
        self.near_2p0_rows += int(near_2p0.sum().item())
        self.near_2p0_forward_filtered_sum += float(filtered_forward[near_2p0].sum().item())
        self.high_speed_near_2p0_rows += int(((filtered_forward > 0.66) & near_2p0).sum().item())

    def summary(self) -> dict[str, Any]:
        if self.profile is None:
            return {
                "enabled": False,
                "profile": None,
            }
        profile_dict = asdict(self.profile)
        return {
            "enabled": True,
            "profile": profile_dict,
            "action_rows": self.action_rows,
            "capped_action_fraction": self._ratio(self.capped_action_rows, self.action_rows),
            "mean_raw_forward_speed": self._ratio(self.forward_raw_sum, self.action_rows),
            "mean_filtered_forward_speed": self._ratio(self.forward_filtered_sum, self.action_rows),
            "near_1p5_action_rows": self.near_1p5_rows,
            "mean_filtered_forward_speed_near_1p5m": self._ratio(
                self.near_1p5_forward_filtered_sum,
                self.near_1p5_rows,
            ),
            "near_2p0_action_rows": self.near_2p0_rows,
            "mean_filtered_forward_speed_near_2p0m": self._ratio(
                self.near_2p0_forward_filtered_sum,
                self.near_2p0_rows,
            ),
            "high_speed_near_2p0_fraction": self._ratio(
                self.high_speed_near_2p0_rows,
                self.near_2p0_rows,
            ),
        }

    @staticmethod
    def _ratio(numerator: float | int, denominator: float | int) -> float | None:
        return float(numerator) / float(denominator) if denominator else None


class SpeedRiskActionWrapper(gym.ActionWrapper):
    """Gym wrapper that applies the speed-risk filter before the base env step."""

    def __init__(self, env, profile: SpeedRiskFilterProfile | None):
        super().__init__(env)
        self.profile = profile
        self.speed_filter_stats = SpeedRiskFilterStats(profile)

    def action(self, action):
        if self.profile is None:
            return action
        center_distance = current_center_distance_to_rock(self.env)
        filtered = apply_speed_risk_filter(action, center_distance, self.profile)
        self.speed_filter_stats.update(action, filtered, center_distance)
        return filtered
