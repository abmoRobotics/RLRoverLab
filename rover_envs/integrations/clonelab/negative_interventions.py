"""Near-rock negative intervention utilities for offline RC-IQL data.

The interventions are intentionally policy-side/action-side wrappers. They do
not change the task, rewards, terminations, sensors, or recorder schema used by
normal teacher data collection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import torch


MODE_NONE = "none"
MODE_FAST_NEAR_ROCK = "fast_near_rock"
MODE_ACTION_NOISE = "action_noise"
MODE_STEERING_DELAY = "steering_delay"
MODE_STEERING_GAIN = "steering_gain"
MODE_HEIGHTMAP_BLIND = "heightmap_blind"
MODE_FAST_STEERING_NOISE = "fast_steering_noise"
MODE_FAST_STEERING_DELAY = "fast_steering_delay"
MODE_FAST_STEERING_GAIN = "fast_steering_gain"

NEGATIVE_INTERVENTION_MODES = (
    MODE_NONE,
    MODE_FAST_NEAR_ROCK,
    MODE_ACTION_NOISE,
    MODE_STEERING_DELAY,
    MODE_STEERING_GAIN,
    MODE_HEIGHTMAP_BLIND,
    MODE_FAST_STEERING_NOISE,
    MODE_FAST_STEERING_DELAY,
    MODE_FAST_STEERING_GAIN,
)
MODE_TO_ID = {name: index for index, name in enumerate(NEGATIVE_INTERVENTION_MODES)}
ID_TO_MODE = {index: name for name, index in MODE_TO_ID.items()}

NEGATIVE_INTERVENTION_STATE_ATTR = "_negative_intervention_state"


@dataclass(frozen=True)
class NearRockGateCfg:
    """Smooth gate that activates interventions only near rocks."""

    rover_radius_m: float = 0.5
    slow_start_body_clearance_m: float = 2.5
    full_strength_body_clearance_m: float = 1.0
    active_gate_threshold: float = 0.05


@dataclass(frozen=True)
class NegativeInterventionProfile:
    """Configuration for per-episode negative interventions."""

    name: str
    mode_weights: dict[str, float] = field(default_factory=dict)
    gate: NearRockGateCfg = field(default_factory=NearRockGateCfg)
    fast_forward_speed: float = 1.0
    steering_noise_std: float = 0.65
    throttle_noise_std: float = 0.10
    steering_delay_steps: int = 3
    steering_gain: float = 0.35
    action_clip_min: float = -1.0
    action_clip_max: float = 1.0

    def normalized_mode_weights(self) -> tuple[list[str], torch.Tensor]:
        if not self.mode_weights:
            raise ValueError("Negative intervention profile must define at least one mode weight.")

        modes: list[str] = []
        weights: list[float] = []
        for mode, weight in self.mode_weights.items():
            if mode not in MODE_TO_ID:
                raise ValueError(f"Unknown negative intervention mode {mode!r}.")
            if weight < 0:
                raise ValueError(f"Mode weight for {mode!r} must be non-negative, got {weight}.")
            if weight == 0:
                continue
            modes.append(mode)
            weights.append(float(weight))

        if not weights:
            raise ValueError("At least one negative intervention mode must have positive weight.")
        probabilities = torch.as_tensor(weights, dtype=torch.float32)
        probabilities = probabilities / probabilities.sum()
        return modes, probabilities


DEFAULT_NEGATIVE_PROFILES: dict[str, NegativeInterventionProfile] = {
    "mixed_action": NegativeInterventionProfile(
        name="mixed_action",
        mode_weights={
            MODE_FAST_NEAR_ROCK: 0.40,
            MODE_ACTION_NOISE: 0.25,
            MODE_STEERING_DELAY: 0.25,
            MODE_STEERING_GAIN: 0.10,
        },
    ),
    "mixed_with_blind": NegativeInterventionProfile(
        name="mixed_with_blind",
        mode_weights={
            MODE_FAST_NEAR_ROCK: 0.35,
            MODE_ACTION_NOISE: 0.25,
            MODE_STEERING_DELAY: 0.25,
            MODE_STEERING_GAIN: 0.10,
            MODE_HEIGHTMAP_BLIND: 0.05,
        },
    ),
    "fast_only": NegativeInterventionProfile(
        name="fast_only",
        mode_weights={MODE_FAST_NEAR_ROCK: 1.0},
    ),
    "collision_hunt": NegativeInterventionProfile(
        name="collision_hunt",
        mode_weights={
            MODE_FAST_STEERING_NOISE: 0.45,
            MODE_FAST_STEERING_DELAY: 0.35,
            MODE_FAST_STEERING_GAIN: 0.20,
        },
        steering_noise_std=1.0,
        throttle_noise_std=0.0,
        steering_delay_steps=4,
        steering_gain=0.10,
    ),
    "action_noise": NegativeInterventionProfile(
        name="action_noise",
        mode_weights={MODE_ACTION_NOISE: 1.0},
    ),
    "steering_delay": NegativeInterventionProfile(
        name="steering_delay",
        mode_weights={MODE_STEERING_DELAY: 1.0},
    ),
    "steering_gain": NegativeInterventionProfile(
        name="steering_gain",
        mode_weights={MODE_STEERING_GAIN: 1.0},
    ),
    "heightmap_blind": NegativeInterventionProfile(
        name="heightmap_blind",
        mode_weights={MODE_HEIGHTMAP_BLIND: 1.0},
    ),
}


def make_negative_intervention_profile(
    name: str,
    *,
    mode_weights: Mapping[str, float] | None = None,
) -> NegativeInterventionProfile:
    """Return a built-in profile, optionally with overridden mode weights."""

    try:
        profile = DEFAULT_NEGATIVE_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(DEFAULT_NEGATIVE_PROFILES))
        raise ValueError(f"Unknown negative profile {name!r}. Available profiles: {available}.") from exc

    if mode_weights is None:
        return profile
    return NegativeInterventionProfile(
        name=f"{name}_custom",
        mode_weights=dict(mode_weights),
        gate=profile.gate,
        fast_forward_speed=profile.fast_forward_speed,
        steering_noise_std=profile.steering_noise_std,
        throttle_noise_std=profile.throttle_noise_std,
        steering_delay_steps=profile.steering_delay_steps,
        steering_gain=profile.steering_gain,
        action_clip_min=profile.action_clip_min,
        action_clip_max=profile.action_clip_max,
    )


def near_rock_gate(center_distance_to_rock: torch.Tensor, cfg: NearRockGateCfg) -> torch.Tensor:
    """Return a smooth [0, 1] intervention strength from rock distance."""

    clearance = center_distance_to_rock.to(dtype=torch.float32).reshape(-1)
    body_clearance = torch.clamp(clearance - float(cfg.rover_radius_m), min=0.0)
    denom = float(cfg.slow_start_body_clearance_m - cfg.full_strength_body_clearance_m)
    if denom <= 0:
        raise ValueError("slow_start_body_clearance_m must exceed full_strength_body_clearance_m.")

    gate = torch.clamp((float(cfg.slow_start_body_clearance_m) - body_clearance) / denom, 0.0, 1.0)
    return gate * gate * (3.0 - 2.0 * gate)


class NegativeInterventionController:
    """Stateful per-environment intervention controller.

    Modes are sampled once per episode per vectorized environment. This avoids
    unrealistic step-to-step corruption while still producing a mixed shard.
    """

    def __init__(
        self,
        profile: NegativeInterventionProfile,
        *,
        num_envs: int,
        action_dim: int,
        device: str | torch.device,
        seed: int | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}.")

        self.profile = profile
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.mode_names, self.mode_probabilities = profile.normalized_mode_weights()
        self.mode_ids = torch.full((self.num_envs,), MODE_TO_ID[MODE_NONE], device=self.device, dtype=torch.int16)
        self.mode_counts = {mode: 0 for mode in NEGATIVE_INTERVENTION_MODES}
        self.action_rows = 0
        self.active_rows = 0
        self.near_rows = 0

        self._sample_generator = torch.Generator(device="cpu")
        if seed is not None:
            self._sample_generator.manual_seed(int(seed))
        self._noise_generator = torch.Generator(device=self.device)
        if seed is not None:
            self._noise_generator.manual_seed(int(seed) + 1)

        history_length = max(1, int(profile.steering_delay_steps))
        self._steering_history = torch.zeros((history_length, self.num_envs), device=self.device, dtype=torch.float32)
        self.reset_done(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))

    @property
    def has_heightmap_blind_mode(self) -> bool:
        return any(mode == MODE_HEIGHTMAP_BLIND for mode in self.mode_names)

    def reset_done(self, done: torch.Tensor) -> None:
        done = torch.as_tensor(done, device=self.device).bool().reshape(-1)
        if done.shape[0] != self.num_envs:
            raise ValueError(f"Expected done shape ({self.num_envs},), got {tuple(done.shape)}.")
        if not done.any():
            return

        indices = done.nonzero(as_tuple=False).reshape(-1)
        sampled = torch.multinomial(
            self.mode_probabilities,
            num_samples=int(indices.numel()),
            replacement=True,
            generator=self._sample_generator,
        )
        sampled_modes = [self.mode_names[int(index)] for index in sampled.tolist()]
        sampled_ids = torch.as_tensor(
            [MODE_TO_ID[mode] for mode in sampled_modes],
            device=self.device,
            dtype=torch.int16,
        )
        self.mode_ids[indices] = sampled_ids
        self._steering_history[:, indices] = 0.0
        for mode in sampled_modes:
            self.mode_counts[mode] += 1

    def intervention_gate(self, center_distance_to_rock: torch.Tensor) -> torch.Tensor:
        return near_rock_gate(center_distance_to_rock.to(device=self.device), self.profile.gate)

    def intervene(
        self,
        actions: torch.Tensor,
        center_distance_to_rock: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        input_was_1d = actions.ndim == 1
        actions = torch.as_tensor(actions, device=self.device)
        if input_was_1d:
            actions = actions.unsqueeze(0)
        if actions.ndim != 2 or actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected actions with shape ({self.num_envs}, A), got {tuple(actions.shape)}.")

        original = actions.clone()
        filtered = actions.clone()
        gate = self.intervention_gate(center_distance_to_rock)
        active_gate = gate > float(self.profile.gate.active_gate_threshold)

        mode_ids = self.mode_ids
        fast_mask = (
            (mode_ids == MODE_TO_ID[MODE_FAST_NEAR_ROCK])
            | (mode_ids == MODE_TO_ID[MODE_FAST_STEERING_NOISE])
            | (mode_ids == MODE_TO_ID[MODE_FAST_STEERING_DELAY])
            | (mode_ids == MODE_TO_ID[MODE_FAST_STEERING_GAIN])
        ) & active_gate
        noise_mask = (
            (mode_ids == MODE_TO_ID[MODE_ACTION_NOISE])
            | (mode_ids == MODE_TO_ID[MODE_FAST_STEERING_NOISE])
        ) & active_gate
        delay_mask = (
            (mode_ids == MODE_TO_ID[MODE_STEERING_DELAY])
            | (mode_ids == MODE_TO_ID[MODE_FAST_STEERING_DELAY])
        ) & active_gate
        gain_mask = (
            (mode_ids == MODE_TO_ID[MODE_STEERING_GAIN])
            | (mode_ids == MODE_TO_ID[MODE_FAST_STEERING_GAIN])
        ) & active_gate
        active_mask = fast_mask | noise_mask | delay_mask | gain_mask | (
            (mode_ids == MODE_TO_ID[MODE_HEIGHTMAP_BLIND]) & active_gate
        )

        if fast_mask.any():
            target_speed = torch.maximum(
                filtered[fast_mask, 0],
                torch.full_like(filtered[fast_mask, 0], float(self.profile.fast_forward_speed)),
            )
            g = gate[fast_mask].to(dtype=filtered.dtype)
            filtered[fast_mask, 0] = filtered[fast_mask, 0] + g * (target_speed - filtered[fast_mask, 0])

        if noise_mask.any():
            g = gate[noise_mask].to(dtype=filtered.dtype)
            if self.action_dim > 1:
                steering_noise = torch.randn(
                    (int(noise_mask.sum().item()),),
                    device=self.device,
                    generator=self._noise_generator,
                    dtype=filtered.dtype,
                )
                filtered[noise_mask, 1] = (
                    filtered[noise_mask, 1] + g * steering_noise * float(self.profile.steering_noise_std)
                )
            throttle_noise = torch.randn(
                (int(noise_mask.sum().item()),),
                device=self.device,
                generator=self._noise_generator,
                dtype=filtered.dtype,
            )
            filtered[noise_mask, 0] = (
                filtered[noise_mask, 0] + g * throttle_noise * float(self.profile.throttle_noise_std)
            )

        delayed_steering = self._steering_history[0].to(dtype=filtered.dtype)
        if delay_mask.any() and self.action_dim > 1:
            g = gate[delay_mask].to(dtype=filtered.dtype)
            filtered[delay_mask, 1] = filtered[delay_mask, 1] + g * (
                delayed_steering[delay_mask] - filtered[delay_mask, 1]
            )

        if gain_mask.any() and self.action_dim > 1:
            g = gate[gain_mask].to(dtype=filtered.dtype)
            target_steering = filtered[gain_mask, 1] * float(self.profile.steering_gain)
            filtered[gain_mask, 1] = filtered[gain_mask, 1] + g * (target_steering - filtered[gain_mask, 1])

        filtered = filtered.clamp(float(self.profile.action_clip_min), float(self.profile.action_clip_max))
        self._update_steering_history(original)

        self.action_rows += int(filtered.shape[0])
        self.near_rows += int(active_gate.sum().item())
        self.active_rows += int(active_mask.sum().item())

        metadata = {
            "mode_id": mode_ids.clone(),
            "gate": gate.detach().clone(),
            "active": active_mask.detach().clone(),
            "center_distance_to_rock": center_distance_to_rock.to(device=self.device, dtype=torch.float32).reshape(-1),
            "original_actions": original.detach().clone(),
            "action_delta": (filtered - original).detach().clone(),
        }
        if input_was_1d:
            filtered = filtered.squeeze(0)
        return filtered, metadata

    def summary(self) -> dict[str, Any]:
        return {
            "profile": asdict(self.profile),
            "mode_ids": MODE_TO_ID,
            "sampled_episode_modes": dict(self.mode_counts),
            "action_rows": self.action_rows,
            "near_rock_action_rows": self.near_rows,
            "active_intervention_rows": self.active_rows,
            "near_rock_fraction": self._ratio(self.near_rows, self.action_rows),
            "active_intervention_fraction": self._ratio(self.active_rows, self.action_rows),
        }

    def _update_steering_history(self, original_actions: torch.Tensor) -> None:
        if self.action_dim <= 1:
            return
        if self._steering_history.shape[0] > 1:
            self._steering_history[:-1] = self._steering_history[1:].clone()
        self._steering_history[-1] = original_actions[:, 1].detach().to(dtype=torch.float32)

    @staticmethod
    def _ratio(numerator: int | float, denominator: int | float) -> float | None:
        return float(numerator) / float(denominator) if denominator else None


def set_negative_intervention_state(env: Any, metadata: Mapping[str, torch.Tensor]) -> None:
    """Expose current intervention metadata to recorder terms."""

    base_env = getattr(env, "unwrapped", env)
    setattr(base_env, NEGATIVE_INTERVENTION_STATE_ATTR, dict(metadata))


def get_negative_intervention_state(env: Any) -> Mapping[str, torch.Tensor] | None:
    base_env = getattr(env, "unwrapped", env)
    return getattr(base_env, NEGATIVE_INTERVENTION_STATE_ATTR, None)


def blind_height_scan_observations(
    observations: Any,
    *,
    mode_ids: torch.Tensor,
    gate: torch.Tensor,
    height_scan_key: str = "height_scan",
) -> Any:
    """Return observations with height scans zeroed for active blind-mode envs.

    This is intended for policy input only. The environment and recorder should
    keep the original observations.
    """

    mode_ids = torch.as_tensor(mode_ids)
    gate = torch.as_tensor(gate, device=mode_ids.device, dtype=torch.float32).reshape(-1)
    blind = (mode_ids == MODE_TO_ID[MODE_HEIGHTMAP_BLIND]) & (gate > 0.0)
    if not bool(blind.any().item()):
        return observations

    def _copy_and_blind(value: Any, key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            return {child_key: _copy_and_blind(child_value, child_key) for child_key, child_value in value.items()}
        if key != height_scan_key or not isinstance(value, torch.Tensor):
            return value
        if value.shape[0] != blind.shape[0]:
            return value
        blinded = value.clone()
        row_gate = gate.to(device=blinded.device, dtype=blinded.dtype)
        row_mask = blind.to(device=blinded.device)
        scale_shape = (row_gate.shape[0],) + (1,) * (blinded.ndim - 1)
        scale = (1.0 - row_gate).reshape(scale_shape)
        blinded[row_mask] = blinded[row_mask] * scale[row_mask]
        return blinded

    return _copy_and_blind(observations)
