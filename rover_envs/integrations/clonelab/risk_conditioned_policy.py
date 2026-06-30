from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .policy import CloneLabActorPolicy


RISK_PREFERENCE_KEY = "risk_preference"


def _validate_risk_preference(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Risk preference alpha must lie in [0, 1], got {value}.")
    return value


class _RiskPreferenceStateAdapter:
    """Append alpha after any online visual preprocessing has completed."""

    def __init__(self, wrapped_adapter, risk_preference: float, device: str | torch.device):
        self.wrapped_adapter = wrapped_adapter
        self.device = torch.device(device)
        self.risk_preference = _validate_risk_preference(risk_preference)

    def to_state(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.wrapped_adapter is not None:
            adapted = self.wrapped_adapter.to_state(state)
        else:
            adapted = dict(state)

        batch_size = self._batch_size(adapted)
        adapted[RISK_PREFERENCE_KEY] = torch.full(
            (batch_size, 1),
            self.risk_preference,
            dtype=torch.float32,
            device=self.device,
        )
        return adapted

    def set_risk_preference(self, value: float) -> None:
        self.risk_preference = _validate_risk_preference(value)

    @staticmethod
    def _batch_size(state: dict[str, torch.Tensor]) -> int:
        preferred_keys = ("proprioceptive", "dino_tokens", "image", "da_depth", "depth")
        for key in preferred_keys:
            value = state.get(key)
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
        raise ValueError("Could not infer the policy batch size while adding risk preference.")


class RiskConditionedCloneLabActorPolicy:
    """Add risk conditioning around the existing CloneLab runtime policy.

    This wrapper intentionally decorates an individual policy instance instead
    of modifying the baseline runtime integration.
    """

    def __init__(self, policy: CloneLabActorPolicy, risk_preference: float):
        self._policy = policy
        self._risk_adapter = _RiskPreferenceStateAdapter(
            wrapped_adapter=policy.online_adapter,
            risk_preference=risk_preference,
            device=policy.device,
        )
        self._policy.online_adapter = self._risk_adapter

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint: str | Path,
        risk_preference: float,
        device: str | torch.device,
        factory_spec: str | None = None,
        checkpoint_name: str = "final_model.pt",
        model_config: dict[str, Any] | None = None,
    ) -> "RiskConditionedCloneLabActorPolicy":
        policy = CloneLabActorPolicy.from_checkpoint(
            factory_spec=factory_spec,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            model_config=model_config,
            device=device,
        )
        return cls(policy, risk_preference)

    @property
    def actor(self) -> torch.nn.Module:
        return self._policy.actor

    @property
    def device(self) -> str | torch.device:
        return self._policy.device

    @property
    def proprioceptive_keys(self) -> tuple[str, ...]:
        return self._policy.proprioceptive_keys

    @property
    def risk_preference(self) -> float:
        return self._risk_adapter.risk_preference

    def set_risk_preference(self, value: float) -> None:
        self._risk_adapter.set_risk_preference(value)

    def act(self, state: dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        return self._policy.act(state, deterministic=deterministic)

    def reset(self, batch_size: int) -> None:
        self._policy.reset(batch_size)

    def reset_done(self, done: torch.Tensor) -> None:
        self._policy.reset_done(done)
