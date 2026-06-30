"""Recorder config for isolated negative-example collection."""

from __future__ import annotations

import torch
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from rover_envs.mdp.recorders.compressed_rgbd_hdf5 import CompressedRGBDHDF5DatasetFileHandler
from rover_envs.mdp.recorders.recorders_cfg import (
    ActionRecorderCfg,
    DoneRecorderCfg,
    RockDistanceRecorderCfg,
    RoverAttitudeRecorderCfg,
    TerminalRecorderCfg,
    TimelineObservationRecorderCfg,
    TimeoutRecorderCfg,
    RewardRecorderCfg,
)

from .negative_interventions import get_negative_intervention_state


def _num_envs(env) -> int:
    return int(getattr(env, "num_envs", 1))


def _action_dim(env) -> int:
    action = getattr(getattr(env, "action_manager", None), "action", None)
    if isinstance(action, torch.Tensor) and action.ndim >= 2:
        return int(action.shape[-1])
    return 2


def _state_value(env, key: str, *, vector_default: float = 0.0, matrix_default: float = 0.0) -> torch.Tensor:
    state = get_negative_intervention_state(env)
    if state is not None and key in state:
        return torch.as_tensor(state[key]).detach()

    num_envs = _num_envs(env)
    device = getattr(env, "device", "cpu")
    if key in {"original_actions", "action_delta"}:
        return torch.full((num_envs, _action_dim(env)), matrix_default, device=device, dtype=torch.float32)
    if key == "mode_id":
        return torch.zeros((num_envs,), device=device, dtype=torch.int16)
    if key == "active":
        return torch.zeros((num_envs,), device=device, dtype=torch.bool)
    return torch.full((num_envs,), vector_default, device=device, dtype=torch.float32)


class NegativeModeRecorder(RecorderTerm):
    def record_pre_step(self):
        return "negative/mode_id", _state_value(self._env, "mode_id")


class NegativeGateRecorder(RecorderTerm):
    def record_pre_step(self):
        return "negative/gate", _state_value(self._env, "gate")


class NegativeActiveRecorder(RecorderTerm):
    def record_pre_step(self):
        return "negative/active", _state_value(self._env, "active")


class NegativeOriginalActionRecorder(RecorderTerm):
    def record_pre_step(self):
        return "negative/original_actions", _state_value(self._env, "original_actions")


class NegativeActionDeltaRecorder(RecorderTerm):
    def record_pre_step(self):
        return "negative/action_delta", _state_value(self._env, "action_delta")


class NegativeDecisionRockDistanceRecorder(RecorderTerm):
    def record_pre_step(self):
        return "negative/decision_center_distance_to_rock", _state_value(
            self._env,
            "center_distance_to_rock",
            vector_default=float("inf"),
        )


class TerminationMaskRecorder(RecorderTerm):
    def record_post_step(self):
        manager = getattr(self._env, "termination_manager", None)
        num_envs = _num_envs(self._env)
        device = getattr(self._env, "device", "cpu")
        if manager is None or self.cfg.term_name not in getattr(manager, "active_terms", ()):
            mask = torch.zeros((num_envs,), device=device, dtype=torch.bool)
        else:
            mask = torch.as_tensor(manager.get_term(self.cfg.term_name), device=device).bool().reshape(-1)
        return f"termination/{self.cfg.term_name}", mask


@configclass
class NegativeModeRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = NegativeModeRecorder


@configclass
class NegativeGateRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = NegativeGateRecorder


@configclass
class NegativeActiveRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = NegativeActiveRecorder


@configclass
class NegativeOriginalActionRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = NegativeOriginalActionRecorder


@configclass
class NegativeActionDeltaRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = NegativeActionDeltaRecorder


@configclass
class NegativeDecisionRockDistanceRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = NegativeDecisionRockDistanceRecorder


@configclass
class TerminationMaskRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = TerminationMaskRecorder
    term_name: str = ""


@configclass
class NegativeCompressedRGBDRecorderManagerCfg(RecorderManagerBaseCfg):
    """Compressed RGB-D recorder with extra negative-example metadata."""

    dataset_file_handler_class_type: type = CompressedRGBDHDF5DatasetFileHandler

    record_actions: ActionRecorderCfg = ActionRecorderCfg()
    record_rock_distance: RockDistanceRecorderCfg = RockDistanceRecorderCfg()
    record_rover_attitude: RoverAttitudeRecorderCfg = RoverAttitudeRecorderCfg()
    record_observation_timeline: TimelineObservationRecorderCfg = TimelineObservationRecorderCfg()
    record_rewards: RewardRecorderCfg = RewardRecorderCfg()
    record_dones: DoneRecorderCfg = DoneRecorderCfg()
    record_terminals: TerminalRecorderCfg = TerminalRecorderCfg()
    record_timeouts: TimeoutRecorderCfg = TimeoutRecorderCfg()

    record_negative_mode: NegativeModeRecorderCfg = NegativeModeRecorderCfg()
    record_negative_gate: NegativeGateRecorderCfg = NegativeGateRecorderCfg()
    record_negative_active: NegativeActiveRecorderCfg = NegativeActiveRecorderCfg()
    record_negative_original_action: NegativeOriginalActionRecorderCfg = NegativeOriginalActionRecorderCfg()
    record_negative_action_delta: NegativeActionDeltaRecorderCfg = NegativeActionDeltaRecorderCfg()
    record_negative_decision_distance: NegativeDecisionRockDistanceRecorderCfg = (
        NegativeDecisionRockDistanceRecorderCfg()
    )

    record_success_term: TerminationMaskRecorderCfg = TerminationMaskRecorderCfg(term_name="is_success")
    record_collision_term: TerminationMaskRecorderCfg = TerminationMaskRecorderCfg(term_name="collision")
    record_far_from_target_term: TerminationMaskRecorderCfg = TerminationMaskRecorderCfg(term_name="far_from_target")
    record_time_limit_term: TerminationMaskRecorderCfg = TerminationMaskRecorderCfg(term_name="time_limit")
