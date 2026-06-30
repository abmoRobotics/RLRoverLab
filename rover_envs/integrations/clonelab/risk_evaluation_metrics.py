from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Mapping, Sequence

import torch


CLEARANCE_DEFINITION = "rover_root_xy_to_projected_rock_map"
RECORDER_TERM_NAME = "risk_evaluation_state"


def clearance_cost(
    clearance: torch.Tensor,
    *,
    d_ref: float = 5.0,
    clearance_exponent: float = 2.5,
) -> torch.Tensor:
    if d_ref <= 0:
        raise ValueError("d_ref must be positive.")
    if clearance_exponent <= 0:
        raise ValueError("clearance_exponent must be positive.")
    if torch.any(clearance < 0):
        raise ValueError("Clearance must be nonnegative.")
    return torch.exp(
        -math.log(100.0)
        * (clearance.to(dtype=torch.float32) / float(d_ref)).pow(float(clearance_exponent))
    )


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"Quantile must lie in [0, 1], got {quantile}.")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class EpisodeRiskMetrics:
    episode_index: int
    success: bool
    collision: bool
    termination_reasons: list[str]
    episode_return: float
    episode_steps: int
    episode_duration_s: float
    minimum_clearance_m: float
    risk_exposure: float
    risk_exposure_per_second: float
    traveled_path_length_m: float
    start_goal_distance_m: float
    path_efficiency: float | None
    time_to_goal_s: float | None
    fraction_steps_below_0p55m: float
    fraction_steps_below_0p75m: float
    fraction_steps_below_1m: float
    fraction_steps_below_2m: float

    def to_dict(self) -> dict:
        return asdict(self)


class RiskEvaluationMetrics:
    """Accumulate vectorized episode metrics without entering policy inputs."""

    def __init__(
        self,
        num_envs: int,
        step_dt: float,
        *,
        device: str | torch.device,
        d_ref: float = 5.0,
        clearance_exponent: float = 2.5,
        target_episodes: int | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if step_dt <= 0:
            raise ValueError("step_dt must be positive.")
        if target_episodes is not None and target_episodes <= 0:
            raise ValueError("target_episodes must be positive when provided.")

        self.num_envs = int(num_envs)
        self.step_dt = float(step_dt)
        self.device = torch.device(device)
        self.d_ref = float(d_ref)
        self.clearance_exponent = float(clearance_exponent)
        self.target_episodes = target_episodes
        self.records: list[EpisodeRiskMetrics] = []

        self._active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._previous_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._start_goal_distance = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._path_length = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._minimum_clearance = torch.full(
            (self.num_envs,),
            float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        self._risk_exposure = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._below_0p55m = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._below_0p75m = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._below_1m = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._below_2m = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    @property
    def completed_episodes(self) -> int:
        return len(self.records)

    @property
    def target_reached(self) -> bool:
        return self.target_episodes is not None and self.completed_episodes >= self.target_episodes

    def start_episodes(
        self,
        positions_xy: torch.Tensor,
        goal_distances: torch.Tensor,
        env_mask: torch.Tensor | None = None,
    ) -> None:
        positions_xy = self._vector(positions_xy, trailing_shape=(2,))
        goal_distances = self._vector(goal_distances)
        mask = self._mask(env_mask)

        self._active[mask] = True
        self._previous_xy[mask] = positions_xy[mask]
        self._start_goal_distance[mask] = goal_distances[mask]
        self._reset_accumulators(mask)

    def update(
        self,
        *,
        final_positions_xy: torch.Tensor,
        decision_clearance: torch.Tensor,
        final_clearance: torch.Tensor,
        rewards: torch.Tensor,
        done: torch.Tensor,
        termination_masks: Mapping[str, torch.Tensor],
    ) -> list[EpisodeRiskMetrics]:
        final_positions_xy = self._vector(final_positions_xy, trailing_shape=(2,))
        decision_clearance = self._vector(decision_clearance)
        final_clearance = self._vector(final_clearance)
        rewards = self._vector(rewards)
        done = self._mask(done)
        if not torch.all(self._active):
            inactive = (~self._active).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(f"Risk metrics received updates for inactive environments: {inactive}")

        self._path_length += torch.linalg.vector_norm(final_positions_xy - self._previous_xy, dim=-1)
        self._previous_xy.copy_(final_positions_xy)
        self._returns += rewards
        self._steps += 1
        self._minimum_clearance = torch.minimum(
            self._minimum_clearance,
            torch.minimum(decision_clearance, final_clearance),
        )
        self._risk_exposure += clearance_cost(
            decision_clearance,
            d_ref=self.d_ref,
            clearance_exponent=self.clearance_exponent,
        )
        self._below_0p55m += (decision_clearance < 0.55).long()
        self._below_0p75m += (decision_clearance < 0.75).long()
        self._below_1m += (decision_clearance < 1.0).long()
        self._below_2m += (decision_clearance < 2.0).long()

        normalized_terminations = {
            name: self._mask(mask)
            for name, mask in termination_masks.items()
        }
        completed: list[EpisodeRiskMetrics] = []
        for env_id in done.nonzero(as_tuple=False).flatten().tolist():
            if self.target_episodes is None or len(self.records) < self.target_episodes:
                record = self._finish_episode(env_id, normalized_terminations)
                self.records.append(record)
                completed.append(record)
            self._active[env_id] = False

        if done.any():
            self._reset_accumulators(done)
        return completed

    def summary(self) -> dict:
        successful = [record for record in self.records if record.success]
        minimum_clearances = [record.minimum_clearance_m for record in self.records]
        risk_exposures = [record.risk_exposure for record in self.records]
        risk_exposures_per_second = [record.risk_exposure_per_second for record in self.records]
        path_lengths = [record.traveled_path_length_m for record in self.records]
        termination_counts: dict[str, int] = {}
        for record in self.records:
            for reason in record.termination_reasons:
                termination_counts[reason] = termination_counts.get(reason, 0) + 1

        total_steps = sum(record.episode_steps for record in self.records)
        below_0p55m_steps = sum(
            record.fraction_steps_below_0p55m * record.episode_steps
            for record in self.records
        )
        below_0p75m_steps = sum(
            record.fraction_steps_below_0p75m * record.episode_steps
            for record in self.records
        )
        below_1m_steps = sum(
            record.fraction_steps_below_1m * record.episode_steps
            for record in self.records
        )
        below_2m_steps = sum(
            record.fraction_steps_below_2m * record.episode_steps
            for record in self.records
        )
        successes = len(successful)
        collisions = sum(record.collision for record in self.records)
        completed = len(self.records)

        return {
            "clearance_definition": CLEARANCE_DEFINITION,
            "completed_episodes": completed,
            "successes": successes,
            "success_rate": successes / completed if completed else None,
            "collisions": collisions,
            "collision_rate": collisions / completed if completed else None,
            "termination_counts": termination_counts,
            "mean_episode_return": self._mean(record.episode_return for record in self.records),
            "mean_episode_steps": self._mean(record.episode_steps for record in self.records),
            "mean_episode_duration_s": self._mean(record.episode_duration_s for record in self.records),
            "mean_episode_minimum_clearance_m": self._mean(minimum_clearances),
            "median_episode_minimum_clearance_m": self._median(minimum_clearances),
            "lower_tail_clearance_p05_m": percentile(minimum_clearances, 0.05),
            "minimum_clearance_m": min(minimum_clearances) if minimum_clearances else None,
            "mean_risk_exposure": self._mean(risk_exposures),
            "median_risk_exposure": self._median(risk_exposures),
            "mean_risk_exposure_per_second": self._mean(risk_exposures_per_second),
            "fraction_steps_below_0p55m": below_0p55m_steps / total_steps if total_steps else None,
            "fraction_steps_below_0p75m": below_0p75m_steps / total_steps if total_steps else None,
            "fraction_steps_below_1m": below_1m_steps / total_steps if total_steps else None,
            "fraction_steps_below_2m": below_2m_steps / total_steps if total_steps else None,
            "mean_traveled_path_length_m": self._mean(path_lengths),
            "mean_successful_path_efficiency": self._mean(
                record.path_efficiency
                for record in successful
                if record.path_efficiency is not None
            ),
            "mean_successful_time_to_goal_s": self._mean(
                record.time_to_goal_s
                for record in successful
                if record.time_to_goal_s is not None
            ),
            "successful_episodes_with_path_metrics": sum(
                record.path_efficiency is not None
                for record in successful
            ),
        }

    def _finish_episode(
        self,
        env_id: int,
        termination_masks: Mapping[str, torch.Tensor],
    ) -> EpisodeRiskMetrics:
        reasons = sorted(
            name
            for name, mask in termination_masks.items()
            if bool(mask[env_id].item())
        )
        success = bool(termination_masks.get("is_success", self._false_mask())[env_id].item())
        collision = bool(termination_masks.get("collision", self._false_mask())[env_id].item())
        steps = int(self._steps[env_id].item())
        duration = steps * self.step_dt
        path_length = float(self._path_length[env_id].item())
        start_goal_distance = float(self._start_goal_distance[env_id].item())
        path_efficiency = (
            start_goal_distance / max(path_length, start_goal_distance)
            if path_length > 1e-8 and start_goal_distance > 1e-8
            else None
        )
        risk_exposure = float(self._risk_exposure[env_id].item())

        return EpisodeRiskMetrics(
            episode_index=len(self.records),
            success=success,
            collision=collision,
            termination_reasons=reasons,
            episode_return=float(self._returns[env_id].item()),
            episode_steps=steps,
            episode_duration_s=duration,
            minimum_clearance_m=float(self._minimum_clearance[env_id].item()),
            risk_exposure=risk_exposure,
            risk_exposure_per_second=risk_exposure / duration if duration > 0 else 0.0,
            traveled_path_length_m=path_length,
            start_goal_distance_m=start_goal_distance,
            path_efficiency=path_efficiency,
            time_to_goal_s=duration if success else None,
            fraction_steps_below_0p55m=float(self._below_0p55m[env_id].item()) / steps if steps else 0.0,
            fraction_steps_below_0p75m=float(self._below_0p75m[env_id].item()) / steps if steps else 0.0,
            fraction_steps_below_1m=float(self._below_1m[env_id].item()) / steps if steps else 0.0,
            fraction_steps_below_2m=float(self._below_2m[env_id].item()) / steps if steps else 0.0,
        )

    def _reset_accumulators(self, mask: torch.Tensor) -> None:
        self._returns[mask] = 0.0
        self._steps[mask] = 0
        self._path_length[mask] = 0.0
        self._minimum_clearance[mask] = float("inf")
        self._risk_exposure[mask] = 0.0
        self._below_0p55m[mask] = 0
        self._below_0p75m[mask] = 0
        self._below_1m[mask] = 0
        self._below_2m[mask] = 0

    def _vector(
        self,
        value: torch.Tensor,
        trailing_shape: tuple[int, ...] = (),
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device)
        expected = (self.num_envs, *trailing_shape)
        if tuple(tensor.shape) != expected:
            if not trailing_shape and tensor.numel() == self.num_envs:
                tensor = tensor.reshape(self.num_envs)
            else:
                raise ValueError(f"Expected tensor shape {expected}, got {tuple(tensor.shape)}.")
        return tensor.to(dtype=torch.float32)

    def _mask(self, value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        tensor = torch.as_tensor(value, device=self.device).bool().reshape(-1)
        if tensor.shape != (self.num_envs,):
            raise ValueError(f"Expected mask shape ({self.num_envs},), got {tuple(tensor.shape)}.")
        return tensor

    def _false_mask(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @staticmethod
    def _mean(values) -> float | None:
        values = [float(value) for value in values]
        return mean(values) if values else None

    @staticmethod
    def _median(values) -> float | None:
        values = [float(value) for value in values]
        return median(values) if values else None


def _base_env(env):
    return getattr(env, "unwrapped", env)


def _to_torch(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    import warp as wp

    return wp.to_torch(value)


def current_robot_xy(env, asset_name: str = "robot") -> torch.Tensor:
    base_env = _base_env(env)
    asset = base_env.scene[asset_name]
    return _to_torch(asset.data.root_link_pos_w)[:, :2].detach().clone()


def current_goal_distances(
    env,
    command_name: str = "target_pose",
    asset_name: str = "robot",
) -> torch.Tensor:
    base_env = _base_env(env)
    command_term = base_env.command_manager.get_term(command_name)
    target_positions_w = getattr(command_term, "pos_command_w", None)
    if target_positions_w is not None:
        robot_positions_xy = current_robot_xy(base_env, asset_name)
        return torch.linalg.vector_norm(
            target_positions_w[:, :2] - robot_positions_xy,
            dim=-1,
        ).detach().clone()

    command = base_env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(command[:, :2], dim=-1).detach().clone()


def clearance_at_xy(env, positions_xy: torch.Tensor) -> torch.Tensor:
    base_env = _base_env(env)
    terrain_manager = getattr(base_env.scene.terrain, "_terrainManager", None)
    risk_map = getattr(terrain_manager, "obstacle_risk_map", None)
    if risk_map is None:
        raise RuntimeError(
            "Risk evaluation requires scene.terrain._terrainManager.obstacle_risk_map. "
            "Use a terrain configuration with the projected-rock risk map enabled."
        )
    clearance = risk_map.distance_at_world_xy(positions_xy).to(
        device=positions_xy.device,
        dtype=torch.float32,
    )
    if not torch.isfinite(clearance).all():
        raise RuntimeError(
            "The obstacle risk map returned non-finite clearance values. "
            "Risk metrics require a terrain containing projected hazards."
        )
    return clearance


def current_clearance(env, asset_name: str = "robot") -> torch.Tensor:
    return clearance_at_xy(env, current_robot_xy(env, asset_name))


def termination_masks(env, device: str | torch.device) -> dict[str, torch.Tensor]:
    manager = getattr(_base_env(env), "termination_manager", None)
    if manager is None:
        return {}
    return {
        name: torch.as_tensor(manager.get_term(name), device=device).bool().reshape(-1)
        for name in getattr(manager, "active_terms", ())
    }


def make_risk_evaluation_recorder_cfg(
    *,
    asset_name: str = "robot",
):
    """Create an in-memory recorder that captures final state before auto-reset."""

    from isaaclab.managers.recorder_manager import (
        DatasetExportMode,
        RecorderManagerBaseCfg,
        RecorderTerm,
        RecorderTermCfg,
    )
    from isaaclab.utils import configclass

    configured_asset_name = asset_name
    class RiskEvaluationStateRecorder(RecorderTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.latest_positions_xy = None
            self.latest_clearance = None

        def record_post_step(self):
            self.latest_positions_xy = current_robot_xy(self._env, self.cfg.asset_name)
            self.latest_clearance = clearance_at_xy(self._env, self.latest_positions_xy)
            return None, None

    @configclass
    class RiskEvaluationStateRecorderCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = RiskEvaluationStateRecorder
        asset_name: str = configured_asset_name

    @configclass
    class RiskEvaluationRecorderManagerCfg(RecorderManagerBaseCfg):
        dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
        export_in_record_pre_reset: bool = False
        export_in_close: bool = False
        risk_evaluation_state: RiskEvaluationStateRecorderCfg = RiskEvaluationStateRecorderCfg()

    return RiskEvaluationRecorderManagerCfg()


def post_step_risk_state(env) -> tuple[torch.Tensor, torch.Tensor]:
    manager = getattr(_base_env(env), "recorder_manager", None)
    terms = getattr(manager, "_terms", {})
    term = terms.get(RECORDER_TERM_NAME)
    if term is None:
        raise RuntimeError(
            f"Recorder term {RECORDER_TERM_NAME!r} is not active. "
            "Set env_cfg.recorders = make_risk_evaluation_recorder_cfg() before gym.make()."
        )
    if term.latest_positions_xy is None or term.latest_clearance is None:
        raise RuntimeError("Risk evaluation recorder has not captured a completed environment step yet.")
    return term.latest_positions_xy, term.latest_clearance
