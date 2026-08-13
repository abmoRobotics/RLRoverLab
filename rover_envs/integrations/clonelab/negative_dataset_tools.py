"""HDF5 utilities for isolated negative-example shards."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REQUIRED_NEGATIVE_PATHS = (
    "transitions/actions",
    "transitions/rewards",
    "transitions/dones",
    "transitions/terminals",
    "transitions/timeouts",
    "transitions/extra/risk/min_distance_to_rock",
)
TERMINATION_PATH_PREFIX = "transitions/extra/termination"


@dataclass(frozen=True)
class EpisodeCandidate:
    source_episode: int
    transition_offset: int
    obs_offset: int
    length: int
    min_clearance_m: float
    failure: bool
    collision: bool
    far_from_target: bool
    timeout: bool
    success: bool


def validate_negative_dataset(
    path: str | Path,
    *,
    require_negative_metadata: bool = True,
    require_termination_metadata: bool = True,
    require_collision_penalty: bool = True,
    require_timeout_penalty: bool = False,
    terminal_reward_max: float = -1e-6,
) -> dict[str, Any]:
    """Validate reward/termination/clearance fields needed by RC-IQL relabeling."""

    path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {
        "path": str(path),
        "valid": False,
        "errors": errors,
        "warnings": warnings,
    }

    if not path.exists():
        errors.append(f"Dataset does not exist: {path}")
        return summary

    with h5py.File(path, "r") as file:
        summary["schema_name"] = _jsonable_attr(file.attrs.get("schema_name"))
        summary["writer_status"] = _jsonable_attr(file.attrs.get("writer_status"))
        summary["total_transitions"] = int(file.attrs.get("total_transitions", 0))
        summary["total_episodes"] = int(file.attrs.get("total_episodes", 0))

        if file.attrs.get("writer_status") != "complete":
            errors.append("writer_status must be 'complete'.")

        for required_path in REQUIRED_NEGATIVE_PATHS:
            if required_path not in file:
                errors.append(f"Missing required dataset path: /{required_path}")

        if errors:
            return summary

        rewards = np.asarray(file["transitions/rewards"][:]).reshape(-1)
        dones = np.asarray(file["transitions/dones"][:]).astype(bool).reshape(-1)
        terminals = np.asarray(file["transitions/terminals"][:]).astype(bool).reshape(-1)
        timeouts = np.asarray(file["transitions/timeouts"][:]).astype(bool).reshape(-1)
        clearance = np.asarray(file["transitions/extra/risk/min_distance_to_rock"][:]).reshape(-1)

        transition_count = summary["total_transitions"]
        for name, values in {
            "rewards": rewards,
            "dones": dones,
            "terminals": terminals,
            "timeouts": timeouts,
            "min_distance_to_rock": clearance,
        }.items():
            if values.shape[0] != transition_count:
                errors.append(f"{name} length {values.shape[0]} does not match total_transitions {transition_count}.")

        done_union = terminals | timeouts
        mismatch_count = int(np.count_nonzero(dones != done_union))
        summary["done_terminal_timeout_mismatches"] = mismatch_count
        if mismatch_count:
            errors.append("dones must equal terminals | timeouts.")

        finite_clearance = np.isfinite(clearance)
        summary["finite_clearance_fraction"] = _ratio(np.count_nonzero(finite_clearance), clearance.shape[0])
        if not finite_clearance.any():
            errors.append("min_distance_to_rock contains no finite values.")

        if require_negative_metadata:
            for path_suffix in ("mode_id", "gate", "active", "original_actions", "action_delta"):
                full_path = f"transitions/extra/negative/{path_suffix}"
                if full_path not in file:
                    errors.append(f"Missing negative metadata path: /{full_path}")

        termination_group_exists = TERMINATION_PATH_PREFIX in file
        if require_termination_metadata and not termination_group_exists:
            errors.append(f"Missing termination metadata group: /{TERMINATION_PATH_PREFIX}")

        collision = _optional_bool(file, f"{TERMINATION_PATH_PREFIX}/collision", transition_count)
        far_from_target = _optional_bool(file, f"{TERMINATION_PATH_PREFIX}/far_from_target", transition_count)
        success = _optional_bool(file, f"{TERMINATION_PATH_PREFIX}/is_success", transition_count)

        collision_count = int(np.count_nonzero(collision))
        timeout_count = int(np.count_nonzero(timeouts))
        success_count = int(np.count_nonzero(success))
        failure = collision | far_from_target | (terminals & ~success)
        summary.update(
            {
                "done_count": int(np.count_nonzero(dones)),
                "terminal_count": int(np.count_nonzero(terminals)),
                "timeout_count": timeout_count,
                "collision_count": collision_count,
                "far_from_target_count": int(np.count_nonzero(far_from_target)),
                "success_count": success_count,
                "failure_transition_count": int(np.count_nonzero(failure)),
                "min_clearance_m": float(np.nanmin(clearance[finite_clearance])) if finite_clearance.any() else None,
                "mean_reward": float(np.mean(rewards)) if rewards.size else None,
            }
        )

        if require_collision_penalty and collision_count:
            max_collision_reward = float(np.max(rewards[collision]))
            summary["max_collision_reward"] = max_collision_reward
            if max_collision_reward > terminal_reward_max:
                errors.append(
                    "Collision terminal rows do not appear to include a negative penalty: "
                    f"max collision reward is {max_collision_reward:.6g}."
                )
        elif require_collision_penalty:
            warnings.append("No collision rows were found; collision penalty could not be verified.")

        if require_timeout_penalty and timeout_count:
            max_timeout_reward = float(np.max(rewards[timeouts]))
            summary["max_timeout_reward"] = max_timeout_reward
            if max_timeout_reward > terminal_reward_max:
                errors.append(
                    "Timeout rows do not appear to include a negative failure penalty: "
                    f"max timeout reward is {max_timeout_reward:.6g}."
                )

    summary["valid"] = not errors
    return summary


def select_negative_episodes(
    path: str | Path,
    *,
    target_transitions: int,
    failures_only: bool = True,
    include_timeouts: bool = False,
    max_min_clearance_m: float | None = 2.5,
    seed: int = 0,
) -> tuple[list[EpisodeCandidate], dict[str, Any]]:
    """Select failure-heavy near-rock episodes from a candidate collection shard."""

    if target_transitions <= 0:
        raise ValueError(f"target_transitions must be positive, got {target_transitions}.")

    rng = np.random.default_rng(seed)
    candidates: list[EpisodeCandidate] = []
    path = Path(path)
    with h5py.File(path, "r") as file:
        lengths = np.asarray(file["index/episode_lengths"][:], dtype=np.int64)
        obs_offsets = np.asarray(file["index/obs_offsets"][:], dtype=np.int64)
        transition_offsets = np.asarray(file["index/transition_offsets"][:], dtype=np.int64)
        clearance = np.asarray(file["transitions/extra/risk/min_distance_to_rock"][:]).reshape(-1)
        terminals = np.asarray(file["transitions/terminals"][:]).astype(bool).reshape(-1)
        timeouts = np.asarray(file["transitions/timeouts"][:]).astype(bool).reshape(-1)
        collision = _optional_bool(file, f"{TERMINATION_PATH_PREFIX}/collision", clearance.shape[0])
        far_from_target = _optional_bool(file, f"{TERMINATION_PATH_PREFIX}/far_from_target", clearance.shape[0])
        success = _optional_bool(file, f"{TERMINATION_PATH_PREFIX}/is_success", clearance.shape[0])

        for episode_id, length in enumerate(lengths.tolist()):
            transition_offset = int(transition_offsets[episode_id])
            obs_offset = int(obs_offsets[episode_id])
            length = int(length)
            if length <= 0:
                continue
            slc = slice(transition_offset, transition_offset + length)
            episode_clearance = clearance[slc]
            finite_clearance = episode_clearance[np.isfinite(episode_clearance)]
            min_clearance = float(np.min(finite_clearance)) if finite_clearance.size else math.inf
            episode_success = bool(np.any(success[slc]))
            episode_collision = bool(np.any(collision[slc]))
            episode_far = bool(np.any(far_from_target[slc]))
            episode_timeout = bool(np.any(timeouts[slc]))
            episode_terminal_failure = bool(np.any(terminals[slc] & ~success[slc]))
            episode_failure = episode_collision or episode_far or episode_terminal_failure
            if include_timeouts:
                episode_failure = episode_failure or episode_timeout
            if episode_success:
                episode_failure = False

            if failures_only and not episode_failure:
                continue
            if max_min_clearance_m is not None and min_clearance > float(max_min_clearance_m):
                continue

            candidates.append(
                EpisodeCandidate(
                    source_episode=episode_id,
                    transition_offset=transition_offset,
                    obs_offset=obs_offset,
                    length=length,
                    min_clearance_m=min_clearance,
                    failure=episode_failure,
                    collision=episode_collision,
                    far_from_target=episode_far,
                    timeout=episode_timeout,
                    success=episode_success,
                )
            )

    order = rng.permutation(len(candidates))
    selected: list[EpisodeCandidate] = []
    selected_transitions = 0
    for index in order.tolist():
        episode = candidates[index]
        selected.append(episode)
        selected_transitions += episode.length
        if selected_transitions >= target_transitions:
            break

    selected.sort(key=lambda episode: episode.source_episode)
    summary = {
        "candidate_episodes": len(candidates),
        "selected_episodes": len(selected),
        "selected_transitions": selected_transitions,
        "target_transitions": target_transitions,
        "failures_only": failures_only,
        "include_timeouts": include_timeouts,
        "max_min_clearance_m": max_min_clearance_m,
        "selected_collision_episodes": sum(1 for episode in selected if episode.collision),
        "selected_far_from_target_episodes": sum(1 for episode in selected if episode.far_from_target),
        "selected_timeout_episodes": sum(1 for episode in selected if episode.timeout),
        "selected_success_episodes": sum(1 for episode in selected if episode.success),
    }
    return selected, summary


def copy_selected_episodes(
    input_path: str | Path,
    output_path: str | Path,
    selected_episodes: list[EpisodeCandidate],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy selected episodes into a standalone optimized RGB-D HDF5 shard."""

    if not selected_episodes:
        raise ValueError("No episodes were selected for the negative shard.")

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = output_path.with_name(f"{output_path.name}.incomplete.{os.getpid()}")
    if staging_path.exists():
        staging_path.unlink()

    obs_ranges = [slice(episode.obs_offset, episode.obs_offset + episode.length + 1) for episode in selected_episodes]
    transition_ranges = [
        slice(episode.transition_offset, episode.transition_offset + episode.length) for episode in selected_episodes
    ]

    with h5py.File(input_path, "r") as src, h5py.File(staging_path, "w") as dst:
        _copy_attrs(src, dst)
        dst.attrs["writer_status"] = "incomplete"

        data_group = dst.create_group("data")
        if "data" in src:
            _copy_attrs(src["data"], data_group)

        observations_group = dst.create_group("observations")
        _copy_rows_recursive(src["observations"], observations_group, obs_ranges)

        transitions_group = dst.create_group("transitions")
        _copy_rows_recursive(src["transitions"], transitions_group, transition_ranges)

        index_group = dst.create_group("index")
        episodes_group = dst.create_group("episodes")
        _write_rebuilt_index_and_episodes(src, index_group, episodes_group, selected_episodes)

        total_transitions = int(sum(episode.length for episode in selected_episodes))
        total_observations = int(sum(episode.length + 1 for episode in selected_episodes))
        dst.attrs["total_episodes"] = len(selected_episodes)
        dst.attrs["total_transitions"] = total_transitions
        dst.attrs["total_observations"] = total_observations
        dst.attrs["negative_shard"] = True
        if metadata:
            dst.attrs["negative_shard_metadata"] = json.dumps(metadata, sort_keys=True)
        data_group.attrs["total"] = total_transitions
        dst.attrs["writer_status"] = "complete"

    os.replace(staging_path, output_path)
    validation = validate_negative_dataset(output_path, require_timeout_penalty=False)
    if not validation["valid"]:
        raise RuntimeError(f"Copied negative shard failed validation: {validation['errors']}")
    return validation


def filter_negative_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_transitions: int,
    failures_only: bool = True,
    include_timeouts: bool = False,
    max_min_clearance_m: float | None = 2.5,
    seed: int = 0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected, selection_summary = select_negative_episodes(
        input_path,
        target_transitions=target_transitions,
        failures_only=failures_only,
        include_timeouts=include_timeouts,
        max_min_clearance_m=max_min_clearance_m,
        seed=seed,
    )
    validation = copy_selected_episodes(
        input_path,
        output_path,
        selected,
        metadata={**(metadata or {}), "selection": selection_summary},
    )
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "selection": selection_summary,
        "validation": validation,
    }


def write_multishard_manifest(
    path: str | Path,
    *,
    positive_paths: list[str],
    negative_path: str,
    negative_transition_fraction: float,
    d_ref: float = 5.0,
    clearance_exponent: float = 2.5,
) -> dict[str, Any]:
    if not 0.0 <= negative_transition_fraction <= 1.0:
        raise ValueError("negative_transition_fraction must lie in [0, 1].")

    manifest = {
        "schema_version": 1,
        "dataset_type": "rlroverlab_rc_iql_multishard",
        "sampling": {
            "negative_transition_fraction": float(negative_transition_fraction),
            "positive_transition_fraction": float(1.0 - negative_transition_fraction),
        },
        "reward_relabeling": {
            "task_reward_path": "/transitions/rewards",
            "clearance_path": "/transitions/extra/risk/min_distance_to_rock",
            "formula": (
                "reward_alpha = task_reward - lambda_risk * alpha * "
                "exp(-log(100) * (clearance / d_ref) ** clearance_exponent)"
            ),
            "d_ref": float(d_ref),
            "clearance_exponent": float(clearance_exponent),
        },
        "shards": [
            {"role": "positive_teacher", "path": str(dataset_path), "preserve": True}
            for dataset_path in positive_paths
        ]
        + [{"role": "negative_examples", "path": str(negative_path), "preserve": False}],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _optional_bool(file: h5py.File, path: str, expected_length: int) -> np.ndarray:
    if path not in file:
        return np.zeros((expected_length,), dtype=bool)
    values = np.asarray(file[path][:]).astype(bool).reshape(-1)
    if values.shape[0] != expected_length:
        raise ValueError(f"/{path} length {values.shape[0]} does not match expected length {expected_length}.")
    return values


def _copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _copy_rows_recursive(src_group: h5py.Group, dst_group: h5py.Group, row_ranges: list[slice]) -> None:
    _copy_attrs(src_group, dst_group)
    for name, item in src_group.items():
        if isinstance(item, h5py.Group):
            child = dst_group.create_group(name)
            _copy_rows_recursive(item, child, row_ranges)
        elif isinstance(item, h5py.Dataset):
            dataset = _create_empty_like(dst_group, name, item)
            for row_range in row_ranges:
                _append_rows(dataset, item[row_range])
        else:
            raise TypeError(f"Unsupported HDF5 object at {item.name}: {type(item)!r}")


def _create_empty_like(dst_group: h5py.Group, name: str, src_dataset: h5py.Dataset) -> h5py.Dataset:
    sample_shape = tuple(src_dataset.shape[1:])
    chunks = src_dataset.chunks
    if chunks is None:
        chunks = (min(1024, max(1, int(src_dataset.shape[0]) if src_dataset.shape else 1)), *sample_shape)
    maxshape = (None, *sample_shape)
    dataset = dst_group.create_dataset(
        name,
        shape=(0, *sample_shape),
        maxshape=maxshape,
        dtype=src_dataset.dtype,
        chunks=chunks,
    )
    _copy_attrs(src_dataset, dataset)
    return dataset


def _append_rows(dataset: h5py.Dataset, values: np.ndarray) -> None:
    values = np.asarray(values)
    old_size = int(dataset.shape[0])
    if h5py.check_dtype(vlen=dataset.dtype) is not None:
        row_count = int(values.shape[0])
        dataset.resize((old_size + row_count,))
        for index in range(row_count):
            dataset[old_size + index] = values[index]
        return

    dataset.resize((old_size + int(values.shape[0]), *dataset.shape[1:]))
    dataset[old_size : old_size + int(values.shape[0])] = values


def _write_rebuilt_index_and_episodes(
    src: h5py.File,
    index_group: h5py.Group,
    episodes_group: h5py.Group,
    selected_episodes: list[EpisodeCandidate],
) -> None:
    episode_lengths = []
    obs_offsets = []
    transition_offsets = []
    obs_index = []
    next_obs_index = []
    episode_id = []
    episode_transition_index = []
    next_obs_offset = 0
    next_transition_offset = 0

    for new_episode_id, episode in enumerate(selected_episodes):
        episode_lengths.append(episode.length)
        obs_offsets.append(next_obs_offset)
        transition_offsets.append(next_transition_offset)
        obs_index.extend(range(next_obs_offset, next_obs_offset + episode.length))
        next_obs_index.extend(range(next_obs_offset + 1, next_obs_offset + episode.length + 1))
        episode_id.extend([new_episode_id] * episode.length)
        episode_transition_index.extend(range(episode.length))

        src_episode_name = f"demo_{episode.source_episode}"
        dst_episode_name = f"demo_{new_episode_id}"
        dst_episode_group = episodes_group.create_group(dst_episode_name)
        if f"episodes/{src_episode_name}" in src:
            _copy_attrs(src[f"episodes/{src_episode_name}"], dst_episode_group)
        dst_episode_group.attrs["num_samples"] = episode.length
        dst_episode_group.attrs["num_observations"] = episode.length + 1
        dst_episode_group.attrs["obs_offset"] = next_obs_offset
        dst_episode_group.attrs["transition_offset"] = next_transition_offset
        dst_episode_group.attrs["source_episode"] = episode.source_episode
        dst_episode_group.attrs["negative_failure"] = bool(episode.failure)
        dst_episode_group.attrs["negative_collision"] = bool(episode.collision)
        dst_episode_group.attrs["negative_far_from_target"] = bool(episode.far_from_target)
        dst_episode_group.attrs["negative_timeout"] = bool(episode.timeout)
        dst_episode_group.attrs["negative_success"] = bool(episode.success)
        dst_episode_group.attrs["minimum_clearance_m"] = float(episode.min_clearance_m)

        next_obs_offset += episode.length + 1
        next_transition_offset += episode.length

    _create_array_dataset(index_group, "episode_lengths", np.asarray(episode_lengths, dtype=np.int64))
    _create_array_dataset(index_group, "obs_offsets", np.asarray(obs_offsets, dtype=np.int64))
    _create_array_dataset(index_group, "transition_offsets", np.asarray(transition_offsets, dtype=np.int64))
    _create_array_dataset(index_group, "obs_index", np.asarray(obs_index, dtype=np.int64))
    _create_array_dataset(index_group, "next_obs_index", np.asarray(next_obs_index, dtype=np.int64))
    _create_array_dataset(index_group, "episode_id", np.asarray(episode_id, dtype=np.int64))
    _create_array_dataset(
        index_group,
        "episode_transition_index",
        np.asarray(episode_transition_index, dtype=np.int64),
    )


def _create_array_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    chunks = (min(1024, max(1, values.shape[0])), *values.shape[1:])
    group.create_dataset(
        name,
        data=values,
        maxshape=(None, *values.shape[1:]),
        chunks=chunks,
    )


def _jsonable_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        return value.item()
    return value


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None
