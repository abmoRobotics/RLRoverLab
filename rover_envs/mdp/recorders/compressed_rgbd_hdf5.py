from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from isaaclab.utils.datasets import EpisodeData
from isaaclab.utils.datasets.dataset_file_handler_base import DatasetFileHandlerBase


SCHEMA_NAME = "rlroverlab.offline_rgbd_v2"
FORMAT_VERSION = 2
RGB_KEY = "rgb_image"
DEPTH_KEY = "depth_image"


class CompressedRGBDHDF5DatasetFileHandler(DatasetFileHandlerBase):
    """HDF5 writer for zero-duplication RGB-D offline RL datasets.

    The handler consumes Isaac Lab ``EpisodeData`` objects but writes a different
    schema than Isaac Lab's default gzip-per-dataset exporter. Visual observations
    are stored once as an observation timeline using variable-length uint8 byte
    streams. Transitions refer to adjacent observations via index datasets.
    """

    rgb_jpeg_quality: int = 88
    depth_min_m: float = 0.1
    depth_max_m: float = 6.0
    depth_unit_m: float = 0.001
    depth_invalid_sentinel: int = 0
    # OpenCV's JPEG2000 flag is a target compression parameter multiplied by 1000.
    # Keep the default conservative; PSNR should be measured in validation runs.
    depth_jpeg2000_compression_x1000: int = 1000
    chunk_length: int = 1024
    target_numeric_chunk_bytes: int = 1024 * 1024

    def __init__(self):
        self._hdf5_file_stream: h5py.File | None = None
        self._final_path: Path | None = None
        self._staging_path: Path | None = None
        self._env_args: dict[str, Any] = {}
        self._demo_count = 0
        self._total_observations = 0
        self._total_transitions = 0
        self._created_for_write = False
        self._write_failed = False

    def open(self, file_path: str, mode: str = "r"):
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        self._hdf5_file_stream = h5py.File(file_path, mode)
        self._created_for_write = mode not in ("r", "r-")
        episodes = self._hdf5_file_stream.get("episodes")
        self._demo_count = len(episodes) if episodes is not None else 0
        self._total_observations = int(self._hdf5_file_stream.attrs.get("total_observations", 0))
        self._total_transitions = int(self._hdf5_file_stream.attrs.get("total_transitions", 0))

    def create(self, file_path: str, env_name: str = None):
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")

        final_path = Path(file_path)
        if final_path.suffix != ".hdf5":
            if final_path.suffix:
                final_path = final_path.with_suffix(final_path.suffix + ".hdf5")
            else:
                final_path = final_path.with_suffix(".hdf5")
        final_path.parent.mkdir(parents=True, exist_ok=True)

        staging_name = f"{final_path.name}.incomplete.{os.getpid()}"
        staging_path = final_path.with_name(staging_name)

        self._final_path = final_path
        self._staging_path = staging_path
        self._created_for_write = True
        self._write_failed = False
        self._hdf5_file_stream = h5py.File(staging_path, "w")
        self._demo_count = 0
        self._total_observations = 0
        self._total_transitions = 0

        file_stream = self._require_file()
        file_stream.attrs["schema_name"] = SCHEMA_NAME
        file_stream.attrs["format_version"] = FORMAT_VERSION
        file_stream.attrs["writer_status"] = "incomplete"
        file_stream.attrs["visual_profile"] = "rgb_jpeg_depth_u16_jpeg2000"
        file_stream.attrs["storage_contract_version"] = 1
        file_stream.attrs["visual_storage_contract"] = "raw_sensor_values_compressed"
        file_stream.attrs["training_normalization_applied"] = False
        file_stream.attrs["normalization_contract"] = "loader_applies_normalization_after_decode"
        file_stream.attrs["recommended_rgb_scale"] = 1.0 / 255.0
        file_stream.attrs["recommended_depth_scale_m"] = self.depth_unit_m
        file_stream.attrs["zero_structural_duplication"] = True
        file_stream.attrs["rgb_codec"] = "jpeg"
        file_stream.attrs["rgb_jpeg_quality"] = self.rgb_jpeg_quality
        file_stream.attrs["rgb_source"] = "tiled_camera.rgb_raw"
        file_stream.attrs["rgb_storage_dtype"] = "uint8"
        file_stream.attrs["rgb_storage_range_min"] = 0
        file_stream.attrs["rgb_storage_range_max"] = 255
        file_stream.attrs["rgb_training_scale"] = 1.0 / 255.0
        file_stream.attrs["depth_codec"] = "jpeg2000_u16_mm"
        file_stream.attrs["depth_source"] = "tiled_camera.depth_raw"
        file_stream.attrs["depth_source_unit"] = "meters"
        file_stream.attrs["depth_storage_dtype"] = "uint16"
        file_stream.attrs["depth_storage_unit"] = "millimeters"
        file_stream.attrs["depth_training_scale_m"] = self.depth_unit_m
        file_stream.attrs["depth_psnr_target_db"] = 50.0
        file_stream.attrs["depth_unit_m"] = self.depth_unit_m
        file_stream.attrs["depth_min_m"] = self.depth_min_m
        file_stream.attrs["depth_max_m"] = self.depth_max_m
        file_stream.attrs["depth_invalid_sentinel"] = self.depth_invalid_sentinel
        file_stream.attrs["total_episodes"] = 0
        file_stream.attrs["total_transitions"] = 0
        file_stream.attrs["total_observations"] = 0

        data_group = file_stream.create_group("data")
        data_group.attrs["total"] = 0
        file_stream.create_group("episodes")

        observations = file_stream.create_group("observations")
        visual_dtype = h5py.vlen_dtype(np.dtype("uint8"))
        rgb_dataset = observations.create_dataset(
            "rgb_jpeg",
            shape=(0,),
            maxshape=(None,),
            dtype=visual_dtype,
            chunks=(self.chunk_length,),
        )
        self._set_rgb_dataset_metadata(rgb_dataset)
        depth_dataset = observations.create_dataset(
            "depth_jp2",
            shape=(0,),
            maxshape=(None,),
            dtype=visual_dtype,
            chunks=(self.chunk_length,),
        )
        self._set_depth_dataset_metadata(depth_dataset)
        observations.create_group("state")

        transitions = file_stream.create_group("transitions")
        transitions.create_group("extra")

        index = file_stream.create_group("index")
        self._create_resizable_numeric_dataset(index, "episode_lengths", np.dtype("int64"), ())
        self._create_resizable_numeric_dataset(index, "obs_offsets", np.dtype("int64"), ())
        self._create_resizable_numeric_dataset(index, "transition_offsets", np.dtype("int64"), ())
        self._create_resizable_numeric_dataset(index, "obs_index", np.dtype("int64"), ())
        self._create_resizable_numeric_dataset(index, "next_obs_index", np.dtype("int64"), ())
        self._create_resizable_numeric_dataset(index, "episode_id", np.dtype("int64"), ())
        self._create_resizable_numeric_dataset(index, "episode_transition_index", np.dtype("int64"), ())

        env_name = env_name if env_name is not None else ""
        self.add_env_args({"env_name": env_name, "type": 2})

    def add_env_args(self, env_args: dict):
        file_stream = self._require_file()
        self._env_args.update(env_args)
        if "data" in file_stream:
            file_stream["data"].attrs["env_args"] = json.dumps(self._env_args)

    def set_env_name(self, env_name: str):
        self.add_env_args({"env_name": env_name})

    def get_env_name(self) -> str | None:
        file_stream = self._require_file()
        data_group = file_stream.get("data")
        if data_group is None or "env_args" not in data_group.attrs:
            return None
        env_args = json.loads(data_group.attrs["env_args"])
        return env_args.get("env_name")

    def get_episode_names(self) -> Iterable[str]:
        file_stream = self._require_file()
        episodes = file_stream.get("episodes")
        return episodes.keys() if episodes is not None else []

    def get_num_episodes(self) -> int:
        return self._demo_count

    @property
    def demo_count(self) -> int:
        return self._demo_count

    def load_episode(
        self,
        episode_name: str,
        device: str = "cpu",
        convert_legacy_quat: bool | None = None,
    ) -> EpisodeData | None:
        raise NotImplementedError(
            "Compressed RGB-D datasets are intended for random-access offline loaders, not Isaac Lab episode replay."
        )

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None):
        file_stream = self._require_file()
        if episode.is_empty():
            return

        data = episode.data
        actions = data.get("actions")
        if actions is None:
            return

        actions_np = self._as_numpy(actions)
        transition_count = int(actions_np.shape[0])
        if transition_count == 0:
            return

        observations = data.get("obs") or data.get("observations")
        if not isinstance(observations, Mapping):
            self._write_failed = True
            raise ValueError(
                "Compressed RGB-D recorder expected episode.data['obs'] to contain a policy observation dict."
            )

        obs_count = self._leaf_length(observations)
        expected_obs_count = transition_count + 1
        if obs_count != expected_obs_count:
            self._write_failed = True
            raise ValueError(
                f"Observation timeline length must equal actions + 1. Got {obs_count} observations and "
                f"{transition_count} actions."
            )

        if RGB_KEY not in observations:
            self._write_failed = True
            raise KeyError(f"Missing visual observation key '{RGB_KEY}' in episode observations.")
        if DEPTH_KEY not in observations:
            self._write_failed = True
            raise KeyError(f"Missing visual observation key '{DEPTH_KEY}' in episode observations.")

        rgb_np = self._as_numpy(observations[RGB_KEY])
        depth_np = self._as_numpy(observations[DEPTH_KEY])
        if rgb_np.shape[0] != obs_count or depth_np.shape[0] != obs_count:
            self._write_failed = True
            raise ValueError("RGB and depth observation timelines must match the episode observation count.")
        self._set_visual_shape_metadata(file_stream, rgb_np, depth_np)

        episode_index = self._demo_count if demo_id is None else int(demo_id)
        episode_group_name = f"demo_{episode_index}"
        episodes_group = file_stream["episodes"]
        if episode_group_name in episodes_group:
            self._write_failed = True
            raise ValueError(f"Episode group '{episode_group_name}' already exists in the compressed dataset.")

        obs_offset = self._total_observations
        transition_offset = self._total_transitions

        rgb_bytes = [self._encode_rgb_jpeg(rgb_np[index]) for index in range(obs_count)]
        depth_bytes = [
            self._encode_depth_jp2(self._quantize_depth_to_mm(depth_np[index])) for index in range(obs_count)
        ]
        self._append_vlen_uint8(file_stream["observations/rgb_jpeg"], rgb_bytes)
        self._append_vlen_uint8(file_stream["observations/depth_jp2"], depth_bytes)

        state_group = file_stream["observations/state"]
        for key, value in observations.items():
            if key in (RGB_KEY, DEPTH_KEY):
                continue
            self._append_nested_numeric(state_group, key, value, expected_length=obs_count)

        transitions_group = file_stream["transitions"]
        self._append_named_transition(transitions_group, "actions", actions_np, transition_count)
        self._append_named_transition(transitions_group, "rewards", self._as_numpy(data["rewards"]), transition_count)

        dones = self._as_numpy(data.get("dones", np.zeros((transition_count,), dtype=np.bool_))).astype(np.bool_)
        terminals = self._as_numpy(data.get("terminals", dones)).astype(np.bool_)
        timeouts = self._as_numpy(data.get("timeouts", np.zeros((transition_count,), dtype=np.bool_))).astype(np.bool_)
        self._append_named_transition(transitions_group, "dones", dones, transition_count)
        self._append_named_transition(transitions_group, "terminals", terminals, transition_count)
        self._append_named_transition(transitions_group, "timeouts", timeouts, transition_count)

        extra_group = transitions_group["extra"]
        reserved = {"actions", "rewards", "dones", "terminals", "timeouts", "obs", "observations"}
        for key, value in data.items():
            if key in reserved:
                continue
            self._append_nested_numeric(extra_group, key, value, expected_length=transition_count)

        obs_indices = np.arange(obs_offset, obs_offset + transition_count, dtype=np.int64)
        next_obs_indices = obs_indices + 1
        index_group = file_stream["index"]
        self._append_numeric(index_group["obs_index"], obs_indices)
        self._append_numeric(index_group["next_obs_index"], next_obs_indices)
        self._append_numeric(index_group["episode_id"], np.full((transition_count,), episode_index, dtype=np.int64))
        self._append_numeric(index_group["episode_transition_index"], np.arange(transition_count, dtype=np.int64))
        self._append_numeric(index_group["episode_lengths"], np.asarray([transition_count], dtype=np.int64))
        self._append_numeric(index_group["obs_offsets"], np.asarray([obs_offset], dtype=np.int64))
        self._append_numeric(index_group["transition_offsets"], np.asarray([transition_offset], dtype=np.int64))

        episode_group = episodes_group.create_group(episode_group_name)
        episode_group.attrs["num_samples"] = transition_count
        episode_group.attrs["num_observations"] = obs_count
        episode_group.attrs["obs_offset"] = obs_offset
        episode_group.attrs["transition_offset"] = transition_offset
        if episode.seed is not None:
            episode_group.attrs["seed"] = episode.seed
        if episode.success is not None:
            episode_group.attrs["success"] = bool(episode.success)

        self._total_observations += obs_count
        self._total_transitions += transition_count
        self._demo_count += 1 if demo_id is None else 0

        file_stream.attrs["total_episodes"] = len(episodes_group)
        file_stream.attrs["total_observations"] = self._total_observations
        file_stream.attrs["total_transitions"] = self._total_transitions
        file_stream["data"].attrs["total"] = self._total_transitions

    def flush(self):
        if self._hdf5_file_stream is not None:
            self._hdf5_file_stream.flush()
            self._fsync_if_possible()

    def close(self):
        if self._hdf5_file_stream is None:
            return

        file_stream = self._hdf5_file_stream
        try:
            if self._created_for_write:
                if self._write_failed:
                    file_stream.attrs["writer_status"] = "failed"
                    file_stream.flush()
                    self._fsync_if_possible()
                    return
                file_stream.attrs["writer_status"] = "closing"
                self._validate_open_file(file_stream)
                file_stream.attrs["writer_status"] = "complete"
            file_stream.flush()
            self._fsync_if_possible()
        finally:
            file_stream.close()
            self._hdf5_file_stream = None

        if self._created_for_write and self._staging_path is not None and self._final_path is not None:
            self._validate_reopen(self._staging_path)
            os.replace(self._staging_path, self._final_path)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _require_file(self) -> h5py.File:
        if self._hdf5_file_stream is None:
            raise RuntimeError("HDF5 dataset file stream is not initialized")
        return self._hdf5_file_stream

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _leaf_length(cls, value: Mapping[str, Any]) -> int:
        for item in value.values():
            if isinstance(item, Mapping):
                return cls._leaf_length(item)
            return int(cls._as_numpy(item).shape[0])
        raise ValueError("Observation dictionary is empty.")

    @staticmethod
    def _ensure_group(parent: h5py.Group, path: str) -> h5py.Group:
        group = parent
        for part in path.split("/")[:-1]:
            group = group.require_group(part)
        return group

    def _create_resizable_numeric_dataset(
        self,
        group: h5py.Group,
        name: str,
        dtype: np.dtype,
        sample_shape: tuple[int, ...],
    ):
        row_elements = int(np.prod(sample_shape, dtype=np.int64)) if sample_shape else 1
        row_bytes = max(1, np.dtype(dtype).itemsize * row_elements)
        chunk_rows = max(1, min(self.chunk_length, self.target_numeric_chunk_bytes // row_bytes))
        return group.create_dataset(
            name,
            shape=(0, *sample_shape),
            maxshape=(None, *sample_shape),
            dtype=dtype,
            chunks=(chunk_rows, *sample_shape),
        )

    def _require_resizable_numeric_dataset(self, group: h5py.Group, name: str, values: np.ndarray):
        sample_shape = tuple(values.shape[1:])
        if name not in group:
            return self._create_resizable_numeric_dataset(group, name, values.dtype, sample_shape)
        dataset = group[name]
        if tuple(dataset.shape[1:]) != sample_shape:
            raise ValueError(f"Dataset '{dataset.name}' shape mismatch: {dataset.shape[1:]} vs {sample_shape}")
        if dataset.dtype != values.dtype:
            values = values.astype(dataset.dtype, copy=False)
        return dataset

    def _append_named_transition(self, group: h5py.Group, name: str, values: np.ndarray, expected_length: int):
        values = np.asarray(values)
        if values.shape[0] != expected_length:
            raise ValueError(f"Transition dataset '{name}' has length {values.shape[0]}, expected {expected_length}.")
        dataset = self._require_resizable_numeric_dataset(group, name, values)
        self._append_numeric(dataset, values)

    def _append_nested_numeric(self, group: h5py.Group, key: str, value: Any, expected_length: int):
        if isinstance(value, Mapping):
            child = group.require_group(key)
            for sub_key, sub_value in value.items():
                self._append_nested_numeric(child, sub_key, sub_value, expected_length)
            return

        values = self._as_numpy(value)
        if values.shape[0] != expected_length:
            raise ValueError(f"Dataset '{key}' has length {values.shape[0]}, expected {expected_length}.")
        parent = self._ensure_group(group, key)
        dataset_name = key.split("/")[-1]
        dataset = self._require_resizable_numeric_dataset(parent, dataset_name, values)
        self._append_numeric(dataset, values)

    @staticmethod
    def _append_numeric(dataset: h5py.Dataset, values: np.ndarray):
        values = np.asarray(values)
        old_size = dataset.shape[0]
        dataset.resize((old_size + values.shape[0], *dataset.shape[1:]))
        dataset[old_size : old_size + values.shape[0]] = values

    @staticmethod
    def _append_vlen_uint8(dataset: h5py.Dataset, byte_arrays: list[np.ndarray]):
        old_size = dataset.shape[0]
        dataset.resize((old_size + len(byte_arrays),))
        for index, array in enumerate(byte_arrays):
            dataset[old_size + index] = np.asarray(array, dtype=np.uint8)

    def _set_rgb_dataset_metadata(self, dataset: h5py.Dataset):
        dataset.attrs["source"] = "tiled_camera.rgb"
        dataset.attrs["source_space"] = "raw_sensor"
        dataset.attrs["codec"] = "jpeg"
        dataset.attrs["jpeg_quality"] = self.rgb_jpeg_quality
        dataset.attrs["stored_as"] = "variable_length_uint8_encoded_bytes"
        dataset.attrs["decoded_dtype"] = "uint8"
        dataset.attrs["decoded_channel_order"] = "RGB"
        dataset.attrs["decoded_range_min"] = 0
        dataset.attrs["decoded_range_max"] = 255
        dataset.attrs["training_normalization_applied"] = False
        dataset.attrs["recommended_training_scale"] = 1.0 / 255.0
        dataset.attrs["recommended_training_dtype"] = "float32"

    def _set_depth_dataset_metadata(self, dataset: h5py.Dataset):
        dataset.attrs["source"] = "tiled_camera.depth"
        dataset.attrs["source_space"] = "raw_sensor"
        dataset.attrs["source_unit"] = "meters"
        dataset.attrs["codec"] = "jpeg2000"
        dataset.attrs["codec_input_dtype"] = "uint16"
        dataset.attrs["stored_as"] = "variable_length_uint8_encoded_bytes"
        dataset.attrs["decoded_dtype"] = "uint16"
        dataset.attrs["decoded_unit"] = "millimeters"
        dataset.attrs["decoded_valid_min"] = int(round(self.depth_min_m / self.depth_unit_m))
        dataset.attrs["decoded_valid_max"] = int(round(self.depth_max_m / self.depth_unit_m))
        dataset.attrs["invalid_sentinel"] = self.depth_invalid_sentinel
        dataset.attrs["source_valid_min_m"] = self.depth_min_m
        dataset.attrs["source_valid_max_m"] = self.depth_max_m
        dataset.attrs["recommended_training_scale_m"] = self.depth_unit_m
        dataset.attrs["recommended_training_dtype"] = "float32"

    def _set_visual_shape_metadata(self, file_stream: h5py.File, rgb: np.ndarray, depth: np.ndarray):
        rgb_shape = tuple(rgb.shape[1:])
        depth_shape = tuple(depth.shape[1:])
        if len(rgb_shape) != 3 or rgb_shape[-1] not in (3, 4):
            self._write_failed = True
            raise ValueError(f"Expected RGB observation shape (N, H, W, 3/4), got {rgb.shape}.")
        if len(depth_shape) == 3 and depth_shape[-1] == 1:
            depth_hw = depth_shape[:2]
        elif len(depth_shape) == 2:
            depth_hw = depth_shape
        else:
            self._write_failed = True
            raise ValueError(f"Expected depth observation shape (N, H, W) or (N, H, W, 1), got {depth.shape}.")
        if rgb_shape[:2] != depth_hw:
            self._write_failed = True
            raise ValueError(f"RGB and depth resolution mismatch: {rgb_shape[:2]} vs {depth_hw}.")

        height, width = rgb_shape[:2]
        channels = 3 if rgb_shape[-1] == 4 else rgb_shape[-1]
        existing_width = file_stream.attrs.get("camera_width")
        existing_height = file_stream.attrs.get("camera_height")
        if existing_width is not None and int(existing_width) != width:
            raise ValueError(f"Camera width changed within dataset: {existing_width} vs {width}.")
        if existing_height is not None and int(existing_height) != height:
            raise ValueError(f"Camera height changed within dataset: {existing_height} vs {height}.")

        file_stream.attrs["camera_width"] = width
        file_stream.attrs["camera_height"] = height
        file_stream.attrs["camera_channels"] = channels
        file_stream["observations/rgb_jpeg"].attrs["width"] = width
        file_stream["observations/rgb_jpeg"].attrs["height"] = height
        file_stream["observations/rgb_jpeg"].attrs["channels"] = channels
        file_stream["observations/depth_jp2"].attrs["width"] = width
        file_stream["observations/depth_jp2"].attrs["height"] = height
        file_stream["observations/depth_jp2"].attrs["channels"] = 1

    def _encode_rgb_jpeg(self, frame: np.ndarray) -> np.ndarray:
        cv2 = self._require_cv2()
        rgb = self._coerce_rgb_to_uint8(frame)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(self.rgb_jpeg_quality)])
        if not ok:
            self._write_failed = True
            raise RuntimeError("OpenCV failed to encode RGB frame as JPEG.")
        return np.asarray(encoded, dtype=np.uint8)

    def _coerce_rgb_to_uint8(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            self._write_failed = True
            raise ValueError(f"Expected RGB frame with shape (H, W, 3/4), got {frame.shape}.")
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype == np.uint8:
            return np.ascontiguousarray(frame)

        frame = frame.astype(np.float32, copy=False)
        finite = frame[np.isfinite(frame)]
        if finite.size == 0:
            return np.zeros(frame.shape, dtype=np.uint8)

        min_value = float(finite.min())
        max_value = float(finite.max())
        if min_value >= -0.05 and max_value <= 1.05:
            frame = frame * 255.0
        elif min_value >= -1.05 and max_value <= 1.05:
            # Defensive fallback for normalized policy observations. The
            # compressed recorder should receive raw camera RGB, but this keeps
            # accidental normalized inputs from collapsing to a black JPEG.
            frame = (frame + 1.0) * 127.5

        frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
        return np.ascontiguousarray(np.clip(np.rint(frame), 0, 255).astype(np.uint8))

    def _quantize_depth_to_mm(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        if frame.ndim != 2:
            self._write_failed = True
            raise ValueError(f"Expected depth frame with shape (H, W) or (H, W, 1), got {frame.shape}.")

        depth_m = frame.astype(np.float32, copy=False)
        quantized = np.zeros(depth_m.shape, dtype=np.uint16)
        valid = np.isfinite(depth_m) & (depth_m >= self.depth_min_m) & (depth_m <= self.depth_max_m)
        scaled = np.rint(depth_m[valid] / self.depth_unit_m)
        max_mm = int(round(self.depth_max_m / self.depth_unit_m))
        min_mm = int(round(self.depth_min_m / self.depth_unit_m))
        quantized[valid] = np.clip(scaled, min_mm, max_mm).astype(np.uint16)
        return quantized

    def _encode_depth_jp2(self, frame_u16: np.ndarray) -> np.ndarray:
        cv2 = self._require_cv2()
        if frame_u16.dtype != np.uint16:
            self._write_failed = True
            raise ValueError(f"Expected uint16 depth frame, got {frame_u16.dtype}.")
        params = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, int(self.depth_jpeg2000_compression_x1000)]
        ok, encoded = cv2.imencode(".jp2", np.ascontiguousarray(frame_u16), params)
        if not ok:
            self._write_failed = True
            raise RuntimeError("OpenCV failed to encode uint16 depth frame as JPEG2000.")
        return np.asarray(encoded, dtype=np.uint8)

    @staticmethod
    def _require_cv2():
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Compressed RGB-D recorder requires opencv-python with JPEG/JPEG2000 support.") from exc
        return cv2

    def _validate_open_file(self, file_stream: h5py.File):
        if file_stream.attrs.get("schema_name") != SCHEMA_NAME:
            raise RuntimeError("Compressed RGB-D dataset schema name is missing or invalid.")
        if "next_obs" in file_stream:
            raise RuntimeError("Compressed RGB-D schema must not contain a top-level next_obs dataset.")

        forbidden_paths: list[str] = []

        def visitor(name: str):
            if name.endswith("/next_obs") or "/next_obs/" in name:
                forbidden_paths.append(name)

        file_stream.visit(visitor)
        if forbidden_paths:
            raise RuntimeError(f"Compressed RGB-D schema contains forbidden next_obs paths: {forbidden_paths}")

        total_transitions = int(file_stream.attrs["total_transitions"])
        total_observations = int(file_stream.attrs["total_observations"])
        episode_lengths = file_stream["index/episode_lengths"][:]
        if total_transitions != int(np.sum(episode_lengths)):
            raise RuntimeError("Transition count does not match the sum of episode lengths.")
        expected_observations = total_transitions + len(episode_lengths)
        if total_observations != expected_observations:
            raise RuntimeError(
                f"Observation count must equal transitions + episodes. Got {total_observations}, "
                f"expected {expected_observations}."
            )

        obs_index = file_stream["index/obs_index"]
        next_obs_index = file_stream["index/next_obs_index"]
        if obs_index.shape != next_obs_index.shape:
            raise RuntimeError("obs_index and next_obs_index must have the same shape.")
        if obs_index.shape[0] != total_transitions:
            raise RuntimeError("obs_index length must equal total transitions.")
        if total_transitions > 0:
            if not np.all(next_obs_index[:] == obs_index[:] + 1):
                raise RuntimeError("next_obs_index must be exactly obs_index + 1.")
            if int(np.max(next_obs_index[:])) >= total_observations:
                raise RuntimeError("next_obs_index points past the observation timeline.")

    @classmethod
    def _validate_reopen(cls, path: Path):
        with h5py.File(path, "r") as file_stream:
            if file_stream.attrs.get("writer_status") != "complete":
                raise RuntimeError("Compressed RGB-D staging file did not finalize cleanly.")
            if file_stream.attrs.get("schema_name") != SCHEMA_NAME:
                raise RuntimeError("Compressed RGB-D staging file schema mismatch after reopen.")
            if "observations/rgb_jpeg" not in file_stream or "observations/depth_jp2" not in file_stream:
                raise RuntimeError("Compressed RGB-D staging file is missing visual datasets after reopen.")

    def _fsync_if_possible(self):
        if self._hdf5_file_stream is None:
            return
        try:
            handle = self._hdf5_file_stream.id.get_vfd_handle()
            if isinstance(handle, int):
                os.fsync(handle)
        except Exception:
            pass


def depth_psnr_db(
    reference_depth_m: np.ndarray,
    decoded_depth_u16: np.ndarray,
    min_m: float = 0.1,
    max_m: float = 6.0,
) -> float:
    """Compute PSNR between metric depth and decoded uint16 millimeter depth."""

    reference = np.asarray(reference_depth_m, dtype=np.float32)
    if reference.ndim == 3 and reference.shape[-1] == 1:
        reference = reference[..., 0]
    decoded_m = np.asarray(decoded_depth_u16, dtype=np.float32) * 0.001
    valid = np.isfinite(reference) & (reference >= min_m) & (reference <= max_m) & (decoded_depth_u16 != 0)
    if not np.any(valid):
        return math.nan
    mse = float(np.mean((reference[valid] - decoded_m[valid]) ** 2))
    if mse == 0.0:
        return math.inf
    return 20.0 * math.log10(max_m / math.sqrt(mse))
