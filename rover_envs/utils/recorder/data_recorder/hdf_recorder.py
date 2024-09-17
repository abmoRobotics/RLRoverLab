from typing import Any, Dict, Optional

import gymnasium as gym
import h5py
import numpy as np

from .base import DataRecorderBase


class HDF5DataRecorder(DataRecorderBase):

    def __init__(self,
                 base_filename: str,
                 num_envs: int,
                 env: gym.Env,
                 extras: Optional[Dict[str, Dict[str, Any]]] = None,
                 max_rows: int = 500_000) -> None:
        super().__init__(num_envs, extras)

        # Check that basefile name does not end with .h5 or .hdf5
        assert not base_filename.endswith(".h5") and not base_filename.endswith(".hdf5"), \
            "Base filename should not end with .h5 or .hdf5"
        self.base_filename = base_filename
        self.max_rows = max_rows
        self.current_row = 0
        self.current_file_index = 0
        self.num_observations = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0] if not isinstance(env.action_space, gym.spaces.Discrete) else 1
        self.env = env
        self._create_new_file()

    def _create_new_file(self) -> None:

        self.file_name = f"{self.base_filename}_{self.current_file_index}.h5"
        self.current_file_index += 1

        with h5py.File(self.file_name, "w") as file:
            file.create_dataset("observations", (self.max_rows, self.num_observations),
                                dtype=self.env.observation_space.dtype)
            file.create_dataset("actions", (self.max_rows, self.num_actions), dtype=self.env.action_space.dtype)
            file.create_dataset("rewards", (self.max_rows, 1), dtype=np.float32)
            file.create_dataset("terminated", (self.max_rows, 1), dtype=bool)

            for key, param in self.extras.items():
                file.create_dataset(
                    key,
                    (self.max_rows, *param["shape"]),
                    dtype=param["dtype"]
                )

            file.attrs["number_of_steps"] = 0

    def write_to_disk(self, rover_id: int) -> None:
        data_chunk = {key: np.array(value) for key, value in self.data_buffers[rover_id].items()}
        next_index = self.current_row + len(data_chunk["observations"])

        if next_index > self.max_rows:
            self._create_new_file()
            self.current_row = 0
            next_index = len(data_chunk["observations"])

        with h5py.File(self.file_name, "a") as file:
            for key, value in data_chunk.items():
                file[key][self.current_row:next_index] = value
            file.attrs["number_of_steps"] += len(data_chunk["observations"])

        self.current_row = next_index
        self.data_buffers[rover_id] = self._init_buffer()

    def flush(self) -> None:
        """ Writes any remaining data in the buffers to disk. """
        for rover_id in range(self.num_envs):
            if len(self.data_buffers[rover_id]["observations"]) > 0:
                self.write_to_disk(rover_id)

    def _truncate_datasets(self) -> None:
        """ Truncates the datasets to the number of steps that have been recorded. """
        with h5py.File(self.file_name, "a") as file:
            for key in self.data_buffers.keys():
                file[key].resize(file.attrs["number_of_steps"], axis=0)

    def close(self) -> None:
        self.flush()
        self._truncate_datasets()

    def __exit__(self, exc_type, exc_value, traceback):
        """ Flushes the data buffers and truncates the datasets. """
        self.close()
