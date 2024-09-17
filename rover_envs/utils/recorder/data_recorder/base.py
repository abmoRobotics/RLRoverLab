from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class DataRecorderBase:
    """
    Base class for managing the recording of environment interaction data.

    Args:
    num_envs: The number of environments that are being simulated.
    extras: Optional dictionary for additional datasets to be included.

    Attributes:
    data_buffers: A dictionary of lists that serves as the data buffer for each rover.
    """

    def __init__(self, num_envs: int, extras: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.num_envs = num_envs
        self.extras = extras or {}
        self.data_buffers = {rover_id: self._init_buffer() for rover_id in range(self.num_envs)}

    def _init_buffer(self) -> Dict[str, List]:
        """Initialize the data buffer."""
        buffer = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "terminated": [],
        }
        buffer.update({key: [] for key in self.extras.keys()})
        return buffer

    def append_to_buffer(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        info: Dict[str, torch.Tensor]
    ) -> None:
        """Append data to the buffer."""
        obs, action, reward, done, info = self._pre_process(obs, action, reward, done, info)
        for rover_id in range(self.num_envs):
            self.data_buffers[rover_id]["observations"].append(obs[rover_id])
            self.data_buffers[rover_id]["actions"].append(action[rover_id])
            self.data_buffers[rover_id]["rewards"].append(reward[rover_id])
            self.data_buffers[rover_id]["terminated"].append(done[rover_id])
            for key in self.extras.keys():
                self.data_buffers[rover_id][key].append(info[key][rover_id])

            if done[rover_id]:
                self.write_to_disk(rover_id)
                self.data_buffers[rover_id] = self._init_buffer()

    def _pre_process(self,
                     obs: Union[torch.Tensor, np.ndarray],
                     action: Union[torch.Tensor, np.ndarray],
                     reward: Union[torch.Tensor, np.ndarray],
                     done: Union[torch.Tensor, np.ndarray],
                     info: Dict[str, Any]
                     ) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
        """Pre-process data before writing to disk."""
        pre_proccess_arrays = [obs, action, reward, done]
        for key in self.extras.keys():
            pre_proccess_arrays.append(info[key])

        for i, array in enumerate(pre_proccess_arrays):
            if isinstance(array, torch.Tensor):
                pre_proccess_arrays[i] = array.cpu().numpy()
            elif isinstance(array, list):
                pre_proccess_arrays[i] = np.array(array)

        obs, action, reward, done = pre_proccess_arrays[:4]
        info = {key: pre_proccess_arrays[i+4] for i, key in enumerate(self.extras.keys())}
        return obs, action, reward, done, info

    def write_to_disk(self, rover_id: int) -> None:
        """
        Writes buffered data for a given rover to disk. This method should be implemented by derived classes.
        """
        raise NotImplementedError("This method should be implemented by a derived class.")

    def flush(self) -> None:
        """Writes any remaining data in the buffers to disk. Should be implemented by derived classes."""
        raise NotImplementedError("This method should be implemented by a derived class.")
