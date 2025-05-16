from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch

class DataCollector(gym.Wrapper):

    def __init__(self, env: gym.Env, recorder: Any):
        super().__init__(env)
        self.recorder = recorder
        self._last_action: Union[np.ndarray, torch.Tensor, None] = None
        self._last_observation: Union[np.ndarray, torch.Tensor, None] = None

    def step(self, action: Union[np.ndarray, torch.Tensor]):
        """
        Take a step in the environment with the given action.
        """
        pass

    def reset(self, **kwargs):
        """
        Reset the environment and return the initial observation.
        """
        pass

    def close(self):
        """
        Close the environment and release any resources.
        """
        pass