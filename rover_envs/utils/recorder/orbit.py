import torch
import torch.nn as nn
import tqdm
from skrl.envs.torch.wrappers import Wrapper

from .data_recorder.base import DataRecorderBase


class SequentialCollectorOrbit:
    def __init__(self,
                 env: Wrapper,
                 model: nn.Module,
                 recorder: DataRecorderBase,
                 predict_fn=None,
                 num_episodes: int = 1000):
        self.env = env
        self.model = model
        self.recorder = recorder
        self.predict_fn = predict_fn
        self.num_episodes = num_episodes
        """ Collects data from the environment using the policy and the predict_fn."""

    def collect(self):
        with torch.no_grad():
            obs, info = self.env.reset()
            for episode in tqdm.tqdm(range(self.num_episodes)):
                action = self.predict_fn(self.model, obs)
                next_obs, reward, done, truncated, info = self.env.step(action)

                self.recorder.append_to_buffer(obs, action, reward, done, info)
                if done.any() or truncated.any():
                    obs, info = self.env.reset()
                else:
                    obs = next_obs

    def preprocess_tensors(self, obs, actions, rewards, done, info):
        return obs, actions, rewards, done, info
