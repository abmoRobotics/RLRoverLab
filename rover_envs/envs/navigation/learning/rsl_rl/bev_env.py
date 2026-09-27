"""RSL-RL environment wrapper whose student observes the rover's persistent BEV (Phase 4)."""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from rover_envs.envs.navigation.learning.bev.pipeline import PersistentBev


class PersistentBevVecEnv(VecEnv):
    """Adds the ``bev`` group of ``PersistentBev`` to an RSL-RL environment of a BEV task.

    The wrapped environment provides the ``rgb``, ``depth``, ``camera`` and ``pose`` groups the pipeline consumes. The
    runner only receives the groups in ``keep`` and ``bev``, so its rollout storage holds no camera images.
    """

    def __init__(self, env: VecEnv, pipeline: PersistentBev, keep: tuple[str, ...] = ("policy", "proprio")):
        self.env, self.pipeline, self.keep = env, pipeline, tuple(keep)
        self.num_envs, self.num_actions, self.device = env.num_envs, env.num_actions, env.device
        self.max_episode_length = env.max_episode_length
        self._observations = self._observe(env.get_observations(), torch.zeros(self.num_envs, dtype=torch.bool))

    @property
    def cfg(self):
        return self.env.cfg

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    def get_observations(self) -> TensorDict:
        return self._observations

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, rewards, dones, extras = self.env.step(actions)
        self._observations = self._observe(obs, dones.bool())
        return self._observations, rewards, dones, extras

    def close(self) -> None:
        self.env.close()

    def _observe(self, obs: TensorDict, reset: torch.Tensor) -> TensorDict:
        bev = self.pipeline.step(obs["rgb"], obs["depth"], obs["camera"], obs["pose"], reset.to(obs["pose"].device))
        groups = {group: obs[group] for group in self.keep}
        return TensorDict({**groups, "bev": bev}, batch_size=[self.num_envs], device=bev.device)
