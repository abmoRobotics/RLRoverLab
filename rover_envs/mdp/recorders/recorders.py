from __future__ import annotations

from collections.abc import Sequence

from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
# We define recorders for (o_t, a_t, r_t, o_t+1, d_t)


class ActionRecorder(RecorderTerm):
    """
    Records the actions taken at each step in the environment (a_t)
    """
    def record_pre_step(self):
        return "actions", self._env.action_manager.action 


class ObservationRecorder(RecorderTerm):
    """
    Records the observations received at each step in the environment (o_t)
    """
    def record_pre_step(self):
        return "observations", self._env.obs_buf["policy"]
    
class RewardRecorder(RecorderTerm):
    """
    Records the rewards received at each step in the environment (r_t)
    """
    _env: ManagerBasedRLEnv
    
    def record_post_step(self):
        return "rewards", self._env.reward_manager._reward_buf
    
class DoneRecorder(RecorderTerm):
    """
    Records the done flags received at each step in the environment (d_t)
    """
    _env: ManagerBasedRLEnv
    
    def record_post_step(self):
        return "dones", self._env.reset_buf

class NextObservationRecorder(RecorderTerm):
    """
    Records the next observations received at each step in the environment (o_t+1)
    """
    _env: ManagerBasedRLEnv
    
    def record_post_step(self):
        return "next_observations", self._env.obs_buf["policy"]