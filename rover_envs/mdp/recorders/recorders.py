from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers.recorder_manager import RecorderTerm
# We define recorders for (o_t, a_t, r_t, o_t+1, d_t)


RGB_OBSERVATION_KEY = "rgb_image"
DEPTH_OBSERVATION_KEY = "depth_image"
CAMERA_SENSOR_NAME = "tiled_camera"


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
        return "obs", self._env.obs_buf["policy"]
    
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


class TerminalRecorder(RecorderTerm):
    """
    Records true terminal flags separately from timeouts.
    """
    _env: ManagerBasedRLEnv

    def record_post_step(self):
        return "terminals", self._env.reset_terminated


class TimeoutRecorder(RecorderTerm):
    """
    Records time-limit truncation flags separately from true terminals.
    """
    _env: ManagerBasedRLEnv

    def record_post_step(self):
        return "timeouts", self._env.reset_time_outs


class TimelineObservationRecorder(RecorderTerm):
    """
    Records a single sequential observation timeline per episode.

    For T actions the optimized dataset stores T + 1 observations. The post-reset
    callback records o_0, and the post-step callback records o_{t+1}. No
    physical next_obs dataset is emitted.
    """
    _env: ManagerBasedRLEnv

    def record_post_reset(self, env_ids: Sequence[int] | None):
        observations = self._env.observation_manager.compute()
        policy_observations = observations["policy"] if "policy" in observations else observations
        policy_observations = self._replace_visuals_with_raw_sensor_output(policy_observations)
        return "obs", self._slice_env_ids(policy_observations, env_ids)

    def record_post_step(self):
        policy_observations = self._replace_visuals_with_raw_sensor_output(self._env.obs_buf["policy"])
        return "obs", policy_observations

    def _slice_env_ids(self, value, env_ids: Sequence[int] | None):
        if env_ids is None:
            return value
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        if isinstance(value, Mapping):
            return {key: self._slice_env_ids(sub_value, env_ids) for key, sub_value in value.items()}
        return value[env_ids]

    def _replace_visuals_with_raw_sensor_output(self, policy_observations, env_ids: Sequence[int] | None = None):
        if not isinstance(policy_observations, Mapping):
            return policy_observations

        observations = dict(policy_observations)
        if RGB_OBSERVATION_KEY not in observations and DEPTH_OBSERVATION_KEY not in observations:
            return observations

        try:
            sensor = self._env.scene.sensors[CAMERA_SENSOR_NAME]
        except KeyError as exc:
            raise KeyError(
                f"Compressed RGB-D timeline recording expected camera sensor '{CAMERA_SENSOR_NAME}'."
            ) from exc

        sensor_output = sensor.data.output
        if RGB_OBSERVATION_KEY in observations:
            observations[RGB_OBSERVATION_KEY] = self._slice_env_ids(sensor_output["rgb"].torch.clone(), env_ids)
        if DEPTH_OBSERVATION_KEY in observations:
            observations[DEPTH_OBSERVATION_KEY] = self._slice_env_ids(sensor_output["depth"].torch.clone(), env_ids)
        return observations

class NextObservationRecorder(RecorderTerm):
    """
    Records the next observations received at each step in the environment (o_t+1)
    """
    _env: ManagerBasedRLEnv
    
    def record_post_step(self):
        return "next_obs", self._env.obs_buf["policy"]
