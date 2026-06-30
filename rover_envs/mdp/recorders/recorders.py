from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import warp as wp
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers.recorder_manager import RecorderTerm
# We define recorders for (o_t, a_t, r_t, o_t+1, d_t)


RGB_OBSERVATION_KEY = "rgb_image"
DEPTH_OBSERVATION_KEY = "depth_image"
CAMERA_SENSOR_NAME = "tiled_camera"


def _to_torch(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return wp.to_torch(value)


def _quat_xyzw_to_roll_pitch(quat_xyzw: torch.Tensor) -> torch.Tensor:
    x = quat_xyzw[:, 0]
    y = quat_xyzw[:, 1]
    z = quat_xyzw[:, 2]
    w = quat_xyzw[:, 3]

    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = torch.clamp(2.0 * (w * y - z * x), min=-1.0, max=1.0)
    pitch = torch.asin(sin_pitch)
    return torch.stack((roll, pitch), dim=-1).to(dtype=torch.float32)


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


class RockDistanceRecorder(RecorderTerm):
    """Records the robot distance in meters to the nearest projected rock at o_t."""

    _env: ManagerBasedRLEnv

    def record_pre_step(self):
        asset = self._env.scene[self.cfg.asset_name]
        root_pos_w = _to_torch(asset.data.root_link_pos_w)
        xy = root_pos_w[:, :2]

        terrain_manager = getattr(self._env.scene.terrain, "_terrainManager", None)
        risk_map = getattr(terrain_manager, "obstacle_risk_map", None)
        if risk_map is None:
            distances = torch.full((xy.shape[0],), float("inf"), device=xy.device, dtype=torch.float32)
        else:
            distances = risk_map.distance_at_world_xy(xy).to(device=xy.device, dtype=torch.float32)

        return "risk/min_distance_to_rock", distances


class RoverAttitudeRecorder(RecorderTerm):
    """Records rover roll and pitch in radians at o_t as [roll, pitch]."""

    _env: ManagerBasedRLEnv

    def record_pre_step(self):
        asset = self._env.scene[self.cfg.asset_name]
        root_quat_w = _to_torch(asset.data.root_link_quat_w)
        return "robot/roll_pitch", _quat_xyzw_to_roll_pitch(root_quat_w)


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
            observations[RGB_OBSERVATION_KEY] = self._slice_env_ids(sensor_output["rgb"].clone(), env_ids)
        if DEPTH_OBSERVATION_KEY in observations:
            observations[DEPTH_OBSERVATION_KEY] = self._slice_env_ids(sensor_output["depth"].clone(), env_ids)
        return observations

class NextObservationRecorder(RecorderTerm):
    """
    Records the next observations received at each step in the environment (o_t+1)
    """
    _env: ManagerBasedRLEnv
    
    def record_post_step(self):
        return "next_obs", self._env.obs_buf["policy"]
