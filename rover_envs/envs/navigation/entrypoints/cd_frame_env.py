from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera

from .rover_env import RoverEnv


class _CDFrameObservationView:
    """Expose CD observations as byte images for Isaac Lab's live observation viewer."""

    def __init__(self, manager):
        self._manager = manager

    def __getattr__(self, name):
        return getattr(self._manager, name)

    def get_active_iterable_terms(self, env_idx: int):
        terms = []
        for name, values in self._manager.get_active_iterable_terms(env_idx=env_idx):
            if name != "policy-rgb_image":
                terms.append((name, values))
                continue

            image = np.asarray(values)
            if image.ndim != 3 or image.shape[-1] not in (1, 2):
                terms.append((name, values))
                continue

            channel_names = ("left", "right") if image.shape[-1] == 2 else (None,)
            for channel_index, channel_name in enumerate(channel_names):
                channel = image[..., channel_index : channel_index + 1]
                grayscale = np.rint(np.clip(channel, 0.0, 1.0) * 255).astype(np.uint8)
                rgba = np.repeat(grayscale, 4, axis=-1)
                rgba[..., 3] = 255
                label = f"{name}-{channel_name}" if channel_name is not None else name
                terms.append((label, rgba))
        return terms


class CDFrameEnv(RoverEnv):
    """Rover environment with independent NIGHTRIDER pipelines for its configured cameras."""

    def __init__(self, *args, **kwargs):
        self._event_camera_interfaces = {}
        super().__init__(*args, **kwargs)

    def setup_manager_visualizers(self):
        super().setup_manager_visualizers()
        self.manager_visualizers["observation_manager"]._manager = _CDFrameObservationView(self.observation_manager)

    def _update_event_camera_from_physics_step(self):
        for sensor_name in self.cfg.event_camera_sensor_names:
            if sensor_name not in self.scene.sensors:
                continue
            camera_interface = _get_event_camera_interface(
                env=self,
                enable_ui=False,
                device=None,
                view_mode="tiled",
                env_index=0,
                sensor_name=sensor_name,
            )
            camera_interface.update(dt_us=int(round(self.physics_dt * 1e6)))

    def step(self, action: torch.Tensor):
        """Step physics and update NIGHTRIDER at physics rate before computing observations."""
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.is_rendering

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            self._update_event_camera_from_physics_step()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def close(self):
        for camera_interface in self._event_camera_interfaces.values():
            camera_interface.close()
        self._event_camera_interfaces.clear()
        super().close()


def _nightrider_event_dir() -> Path:
    project_dir = Path(__file__).resolve().parents[4]
    event_source_dir = project_dir / "examples" / "NIGHTRIDER" / "EventCamera"
    if event_source_dir.is_dir():
        return event_source_dir
    return project_dir / "07_gpu_event_RGB"


def _remove_foreign_event_modules(event_source_dir: Path):
    for module_name in tuple(sys.modules):
        if module_name not in ("EventCameraInterfaceGPU", "IECBSLiveGPU", "src2") and not module_name.startswith(
            "src2."
        ):
            continue
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None) if module is not None else None
        if module_file is None:
            continue
        module_path = Path(module_file).resolve()
        if not module_path.is_relative_to(event_source_dir):
            del sys.modules[module_name]


def _event_camera_interface_class(event_source_dir: Path | None = None):
    event_source_dir = (event_source_dir or _nightrider_event_dir()).resolve()
    if not event_source_dir.is_dir():
        raise ModuleNotFoundError(f"NIGHTRIDER EventCamera directory not found: {event_source_dir}")

    _remove_foreign_event_modules(event_source_dir)
    if str(event_source_dir) in sys.path:
        sys.path.remove(str(event_source_dir))
    sys.path.insert(0, str(event_source_dir))

    from EventCameraInterfaceGPU import EventCameraInterfaceGPU

    return EventCameraInterfaceGPU


def _get_event_camera_interface(
    env: CDFrameEnv,
    enable_ui: bool,
    device: str | None,
    view_mode: str,
    env_index: int,
    sensor_name: str = "tiled_camera",
):
    if sensor_name not in env._event_camera_interfaces:
        EventCameraInterfaceGPU = _event_camera_interface_class()
        interface_params = inspect.signature(EventCameraInterfaceGPU).parameters
        if sensor_name != "tiled_camera" and "sensor_name" not in interface_params:
            EventCameraInterfaceGPU = _event_camera_interface_class(
                Path(__file__).resolve().parents[4] / "07_gpu_event_RGB"
            )
            interface_params = inspect.signature(EventCameraInterfaceGPU).parameters
        event_device = device or getattr(env, "device", "cuda")
        sensor_params = {"sensor_name": sensor_name} if "sensor_name" in interface_params else {}
        env._event_camera_interfaces[sensor_name] = EventCameraInterfaceGPU(
            scene=env.scene,
            sim=env.sim,
            enable_ui=enable_ui,
            device=event_device,
            view_mode=view_mode,
            env_index=env_index,
            **sensor_params,
        )
    return env._event_camera_interfaces[sensor_name]


def cd_image(
    env: CDFrameEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
    use_event_camera: bool = True,
    event_enable_ui: bool = False,
    event_device: str | None = None,
    event_view_mode: str = "tiled",
    event_env_index: int = 0,
) -> torch.Tensor:
    """Return raw CD frames generated from the RGB tiled camera.

    For ``data_type="rgb"`` and ``use_event_camera=True``, the RGB sensor frame is
    passed through NIGHTRIDER and returned as a one-channel dense event image with
    OFF=0.0, no event=0.5, ON=1.0. The camera sensor itself remains an IsaacLab RGB
    tiled camera; only this observation term is replaced by CD frames.
    """
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    if use_event_camera and data_type == "rgb":
        camera_interface = _get_event_camera_interface(
            env=env,
            enable_ui=event_enable_ui,
            device=event_device,
            view_mode=event_view_mode,
            env_index=event_env_index,
            sensor_name=sensor_cfg.name,
        )
        get_events = getattr(camera_interface, "get_cached_events", camera_interface.get_events)
        event_images = get_events()
        if event_images is not None:
            if event_images.shape[-1] == 3:
                event_images = 0.5 + 0.5 * (event_images[..., 1:2] - event_images[..., 0:1])
            return event_images.clone()

        rgb_images = sensor.data.output[data_type]
        neutral_shape = (*rgb_images.shape[:-1], 1)
        return torch.full(neutral_shape, 0.5, dtype=torch.float32, device=rgb_images.device)

    images = sensor.data.output[data_type]

    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0
        elif "normals" in data_type:
            images = (images + 1.0) * 0.5

    return images.clone()


def stereo_cd_image(
    env: CDFrameEnv,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera_left"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera_right"),
) -> torch.Tensor:
    """Return CD frames in NHWC format with left and right as the two channels."""
    left_image = cd_image(env, sensor_cfg=left_sensor_cfg, normalize=False)
    right_image = cd_image(env, sensor_cfg=right_sensor_cfg, normalize=False)
    return torch.cat((left_image, right_image), dim=-1)
