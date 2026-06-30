import warnings
import torch
import numpy as np
import omni.ui as ui
from dataclasses import dataclass
from typing import Optional
from IECBSLiveGPU import IECBSLiveGPU
import time


## DEBUGGING ##
import time

@dataclass
class CameraData:
    """Container for camera data"""
    rgb: Optional[torch.Tensor] = None  # Shape: (N, H, W, 3) - normalized [0, 1]
    depth: Optional[torch.Tensor] = None  # Shape: (N, H, W, 1)
    timestamp: Optional[float] = None
    camera_name: str = "tiled_camera"
    resolution: tuple = (0, 0)  # (width, height)


#Initial image based event camera interface for RoverEnv
#Later a RayCasterCamera should be implemented:
#https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#ray-cast-camera
#https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/sensors/ray_caster/ray_caster_camera.html#RayCasterCamera

class EventCameraInterfaceGPU:

    def __init__(self, scene: "InteractiveScene", 
                 sim: "SimulationContext", 
                 enable_ui: bool = True, 
                 device="cuda", 
                 view_mode: str = "single", 
                 env_index: int = 0):
        """
        Initialize the EventCameraInterface.
        
        Args:
            scene: InteractiveScene containing the camera sensor
            sim: SimulationContext for getting physics time
            enable_ui: Whether to enable Omni UI visualization
            device: Device to run the simulation on ("cuda" or "cpu")
        """
        self._scene = scene
        self._sim = sim
        self._device = torch.device(device)
        self._ui_window = None
        self._image_provider = None
        self._status_label = None
        
        self._view_mode = view_mode  # "single" or "tiled"
        self._env_index = env_index
        self._img_w = None
        self._img_h = None

        self._display_scale = 1.0  # default (1.0 = full size)
        self._fps_last_time = time.perf_counter()
        self._fps_ema = 0.00
        self._last_event_image = None

        # Check if camera is enabled in the scene
        if not hasattr(scene, 'sensors') or 'tiled_camera' not in scene.sensors:
            warnings.warn(
                "Camera is not enabled in the environment! "
                "Please run with --enable_cameras flag.",
                UserWarning,
                stacklevel=2
            )
            self._camera_available = False
            self._camera = None
        else:
            self._camera_available = True
            self._camera = scene.sensors['tiled_camera']
            print(f"[INFO] EventCameraInterface: Camera detected at {self._camera.cfg.prim_path}")
            print(f"[INFO] Camera resolution: {self._camera.cfg.width}x{self._camera.cfg.height}")
            
            # Create UI if enabled

            if enable_ui:
                self._create_ui()

                
        # Add event camera simulator
        self._event_camera = None
        if self._camera_available:
            self._event_camera = IECBSLiveGPU(
                width=self._camera.cfg.width,
                height=self._camera.cfg.height,
                device=str(self._device)
            )
            print("[INFO] Event camera simulator initialized")



    def set_display_scale(self, scale: float):
        """Adjust UI display scale (0.1 to 1.0). Does NOT resample pixels."""
        self._display_scale = float(max(0.1, min(1.0, scale)))
        # Force recreate on next render
        self._img_w = self._img_h = None

    def _get_sim_time_s(self) -> Optional[float]:
        """Return simulation time in seconds across Isaac Sim/Isaac Lab API versions."""
        if self._sim is None:
            return None

        for attr_name in ("current_time", "current_time_s"):
            if hasattr(self._sim, attr_name):
                return float(getattr(self._sim, attr_name))

        for method_name in ("get_current_time", "get_sim_time", "get_physics_time"):
            if hasattr(self._sim, method_name):
                return float(getattr(self._sim, method_name)())

        return None

    def _update_fps(self):
        """Compute FPS via real wall-clock delta (EMA)."""
        now = time.perf_counter()
        dt = now - self._fps_last_time
        if dt > 0:
            inst = 1.0 / dt
            # EMA smoothing
            self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)
        self._fps_last_time = now

    def _create_ui(self, width=None, height=None):
        """Create/recreate UI window (scaled)."""
        print("before ui")
        base_w = width or self._camera.cfg.width
        base_h = height or self._camera.cfg.height
        disp_w = int(base_w * self._display_scale)
        disp_h = int(base_h * self._display_scale)
        # Recreate window if size changed
        print(f"Creating UI window with size: {disp_w + 40}x{disp_h + 100}")
        if self._ui_window is not None:
            self._ui_window.visible = False
            self._ui_window = None
        self._ui_window = ui.Window("Event Camera View", width=disp_w + 40, height=disp_h + 100)
        with self._ui_window.frame:
            with ui.VStack(spacing=5):
                ui.Label("Event Camera (Binary Visualization)", alignment=ui.Alignment.CENTER, height=20)
                ui.Label(f"Camera: {self._camera.cfg.prim_path}", alignment=ui.Alignment.CENTER, height=15)
                self._image_provider = ui.ByteImageProvider()
                # Width/height here are display size; provider image can be larger
                ui.ImageWithProvider(self._image_provider, width=disp_w, height=disp_h)
                self._status_label = ui.Label("Waiting for events...", alignment=ui.Alignment.CENTER, height=20)
                ui.Spacer(height=10)
        self._img_w, self._img_h = base_w, base_h  # store logical (unscaled) size

    def _get_camera_data(self) -> Optional[CameraData]:
        """
        Private method to retrieve current camera data from TiledCamera.
        
        Returns:
            CameraData object containing RGB, depth, and metadata, or None if camera unavailable.
        """
        if not self._camera_available or self._camera is None:
            warnings.warn("Camera data requested but camera is not available.", UserWarning)
            return None
        
        camera_data = self._camera.data
        
        rgb = None
        if hasattr(camera_data, 'output') and 'rgb' in camera_data.output:
            rgb_raw = camera_data.output['rgb']
            rgb = rgb_raw[..., :3].float() / 255.0
        
        
        depth = None
        if hasattr(camera_data, 'output') and 'distance_to_image_plane' in camera_data.output:
            depth = camera_data.output['distance_to_image_plane']
        
        # Get timestamp from simulation
        timestamp = self._get_sim_time_s()
     
        # Create and return CameraData object
        return CameraData(
            rgb=rgb,
            depth=depth,
            timestamp=timestamp,
            camera_name=self._camera.cfg.prim_path,
            resolution=(self._camera.cfg.width, self._camera.cfg.height)
        )
    
    def get_latest_frame(self) -> Optional[CameraData]:
        """
        Public method to get the latest camera frame.
        
        Returns:
            CameraData object or None if unavailable.
        """
        return self._get_camera_data()
    
    def _event_tensor_to_rgb(self, event_image: torch.Tensor) -> np.ndarray:
        h, w = event_image.shape[:2]
        img = np.full((h, w, 3), 125, dtype=np.uint8)

        event_image = event_image.detach()
        off_events = event_image[..., 0] > 0
        on_events = event_image[..., 1] > 0

        off_np = off_events.cpu().numpy()
        on_np = on_events.cpu().numpy()
        img[off_np] = 0
        img[on_np] = 255
        return img

    def _render_single_tensor(self, event_image_batch: torch.Tensor, current_time_us: int):
        self._update_fps()
        h, w = self._camera.cfg.height, self._camera.cfg.width
        if self._image_provider is None or self._img_w != w or self._img_h != h:
            self._create_ui(w, h)

        env_idx = min(self._env_index, event_image_batch.shape[0] - 1)
        event_image = event_image_batch[env_idx]
        img = self._event_tensor_to_rgb(event_image)
        rgba = np.concatenate([img, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2)
        self._image_provider.set_bytes_data(rgba.flatten().tolist(), [w, h])

        if self._status_label is not None:
            event_count = int(event_image[..., 2].sum().item())
            self._status_label.text = f"Events: {event_count} | t={current_time_us/1e6:.2f}s | FPS:{self._fps_ema:.2f}"

    def _render_tiled_tensor(self, event_image_batch: torch.Tensor, current_time_us: int):
        self._update_fps()
        N, H, W = event_image_batch.shape[:3]
        cols = int(np.ceil(np.sqrt(N)))
        rows = int(np.ceil(N / cols))
        tile_h, tile_w = rows * H, cols * W

        if self._image_provider is None or self._img_w != tile_w or self._img_h != tile_h:
            self._create_ui(tile_w, tile_h)

        img = np.full((tile_h, tile_w, 3), 125, dtype=np.uint8)
        total = 0
        for idx in range(N):
            r, c = divmod(idx, cols)
            y_off, x_off = r * H, c * W
            tile = self._event_tensor_to_rgb(event_image_batch[idx])
            img[y_off:y_off + H, x_off:x_off + W] = tile
            total += int(event_image_batch[idx, ..., 2].sum().item())

        rgba = np.concatenate([img, np.full((tile_h, tile_w, 1), 255, dtype=np.uint8)], axis=2)
        self._image_provider.set_bytes_data(rgba.flatten().tolist(), [tile_w, tile_h])

        if self._status_label is not None:
            self._status_label.text = f"Envs:{N} | Events(sum):{total} | t={current_time_us/1e6:.2f}s | FPS:{self._fps_ema:.2f}"

    def _render_events_if_enabled(self, event_image_batch: torch.Tensor):
        if self._ui_window is None or not self._ui_window.visible:
            return

        current_time_us = int((self._get_sim_time_s() or 0.0) * 1e6)
        if self._view_mode == "tiled":
            self._render_tiled_tensor(event_image_batch, current_time_us)
        else:
            time_start = time.time()
            self._render_single_tensor(event_image_batch, current_time_us)
            print(f"[DEBUG] Event render time: {time.time() - time_start:.4f}s")

    def _get_event_data(self, return_sparse_events: bool = False):
        if self._event_camera is None:
            return None, None
        
        time_start = time.time()
        cam = self._get_camera_data() ##1cam=0.0137s 35cam=0.5125s ## Almost linear scaling
        #print("Elapsed camera get time:", time.time() - time_start)
        if cam is None or cam.rgb is None:
            return None, None
        # cam.rgb: (N,H,W,3) torch on CPU; move to device for processing

        result = self._event_camera.process_batch(
            cam.rgb.to(self._device),
            return_sparse_events=return_sparse_events,
        ) ##1cam=0.0014s 35cam=0.0423s ## Slightly faster than linear scaling
        if return_sparse_events:
            event_image, events_list = result
        else:
            event_image = result
            events_list = None

        self._render_events_if_enabled(event_image)
        self._last_event_image = event_image
        return event_image, events_list

    def get_events(self):
        """Get dense event image observations for all envs.

        Returns:
            Tensor with shape (N, H, W, 3) on the configured device.
            Channels are: OFF events, ON events, any event.
        """
        event_image, _ = self._get_event_data(return_sparse_events=False)
        return event_image

    def get_sparse_events(self):
        """Get sparse EventBuffer data for debugging or future timestamp use."""
        _, events_list = self._get_event_data(return_sparse_events=True)
        return events_list

    def get_event_image_batch(self):
        """Alias for get_events(), kept for readability at observation call sites.
        
        Returns:
            Tensor with shape (N, H, W, 3) on the configured device.
            Channels are: OFF events, ON events, any event.
        """
        return self.get_events()

    # Optional: allow switching view at runtime
    def set_view_mode(self, mode: str, env_index: int = 0):
        assert mode in ("single", "tiled")
        self._view_mode = mode
        self._env_index = env_index
        # Force UI recreate next draw
        self._img_w = self._img_h = None

    def close(self):
        """Close the UI window"""
        if self._ui_window is not None:
            self._ui_window.visible = False
            self._ui_window = None
        # Do not call close() on IECBSLiveGPU (no such method)
        # if self._event_camera is not None:
        #     self._event_camera.close()
