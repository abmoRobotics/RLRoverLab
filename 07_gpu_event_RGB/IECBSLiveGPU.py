# Class taking input RGB frames from Isaac Sim and generating events
# using the IEBCS event camera model without arbiter

import torch
import numpy as np
import cv2
from src2.dvs_sensor_gpu import DvsSensorGPU
from src2.event_buffer import EventBuffer


class IECBSLiveGPU():

    def __init__(self, width, height, dvs_params=None, device="cuda"):
        """
        Initialize the IEBCS event camera simulator.
        
        Args:
            width: Camera width in pixels
            height: Camera height in pixels
            dvs_params: Optional dict to override default DVS parameters
        """
        # Default DVS parameters
        self.th_pos = 0.4        # ON threshold = 50% (ln(1.5) = 0.4)
        self.th_neg = 0.4       # OFF threshold = 50%
        self.ref_us = 100      # refractory period in us
        self.dt_us = 16667      # ~60 fps
        
        # Override with custom parameters if provided
        if dvs_params is not None:
            for key, value in dvs_params.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        # Camera dimensions
        self.width = width
        self.height = height
        self.device = torch.device(device)

        # Internal state
        self._initialized = False
        self._frame_count = 0
        self._sim_time_us = 0  # Simulation time in microseconds


        # Create the DVS sensor
        self.dvs = DvsSensorGPU(th_pos=self.th_pos, th_neg=self.th_neg, ref_us=self.ref_us, device=str(self.device))
        print("[INFO] Initializing DVS sensor...")
        

    def _initialize(self, initial_rgb_tensor: torch.Tensor):
        """
        Initialize the DVS sensor with the first frame.
        
        Args:
            initial_rgb_tensor: First RGB frame, shape (H, W, 3), range [0, 1]
        """
        # Convert RGB to luminance (LUV L-channel)
        luminance = self._rgb_to_luminance(initial_rgb_tensor)
        
        # Initialize DVS with baseline intensity
        self.dvs.init_luminance(luminance)
        
        self._initialized = True
        self._sim_time_us = 0
        print("[INFO] DVS sensor initialized with first frame")

    def _rgb_to_luminance(self, rgb_tensor: torch.Tensor) -> np.ndarray:
        """
        Convert RGB tensor to luminance array suitable for DVS.
        
        Args:
            rgb_tensor: RGB tensor of shape (H, W, 3), normalized [0, 1]
            
        Returns:
            Luminance array (H, W) in radiometric units [0, 10000]
        """
        # Convert to numpy
        if isinstance(rgb_tensor, torch.Tensor):
            rgb_np = rgb_tensor.cpu().numpy()
        else:
            rgb_np = rgb_tensor
        
        # Convert to uint8 range [0, 255]
        rgb_uint8 = (rgb_np * 255).astype(np.uint8)
        
        # Convert RGB to LUV and extract L channel (luminance)
        # OpenCV expects BGR format
        bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        luv = cv2.cvtColor(bgr, cv2.COLOR_BGR2LUV)
        luminance = luv[:, :, 0]  # L channel
        
        # Scale to radiometric units: [0, 255] -> [0, 10000]
        luminance_scaled = (luminance.astype(np.float64) / 255.0) * 1e4
        
        return luminance_scaled

    def process_frame(self, rgb_tensor: torch.Tensor, dt_us: int = None):
        """
        Process an RGB frame and generate DVS events.
        
        Args:
            rgb_tensor: RGB tensor of shape (H, W, 3), normalized [0, 1]
            dt_us: Time delta in microseconds (optional, uses self.dt if None)
            
        Returns:
            EventBuffer containing generated events
        """
        # Use default dt if not provided
        if dt_us is None:
            dt_us = self.dt_us
        
        # Initialize on first call

        if not self._initialized:
            self._initialize(rgb_tensor)
            return EventBuffer(0)  # Return empty buffer on first frame
        # Convert RGB to luminance
        luminance = self._rgb_to_luminance(rgb_tensor)
        
        # Generate events using DVS model
        events = self.dvs.step(luminance, dt_us)
        
        # Update internal time
        self._sim_time_us += dt_us
        self._frame_count += 1

        return events

    def _rgb_to_luminance_batch(self, rgb_batch: torch.Tensor) -> torch.Tensor:
        # rgb_batch: (N,H,W,3) float32 [0,1]
        w = torch.tensor([0.2126, 0.7152, 0.0722], device=rgb_batch.device).view(1,1,1,3)
        lum = (rgb_batch * w).sum(dim=-1)  # (N,H,W)
        return lum * 1e4

    def process_batch(self, rgb_batch: torch.Tensor, dt_us=None, return_sparse_events: bool = False):
        if dt_us is None:
            dt_us = self.dt_us
        if rgb_batch.dim() != 4:
            raise ValueError(f"Expected (N,H,W,3) rgb batch, got {rgb_batch.shape}")
        lum = self._rgb_to_luminance_batch(rgb_batch)
        if not self._initialized:
            self.dvs.init(lum)              # (N,H,W) torch
            self._initialized = True
            self._frame_count = 1
            self._sim_time_us = 0
            event_image = torch.zeros((*lum.shape, 3), dtype=torch.float32, device=lum.device)
            if return_sparse_events:
                events_list = [EventBuffer(0) for _ in range(lum.size(0))]
                return event_image, events_list
            return event_image
        result = self.dvs.step(lum, dt_us, return_sparse_events=return_sparse_events)
        self._sim_time_us += dt_us
        self._frame_count += 1
        return result

    def get_event_statistics(self) -> dict:
        """
        Get statistics about event generation.
        
        Returns:
            Dictionary with event statistics
        """
        return {
            "frame_count": self._frame_count,
            "sim_time_us": self._sim_time_us,
            "sim_time_s": self._sim_time_us / 1e6,
            "initialized": self._initialized,
        }

    def reset(self):
        """Reset the DVS sensor state"""
        self._initialized = False
        self._frame_count = 0
        self._sim_time_us = 0
        
        if self.event_display is not None:
            self.event_display.reset()
        
        print("[INFO] DVS sensor reset")
