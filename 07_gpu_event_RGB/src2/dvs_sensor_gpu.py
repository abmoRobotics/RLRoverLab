import torch
import numpy as np
from .event_buffer import EventBuffer

class DvsSensorGPU:
    def __init__(self, th_pos=0.4, th_neg=0.4, ref_us=100, device="cuda"):
        self.device = torch.device(device)
        self.th_pos = th_pos
        self.th_neg = th_neg
        self.ref_us = ref_us
        self.initialized = False
        self.log_I = None            # (N,H,W)
        self.last_event_time = None  # (N,H,W)
        self.refractory_until = None # (N,H,W)
        self.current_time_us = 0
        self.num_envs = 0
        self.height = 0
        self.width = 0

    def init(self, init_luminance_batch: torch.Tensor):
        # init_luminance_batch: (N,H,W) float32 on device
        self.num_envs, self.height, self.width = init_luminance_batch.shape
        self.log_I = torch.log(init_luminance_batch + 1e-6)
        self.last_event_time = torch.zeros_like(self.log_I, dtype=torch.int64)
        self.refractory_until = torch.zeros_like(self.log_I, dtype=torch.int64)
        self.current_time_us = 0
        self.initialized = True
        print(f"[GPU-DVS] Initialized ({self.num_envs} envs, {self.height}x{self.width}) on {self.device}")

    def step(self, luminance_batch: torch.Tensor, dt_us: int, return_sparse_events: bool = False):
        if not self.initialized:
            raise RuntimeError("DvsSensorGPU.step called before init.")
        self.current_time_us += dt_us

        new_log = torch.log(luminance_batch + 1e-6)
        delta = new_log - self.log_I

        ready = self.current_time_us > self.refractory_until
        pos_mask = (delta > self.th_pos) & ready
        neg_mask = (delta < -self.th_neg) & ready
        fire_mask = pos_mask | neg_mask

        event_image = torch.zeros(
            (*luminance_batch.shape, 3),
            dtype=torch.float32,
            device=luminance_batch.device,
        )
        event_image[..., 0] = neg_mask.float()
        event_image[..., 1] = pos_mask.float()
        event_image[..., 2] = fire_mask.float()

        self.log_I = torch.where(fire_mask, new_log, self.log_I)
        self.last_event_time = torch.where(
            fire_mask,
            torch.full_like(self.last_event_time, self.current_time_us),
            self.last_event_time
        )
        self.refractory_until = torch.where(
            fire_mask,
            torch.full_like(self.refractory_until, self.current_time_us + self.ref_us),
            self.refractory_until
        )

        if not return_sparse_events:
            return event_image

        # Build event buffers per env
        buffers = []
        for n in range(self.num_envs):
            pos_idx = torch.nonzero(pos_mask[n], as_tuple=False)  # (K,2) y,x
            neg_idx = torch.nonzero(neg_mask[n], as_tuple=False)
            count = pos_idx.size(0) + neg_idx.size(0)
            eb = EventBuffer(count)  # stays on CPU-compatible tensors
            if count == 0:
                buffers.append(eb)
                continue
            # Concatenate
            all_idx = torch.cat([pos_idx, neg_idx], dim=0)
            # Polarity (pos=1, neg=0)
            pol = torch.cat([
                torch.ones(pos_idx.size(0), dtype=torch.uint8, device=self.device),
                torch.zeros(neg_idx.size(0), dtype=torch.uint8, device=self.device)
            ], dim=0)
            ts = torch.full((count,), self.current_time_us, dtype=torch.int64, device=self.device)
            xs = all_idx[:, 1].to(torch.int16)
            ys = all_idx[:, 0].to(torch.int16)

            # Move to CPU numpy for existing downstream logic
            eb.x = xs.cpu()
            eb.y = ys.cpu()
            eb.p = pol.cpu()
            eb.ts = ts.cpu()
            eb.i = count
            buffers.append(eb)
        return event_image, buffers
