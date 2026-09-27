"""World-aligned memory that turns instantaneous BEVs into a persistent ego BEV (Phase 2).

For each environment the memory keeps the latest observation of every world cell in a square window around the rover.
The window is aligned with the world axes and wraps around: world cell (gx, gy) = floor(world xy / resolution) lives in
slot (gx mod size, gy mod size), together with its cell index to tell it from older cells that shared the slot. The
memory therefore never shifts or resamples as the rover moves. An observation is resampled twice: when written (each
world cell takes its nearest ego cell) and when read (each ego cell takes the world cell under its centre). The
resulting position error stays below about one cell and does not accumulate.

Elevations are stored as world heights. The ego BEV measures them from its frame origin, so the pose's z converts them.

Besides the latest observation, the memory keeps the first observation of every world cell. Comparing the two while the
rover drives shows how far its view of the same terrain moves, i.e. how much the odometry drifts, without ground truth.

Poses are the world pose [B, 4] of the ego BEV frame: x, y, z in metres and the yaw in radians (see
``mdp.bev_frame_pose``). Ego BEV maps have the layout [x, y] of ``BevCfg``.
"""

from __future__ import annotations

import math

import torch

UNKNOWN, CURRENT, REMEMBERED = 0, 1, 2
"""Provenance of a persistent BEV cell."""

_EMPTY = torch.iinfo(torch.int64).min


class BevMemory:
    """Persistent ego BEV from instantaneous BEVs and poses, for a batch of environments.

    ``step`` returns the persistent BEV at the current pose and then stores the cells with current observational
    evidence. Only evidence is ever stored: predictions for cells without evidence never enter the memory.

    Write rules for the elevation: ``"latest"`` replaces a stored cell with every new observation. ``"closest"`` keeps
    the observation made closest to the rover, which is usually the more accurate one.

    Class fusion: ``"latest"`` stores the class probabilities together with the elevation. The fusions use every
    observation of a world cell, also those the closest rule skips. ``"mean"`` averages the probabilities; with
    ``fusion_window`` > 0 it weights the last ``fusion_window`` observations equally at first and then becomes an
    exponential average, so it can follow changes. ``"log_odds"`` is a Bayesian update with independent observations,
    which multiplies the probabilities; no class drops below about ``min_probability``, so a few contrary observations
    can still change the class.
    """

    def __init__(
        self,
        num_envs: int,
        num_classes: int,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        resolution: float,
        dt: float,
        size: int = 240,
        write_rule: str = "latest",
        class_fusion: str = "latest",
        fusion_window: int = 0,
        min_probability: float = 0.02,
        device: str | torch.device = "cpu",
    ):
        corner = max(math.hypot(x, y) for x in x_range for y in y_range)
        if size * resolution / 2 < corner:
            raise ValueError(f"A {size} cell window cannot hold the ego BEV, which reaches {corner:.2f} m away.")
        if write_rule not in ("latest", "closest"):
            raise ValueError(f"Unknown write rule '{write_rule}'.")
        if class_fusion not in ("latest", "mean", "log_odds"):
            raise ValueError(f"Unknown class fusion '{class_fusion}'.")
        self.x_range, self.y_range, self.resolution, self.dt = tuple(x_range), tuple(y_range), resolution, dt
        self.size, self.write_rule = size, write_rule
        self.class_fusion, self.fusion_window, self.min_probability = class_fusion, fusion_window, min_probability
        self.shape = (round((x_range[1] - x_range[0]) / resolution), round((y_range[1] - y_range[0]) / resolution))
        x = x_range[0] + (torch.arange(self.shape[0], device=device) + 0.5) * resolution
        y = y_range[0] + (torch.arange(self.shape[1], device=device) + 0.5) * resolution
        self.cell_x, self.cell_y = (c.flatten() for c in torch.meshgrid(x, y, indexing="ij"))
        self.cell_range = torch.hypot(self.cell_x, self.cell_y).view(self.shape)

        slots = size * size
        self.height = torch.zeros(num_envs, slots, device=device)
        self.semantics = torch.zeros(num_envs, num_classes, slots, device=device)
        self.count = torch.zeros(num_envs, slots, device=device)
        self.cell = torch.full((num_envs, 2, slots), _EMPTY, dtype=torch.int64, device=device)
        self.time = torch.zeros(num_envs, slots, device=device)
        self.odometer = torch.zeros(num_envs, slots, device=device)
        self.range = torch.zeros(num_envs, slots, device=device)
        self.first_height = torch.zeros(num_envs, slots, device=device)
        self.first_time = torch.zeros(num_envs, slots, device=device)
        self.first_odometer = torch.zeros(num_envs, slots, device=device)
        self.first_range = torch.zeros(num_envs, slots, device=device)
        self.clock = torch.zeros(num_envs, device=device)
        self.travelled = torch.zeros(num_envs, device=device)
        self.last_xy = torch.full((num_envs, 2), torch.nan, device=device)

    def reset(self, envs: torch.Tensor) -> None:
        """Forget everything about the environments where ``envs`` [B] is true."""
        self.cell[envs] = _EMPTY
        self.clock[envs] = 0.0
        self.travelled[envs] = 0.0
        self.last_xy[envs] = torch.nan

    def step(
        self, pose: torch.Tensor, elevation: torch.Tensor, semantics: torch.Tensor, evidence: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Advance one time step: read the persistent BEV at ``pose``, then store the current evidence cells.

        Args:
            pose: World pose [B, 4] of the current ego BEV frame.
            elevation: Current elevation [B, X, Y] in metres from the BEV frame origin.
            semantics: Current per-class values [B, K, X, Y], e.g. class probabilities.
            evidence: [B, X, Y], true where a current observation supports the cell. Only these cells are stored.

        Returns:
            ``elevation`` [B, X, Y] and ``semantics`` [B, K, X, Y], NaN where unknown; ``status`` [B, X, Y] with
            UNKNOWN, CURRENT or REMEMBERED; ``age`` in seconds and ``distance`` travelled in metres since the cell's
            elevation was observed, both [B, X, Y], zero for current cells and NaN where unknown; ``range`` [B, X, Y],
            the distance in metres from the rover at which that observation was made, NaN where unknown.
            The cell's first observation, for cells observed before this step and NaN elsewhere: ``first_elevation``
            [B, X, Y] in the current frame, and the ``first_age``, ``first_distance`` and ``first_range`` since and at
            which it was made.
        """
        self.clock += self.dt
        self.travelled += torch.nan_to_num((pose[:, :2] - self.last_xy).norm(dim=1))
        self.last_xy = pose[:, :2].clone()

        # Read before writing, so that remembered cells only come from earlier steps.
        known, height, stored_semantics, time, odometer, stored_range, *first = self._read(pose)
        remembered = known & ~evidence
        status = torch.where(evidence, CURRENT, torch.where(remembered, REMEMBERED, UNKNOWN))
        persistent = {
            "elevation": torch.where(evidence, elevation, torch.where(remembered, height, torch.nan)),
            "semantics": torch.where(
                evidence.unsqueeze(1), semantics, torch.where(remembered.unsqueeze(1), stored_semantics, torch.nan)
            ),
            "status": status,
            "age": torch.where(evidence, 0.0, torch.where(remembered, self.clock.view(-1, 1, 1) - time, torch.nan)),
            "distance": torch.where(
                evidence, 0.0, torch.where(remembered, self.travelled.view(-1, 1, 1) - odometer, torch.nan)
            ),
            "range": torch.where(evidence, self.cell_range, torch.where(remembered, stored_range, torch.nan)),
        }
        first_height, first_time, first_odometer, first_range = first
        persistent["first_elevation"] = torch.where(known, first_height, torch.nan)
        persistent["first_age"] = torch.where(known, self.clock.view(-1, 1, 1) - first_time, torch.nan)
        persistent["first_distance"] = torch.where(known, self.travelled.view(-1, 1, 1) - first_odometer, torch.nan)
        persistent["first_range"] = torch.where(known, first_range, torch.nan)
        self._write(pose, elevation, semantics, evidence)
        return persistent

    def _read(self, pose: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The stored world cell under each ego cell centre, converted to the ego BEV at ``pose``."""
        x, y, z, yaw = pose.unbind(1)
        cos, sin = yaw.cos().unsqueeze(1), yaw.sin().unsqueeze(1)
        cell_x = torch.floor((x.unsqueeze(1) + cos * self.cell_x - sin * self.cell_y) / self.resolution).long()
        cell_y = torch.floor((y.unsqueeze(1) + sin * self.cell_x + cos * self.cell_y) / self.resolution).long()
        slot = (cell_x % self.size) * self.size + cell_y % self.size
        known = (self.cell[:, 0].gather(1, slot) == cell_x) & (self.cell[:, 1].gather(1, slot) == cell_y)
        height = self.height.gather(1, slot) - z.unsqueeze(1)
        semantics = self.semantics.gather(2, slot.unsqueeze(1).expand(-1, self.semantics.shape[1], -1))
        first_height = self.first_height.gather(1, slot) - z.unsqueeze(1)
        stored = (self.time, self.odometer, self.range)
        first = (self.first_time, self.first_odometer, self.first_range)
        maps = (
            known, height, semantics, *(values.gather(1, slot) for values in stored),
            first_height, *(values.gather(1, slot) for values in first),
        )
        return tuple(m.view(*m.shape[:-1], *self.shape) for m in maps)

    def _write(self, pose: torch.Tensor, elevation: torch.Tensor, semantics: torch.Tensor, evidence: torch.Tensor):
        """Store the nearest ego cell of every world cell in the window around the rover that has evidence."""
        batch = pose.shape[0]
        x, y, z, yaw = pose.unbind(1)
        # Each slot holds the one world cell of the window [centre - size/2, centre + size/2) that maps to it.
        start = torch.floor(pose[:, :2] / self.resolution).long() - self.size // 2
        cells = start.unsqueeze(2) + (torch.arange(self.size, device=pose.device) - start.unsqueeze(2)) % self.size
        dx = ((cells[:, 0] + 0.5) * self.resolution - x.unsqueeze(1)).unsqueeze(2)
        dy = ((cells[:, 1] + 0.5) * self.resolution - y.unsqueeze(1)).unsqueeze(1)
        cos, sin = yaw.cos().view(-1, 1, 1), yaw.sin().view(-1, 1, 1)
        ego_x, ego_y = cos * dx + sin * dy, -sin * dx + cos * dy
        ix = torch.floor((ego_x - self.x_range[0]) / self.resolution).long()
        iy = torch.floor((ego_y - self.y_range[0]) / self.resolution).long()
        inside = (ix >= 0) & (ix < self.shape[0]) & (iy >= 0) & (iy < self.shape[1])
        index = (ix.clamp(0, self.shape[0] - 1) * self.shape[1] + iy.clamp(0, self.shape[1] - 1)).view(batch, -1)
        observed = inside.view(batch, -1) & evidence.view(batch, -1).gather(1, index)

        cell_x = cells[:, 0].unsqueeze(2).expand(-1, -1, self.size).reshape(batch, -1)
        cell_y = cells[:, 1].unsqueeze(1).expand(-1, self.size, -1).reshape(batch, -1)
        same_cell = (self.cell[:, 0] == cell_x) & (self.cell[:, 1] == cell_y)
        distance = torch.hypot(ego_x, ego_y).view(batch, -1)
        write = observed
        if self.write_rule == "closest":
            write = observed & (~same_cell | (distance <= self.range))

        classes = semantics.shape[1]
        values = semantics.view(batch, classes, -1).gather(2, index.unsqueeze(1).expand(-1, classes, -1))
        if self.class_fusion == "latest":
            self.semantics = torch.where(write.unsqueeze(1), values, self.semantics)
        else:
            # A slot that another world cell held before starts over from the new observation.
            fuse = (observed & same_cell).unsqueeze(1)
            if self.class_fusion == "mean":
                count = torch.where(same_cell, self.count + 1, 1.0)
                if self.fusion_window:
                    count = count.clamp(max=self.fusion_window)
                fused = torch.where(fuse, self.semantics + (values - self.semantics) / count.unsqueeze(1), values)
                self.count = torch.where(observed, count, self.count)
            else:
                log_odds = values.clamp(min=1e-6).log() + torch.where(fuse, self.semantics.clamp(min=1e-6).log(), 0.0)
                fused = log_odds.softmax(dim=1).clamp(min=self.min_probability)
                fused = fused / fused.sum(dim=1, keepdim=True)
            self.semantics = torch.where(observed.unsqueeze(1), fused, self.semantics)

        height = elevation.view(batch, -1).gather(1, index) + z.unsqueeze(1)
        first = observed & ~same_cell
        self.first_height = torch.where(first, height, self.first_height)
        self.first_time = torch.where(first, self.clock.unsqueeze(1), self.first_time)
        self.first_odometer = torch.where(first, self.travelled.unsqueeze(1), self.first_odometer)
        self.first_range = torch.where(first, distance, self.first_range)
        self.height = torch.where(write, height, self.height)
        self.cell = torch.where(write.unsqueeze(1), torch.stack((cell_x, cell_y), dim=1), self.cell)
        self.time = torch.where(write, self.clock.unsqueeze(1), self.time)
        self.odometer = torch.where(write, self.travelled.unsqueeze(1), self.odometer)
        self.range = torch.where(write, distance, self.range)
