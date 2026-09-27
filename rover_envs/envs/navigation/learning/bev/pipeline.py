"""The rover's persistent BEV as a policy observation (Phase 4).

Everything here is built from the rover's own sensors, per environment and step. The frozen Phase-1 model turns the
camera frame into an instantaneous BEV. Odometry of unknown drift places it in the world-aligned memory, the drift
monitor measures that drift from re-observed terrain, and the Phase-3 reliability map turns this into an expected
elevation error per cell. The result is an image [B, C, X, Y] of the chosen variant's channels, on the ``BevCfg`` grid.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from rover_envs.envs.navigation.learning.bev.drift import DriftMonitor, local_relief, nominal_effect
from rover_envs.envs.navigation.learning.bev.memory import CURRENT, UNKNOWN, BevMemory
from rover_envs.envs.navigation.learning.bev.models import load_bev_model
from rover_envs.envs.navigation.learning.bev.odometry import OdometryDrift

# Channels of each variant. Elevation is in metres from the BEV frame origin, the rock probability in [0, 1], masks are
# 0 or 1, and the expected elevation error is in decimetres. Every channel is zero where the cell is unknown.
VARIANTS = {
    "instantaneous": ("elevation", "rock", "observed"),
    "persistent": ("elevation", "rock", "known"),
    "full": ("elevation", "rock", "known", "current", "expected_error"),
}


class ReliabilityMap:
    """Expected elevation error of the cells of the persistent BEV (Phase 3).

    A monotone map, exported by ``analyze_bev_reliability.py --export``, of a score that adds two variances: the
    nominal pose effect at the drift level the rover measures, and the weighted squared disagreement between the
    network's elevation and the raw depth lift.
    """

    def __init__(self, path: str, device: str | torch.device = "cpu"):
        data = np.load(path)
        self.edges = torch.tensor(data["edges"], dtype=torch.float32, device=device)
        self.values = torch.tensor(data["values"], dtype=torch.float32, device=device)
        self.weight, self.prior = float(data["weight"]), float(data["prior"])

    def __call__(
        self, persistent: dict[str, torch.Tensor], level: torch.Tensor, cell_range: torch.Tensor, resolution: float
    ) -> torch.Tensor:
        """Expected error [B, X, Y] in metres, NaN where unknown.

        Args:
            persistent: ``BevMemory.step`` output with a disagreement quality channel.
            level: The measured drift level [B] (``DriftMonitor``); where it is NaN, the map's prior stands in.
            cell_range: Distance [X, Y] of each ego cell from the rover.
            resolution: Cell size in metres.
        """
        level = torch.where(torch.isfinite(level), level, self.prior).view(-1, 1, 1)
        slope = local_relief(persistent["elevation"]) / resolution
        effect = nominal_effect(slope, persistent["distance"].nan_to_num(), cell_range)
        score = level**2 * effect + self.weight * persistent["quality"][:, 0].nan_to_num() ** 2
        index = torch.searchsorted(self.edges, score.contiguous(), right=True) - 1
        expected = self.values[index.clamp(0, len(self.values) - 1)]
        return torch.where(persistent["status"] != UNKNOWN, expected, torch.nan)


class PersistentBev:
    """The policy's BEV for a batch of environments: model, drifting odometry, memory, drift monitor and reliability.

    The memory uses the settled Phase-3 configuration: it stores the raw depth-lift elevation, with the network-lift
    disagreement as its quality, and averages the class probabilities. ``severity`` is the odometry drift, as for
    ``OdometryDrift``: a (low, high) range drawn per episode, one multiplier, or 0 for exact poses; ``field_std`` and
    ``field_length`` vary it with location. The policy never sees the level.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_envs: int,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        resolution: float,
        classes: tuple[str, ...],
        dt: float,
        variant: str = "full",
        reliability: ReliabilityMap | None = None,
        severity: float | tuple[float, float] = (0.25, 4.0),
        field_std: float = 0.5,
        field_length: float = 5.0,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'; choose from {list(VARIANTS)}.")
        if "expected_error" in VARIANTS[variant] and reliability is None:
            raise ValueError(f"The '{variant}' variant needs a reliability map.")
        self.model, self.variant, self.reliability, self.resolution = model, variant, reliability, resolution
        self.rock = list(classes).index("rock")
        self.memory = BevMemory(
            num_envs, len(classes), x_range, y_range, resolution, dt, class_fusion="mean", quality_channels=1,
            device=device,
        )
        self.monitor = DriftMonitor(num_envs, resolution, device=device)
        self.odometry = None
        if severity:
            self.odometry = OdometryDrift(num_envs, severity, dt, seed, device, field_std, field_length)
        self.last: dict = {}

    @classmethod
    def from_cfg(cls, cfg, bev_cfg, num_envs: int, dt: float, seed: int, device) -> PersistentBev:
        """The pipeline of a task's ``PolicyBevCfg`` ``cfg`` on its ``BevCfg`` grid ``bev_cfg``."""
        model = load_bev_model(os.path.expanduser(cfg.model_path), device)
        grid = (tuple(bev_cfg.x_range), tuple(bev_cfg.y_range), bev_cfg.resolution)
        if (tuple(model.x_range), tuple(model.y_range), model.resolution) != grid:
            raise ValueError(f"The BEV model's grid does not match the task's {bev_cfg}.")
        reliability = None
        if "expected_error" in VARIANTS[cfg.variant]:
            reliability = ReliabilityMap(os.path.expanduser(cfg.reliability_path), device)
        severity = tuple(cfg.severity) if cfg.severity[0] != cfg.severity[1] else cfg.severity[0]
        return cls(
            model, num_envs, *grid, tuple(bev_cfg.classes), dt, cfg.variant, reliability, severity, cfg.field_std,
            cfg.field_length, seed, device,
        )

    @property
    def channels(self) -> tuple[str, ...]:
        return VARIANTS[self.variant]

    @torch.inference_mode()
    def step(
        self, rgb: torch.Tensor, depth: torch.Tensor, camera: torch.Tensor, pose: torch.Tensor, reset: torch.Tensor
    ) -> torch.Tensor:
        """The policy BEV [B, C, X, Y] after the current frame.

        Args:
            rgb, depth, camera: The BEV task's observation groups of the same name.
            pose: The exact BEV frame pose [B, 4] (the ``pose`` group); the memory gets it through the drifting
                odometry.
            reset: [B], true where the environment restarted in this step. Its frame was rendered before the reset
                moved the rover, so it is not used.
        """
        self.memory.reset(reset)
        self.monitor.reset(reset)
        estimate = self.odometry(pose, reset) if self.odometry is not None else pose
        elevation, logits, lifted = self.model(rgb, depth, camera)
        evidence = (lifted[:, 2] > 0) & ~reset.view(-1, 1, 1)
        probabilities = logits.softmax(dim=1)
        disagreement = (elevation - lifted[:, 0]).abs().unsqueeze(1)
        travelled = self.memory.travelled.clone()
        persistent = self.memory.step(estimate, lifted[:, 0], probabilities, evidence, disagreement)
        drift = self.monitor.step(persistent, self.memory.travelled - travelled)
        self.last = {"persistent": persistent, "drift": drift, "pose": estimate}

        if self.variant == "instantaneous":
            maps = {
                "elevation": torch.where(evidence, lifted[:, 0], 0.0),
                "rock": torch.where(evidence, probabilities[:, self.rock], 0.0),
                "observed": evidence.float(),
            }
        else:
            maps = {
                "elevation": persistent["elevation"].nan_to_num(),
                "rock": persistent["semantics"][:, self.rock].nan_to_num(),
                "known": (persistent["status"] != UNKNOWN).float(),
                "current": (persistent["status"] == CURRENT).float(),
            }
            if self.reliability is not None:
                expected = self.reliability(persistent, drift["level"], self.memory.cell_range, self.resolution)
                maps["expected_error"] = 10.0 * expected.nan_to_num()
        return torch.stack([maps[name] for name in self.channels], dim=1)
