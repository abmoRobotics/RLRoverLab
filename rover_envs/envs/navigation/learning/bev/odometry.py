"""Simulated odometry of unknown quality for the temporal BEV memory: exact poses corrupted by correlated drift.

The drift follows the correlated model of the earlier BEV experiments. Its size is unknown to the memory and can vary:
between runs, since each episode can draw its own severity, and along a route, since a smooth random field over the
world scales it by location, like ground on which the wheels slip more or visual odometry finds less texture.
"""

from __future__ import annotations

import math

import torch


class OdometryDrift:
    """Drifting BEV frame poses [B, 4] (x, y, z, yaw) from exact ones.

    Biases follow a first-order Gauss-Markov process with a 4 s correlation time: forward scale (1.2%), lateral drift
    (0.008 m/m), heading drift (0.008 rad/m), gyro bias (0.0015 rad/s) and, for the height reference, vertical drift
    (0.008 m/m). White noise adds 0.004 m/sqrt(m) to the translation and 0.002 rad/sqrt(m) plus 0.0007 rad/sqrt(s) to
    the heading. All standard deviations scale with the current drift ``level`` of each environment.

    The level is the run's severity times a location factor. ``severity`` is one multiplier, or a (low, high) range from
    which every run draws its own log-uniformly: at the start, stratified so that the environments cover the range
    evenly, and independently after each reset. With ``field_std`` > 0 the location
    factor is exp(field_std * g(x, y)) at the rover's true position, where g is a smooth zero-mean, unit-variance random
    field with a correlation length of ``field_length`` metres (random Fourier features of a squared-exponential
    kernel). The field is one function of the world position for all environments, like a property of the terrain;
    without it the factor is 1. ``level`` holds the current true level, for evaluation only.

    The biases are unit Gauss-Markov states scaled by the current level, so they stay continuous where it changes.
    Estimates restart from the exact pose after a reset. Private generators keep the environment's random stream, and
    so the trajectories, unchanged; a fixed severity without a field reproduces the Phase-2 drift.
    """

    BIAS_STD = (0.012, 0.008, 0.008, 0.0015, 0.008)
    """Forward scale, lateral drift (m/m), heading drift (rad/m), gyro bias (rad/s) and vertical drift (m/m) at 1x."""
    WHITE_STD = (0.004, 0.002, 0.0007)
    """Translation (m/sqrt(m)) and heading (rad/sqrt(m), rad/sqrt(s)) white noise at 1x."""

    def __init__(
        self,
        num_envs: int,
        severity: float | tuple[float, float],
        dt: float,
        seed: int,
        device: str | torch.device,
        field_std: float = 0.0,
        field_length: float = 5.0,
        field_features: int = 64,
    ):
        low, high = (severity, severity) if isinstance(severity, (int, float)) else severity
        if not 0.0 <= low <= high or (low == 0.0 < high):
            raise ValueError(f"Severity {severity} must be a non-negative number or a positive (low, high) range.")
        self.severity_range, self.dt = (float(low), float(high)), dt
        self.generator = torch.Generator(device).manual_seed(seed)
        # The severities and the field have their own generators, so the drift stream itself matches Phase 2.
        self.run_generator = torch.Generator(device).manual_seed(seed + 1)
        field_generator = torch.Generator(device).manual_seed(seed + 2)
        self.bias_std = torch.tensor(self.BIAS_STD, device=device)
        self.white_std = torch.tensor(self.WHITE_STD, device=device)
        self.correlation = math.exp(-dt / 4.0)
        self.unit_bias = self._normal(num_envs, 5)
        self.severity = self._draw_severity(num_envs, stratified=True)
        self.field_std = field_std
        self.frequencies = torch.randn(
            field_features, 2, generator=field_generator, device=field_generator.device
        ) / field_length
        self.phases = 2 * math.pi * torch.rand(field_features, generator=field_generator, device=field_generator.device)
        self.level = self.severity.clone()
        self.estimate = self.previous = None

    def _normal(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self.generator, device=self.generator.device)

    def _draw_severity(self, count: int, stratified: bool = False) -> torch.Tensor:
        """Log-uniform severities; stratified draws put one in each of ``count`` equal log intervals, shuffled."""
        low, high = self.severity_range
        if low == high:
            return torch.full((count,), low, device=self.generator.device)
        device = self.run_generator.device
        uniform = torch.rand(count, generator=self.run_generator, device=device)
        if stratified:
            uniform = (torch.randperm(count, generator=self.run_generator, device=device) + uniform) / count
        return torch.exp(math.log(low) + uniform * math.log(high / low))

    def location_factor(self, xy: torch.Tensor) -> torch.Tensor:
        """The drift factor [...] of the ground at world positions ``xy`` [..., 2]; 1 without a field."""
        if not self.field_std:
            return torch.ones(xy.shape[:-1], device=xy.device)
        field = math.sqrt(2.0 / len(self.phases)) * torch.cos(xy @ self.frequencies.T + self.phases).sum(dim=-1)
        return torch.exp(self.field_std * field)

    def __call__(self, pose: torch.Tensor, reset: torch.Tensor) -> torch.Tensor:
        """The estimated pose [B, 4] for the exact ``pose`` [B, 4]; ``reset`` [B] marks environments that restarted."""
        if self.estimate is None:
            self.estimate, self.previous = pose.clone(), pose.clone()
            self.level = self.severity * self.location_factor(pose[:, :2])
            return self.estimate.clone()
        if reset.any():
            self.severity = torch.where(reset, self._draw_severity(len(pose)), self.severity)
        self.level = self.severity * self.location_factor(pose[:, :2])
        level = self.level.unsqueeze(1)
        innovation = math.sqrt(1.0 - self.correlation**2) * self._normal(*self.unit_bias.shape)
        self.unit_bias = self.correlation * self.unit_bias + innovation
        forward_scale, lateral, heading, gyro, vertical = (level * self.bias_std * self.unit_bias).unbind(1)
        white = level * self.white_std[[0, 0, 1, 2]] * self._normal(len(pose), 4)
        # Motion since the previous step in the previous true ego frame, corrupted there.
        delta = pose[:, :2] - self.previous[:, :2]
        distance = delta.norm(dim=1)
        cos, sin = self.previous[:, 3].cos(), self.previous[:, 3].sin()
        forward = (cos * delta[:, 0] + sin * delta[:, 1]) * (1 + forward_scale)
        left = -sin * delta[:, 0] + cos * delta[:, 1] + lateral * distance
        forward = forward + distance.sqrt() * white[:, 0]
        left = left + distance.sqrt() * white[:, 1]
        turn = torch.remainder(pose[:, 3] - self.previous[:, 3] + math.pi, 2 * math.pi) - math.pi
        turn = turn + heading * distance + gyro * self.dt + distance.sqrt() * white[:, 2]
        turn = turn + math.sqrt(self.dt) * white[:, 3]
        # Integrated in the estimated frame.
        cos, sin = self.estimate[:, 3].cos(), self.estimate[:, 3].sin()
        climb = pose[:, 2] - self.previous[:, 2] + vertical * distance
        step = torch.stack((cos * forward - sin * left, sin * forward + cos * left, climb, turn), dim=1)
        self.estimate = torch.where(reset.unsqueeze(1), pose, self.estimate + step)
        self.previous = pose.clone()
        return self.estimate.clone()
