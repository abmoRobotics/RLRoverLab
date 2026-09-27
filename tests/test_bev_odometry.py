"""Checks of the simulated odometry drift of unknown, varying size; no Isaac Sim needed."""

import math

import pytest
import torch

from rover_envs.envs.navigation.learning.bev.odometry import OdometryDrift

DT = 0.2


class Phase2Drift:
    """The Phase-2 drift model, verbatim from the evaluation script it was first used in (fixed severity only)."""

    def __init__(self, num_envs: int, severity: float, dt: float, seed: int, device: str):
        self.generator = torch.Generator(device).manual_seed(seed)
        self.bias_std = severity * torch.tensor([0.012, 0.008, 0.008, 0.0015, 0.008], device=device)
        self.white_std = severity * torch.tensor([0.004, 0.002, 0.0007], device=device)
        self.correlation, self.dt = math.exp(-dt / 4.0), dt
        self.bias = self.bias_std * self._normal(num_envs, 5)
        self.estimate = self.previous = None

    def _normal(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self.generator, device=self.generator.device)

    def __call__(self, pose: torch.Tensor, reset: torch.Tensor) -> torch.Tensor:
        if self.estimate is None:
            self.estimate, self.previous = pose.clone(), pose.clone()
            return self.estimate.clone()
        innovation = math.sqrt(1.0 - self.correlation**2) * self.bias_std * self._normal(*self.bias.shape)
        self.bias = self.correlation * self.bias + innovation
        forward_scale, lateral, heading, gyro, vertical = self.bias.unbind(1)
        white = self._normal(len(pose), 4)
        delta = pose[:, :2] - self.previous[:, :2]
        distance = delta.norm(dim=1)
        cos, sin = self.previous[:, 3].cos(), self.previous[:, 3].sin()
        forward = (cos * delta[:, 0] + sin * delta[:, 1]) * (1 + forward_scale)
        left = -sin * delta[:, 0] + cos * delta[:, 1] + lateral * distance
        forward = forward + self.white_std[0] * distance.sqrt() * white[:, 0]
        left = left + self.white_std[0] * distance.sqrt() * white[:, 1]
        turn = torch.remainder(pose[:, 3] - self.previous[:, 3] + math.pi, 2 * math.pi) - math.pi
        turn = turn + heading * distance + gyro * self.dt + self.white_std[1] * distance.sqrt() * white[:, 2]
        turn = turn + self.white_std[2] * math.sqrt(self.dt) * white[:, 3]
        cos, sin = self.estimate[:, 3].cos(), self.estimate[:, 3].sin()
        climb = pose[:, 2] - self.previous[:, 2] + vertical * distance
        step = torch.stack((cos * forward - sin * left, sin * forward + cos * left, climb, turn), dim=1)
        self.estimate = torch.where(reset.unsqueeze(1), pose, self.estimate + step)
        self.previous = pose.clone()
        return self.estimate.clone()


def drive(num_envs: int = 8, steps: int = 150, speed: float = 0.5, turn_rate: float = 0.1) -> list[torch.Tensor]:
    """Exact poses of rovers driving arcs over a gentle slope, starting at different places."""
    start = torch.stack((torch.arange(num_envs) * 7.0, torch.arange(num_envs) * -3.0), dim=1)
    poses, x, y, yaw = [], start[:, 0].clone(), start[:, 1].clone(), torch.linspace(-1, 1, num_envs)
    for _ in range(steps):
        poses.append(torch.stack((x, y, 0.05 * x, yaw), dim=1).clone())
        x, y, yaw = x + speed * DT * yaw.cos(), y + speed * DT * yaw.sin(), yaw + turn_rate * DT
    return poses


def run(drift, poses: list[torch.Tensor], resets: dict[int, torch.Tensor] | None = None) -> list[torch.Tensor]:
    no_reset = torch.zeros(len(poses[0]), dtype=torch.bool)
    return [drift(pose, (resets or {}).get(k, no_reset)) for k, pose in enumerate(poses)]


def test_a_fixed_severity_reproduces_the_phase2_drift():
    poses = drive()
    for severity in (1.0, 4.0):
        new = run(OdometryDrift(8, severity, DT, seed=42, device="cpu"), poses)
        old = run(Phase2Drift(8, severity, DT, seed=42, device="cpu"), poses)
        torch.testing.assert_close(torch.stack(new), torch.stack(old), rtol=0, atol=1e-5)


def test_zero_severity_gives_the_exact_poses():
    poses = drive()
    estimates = run(OdometryDrift(8, 0.0, DT, seed=0, device="cpu"), poses)
    torch.testing.assert_close(torch.stack(estimates), torch.stack(poses), rtol=0, atol=1e-4)


def test_the_drift_scales_with_the_severity():
    poses = drive()
    errors = []
    for severity in (1.0, 4.0):
        estimate = run(OdometryDrift(8, severity, DT, seed=3, device="cpu"), poses)[-1]
        errors.append((estimate[:, :2] - poses[-1][:, :2]).norm(dim=1).mean())
    # The same random draws scaled by 4; heading errors rotate the path, so it is only nearly linear.
    assert 3.5 < errors[1] / errors[0] < 4.5


def test_each_run_draws_its_severity_log_uniformly():
    # The initial draw is stratified: 8 environments put exactly 2 in each octave of 0.25x to 4x.
    for num_envs, tolerance in ((8, 0.0), (20000, 0.0)):
        drift = OdometryDrift(num_envs, (0.25, 4.0), DT, seed=1, device="cpu")
        assert drift.severity.min() >= 0.25 and drift.severity.max() <= 4.0
        octaves = torch.histc(drift.severity.log2(), bins=4, min=-2.0, max=2.0) / num_envs
        torch.testing.assert_close(octaves, torch.full((4,), 0.25), rtol=0, atol=tolerance)
    # Redraws after resets are independent and log-uniform too.
    poses = drive(num_envs=20000, steps=2)
    drift = OdometryDrift(20000, (0.25, 4.0), DT, seed=1, device="cpu")
    drift(poses[0], torch.zeros(20000, dtype=torch.bool))
    drift(poses[1], torch.ones(20000, dtype=torch.bool))
    octaves = torch.histc(drift.severity.log2(), bins=4, min=-2.0, max=2.0) / 20000
    torch.testing.assert_close(octaves, torch.full((4,), 0.25), rtol=0, atol=0.02)


def test_a_reset_restarts_from_the_exact_pose_with_a_new_severity():
    poses = drive(num_envs=4, steps=40)
    drift = OdometryDrift(4, (0.25, 4.0), DT, seed=5, device="cpu")
    reset = torch.tensor([False, True, False, False])
    estimates = run(drift, poses[:30])
    before = drift.severity.clone()
    estimate = drift(poses[30], reset)
    torch.testing.assert_close(estimate[1], poses[30][1])
    assert (estimate[[0, 2, 3]] != poses[30][[0, 2, 3]]).any(dim=1).all()
    assert drift.severity[1] != before[1] and torch.equal(drift.severity[[0, 2, 3]], before[[0, 2, 3]])
    assert (estimates[-1] != poses[29]).any()


def test_the_location_field_is_smooth_and_has_the_requested_spread():
    drift = OdometryDrift(1, 1.0, DT, seed=7, device="cpu", field_std=0.5, field_length=5.0, field_features=256)
    # Many independent places, far apart compared with the correlation length.
    places = torch.rand(20000, 2, generator=torch.Generator().manual_seed(0)) * 2000.0
    log_factor = drift.location_factor(places).log()
    assert abs(log_factor.mean()) < 0.03 and abs(log_factor.std() - 0.5) < 0.03
    # Nearby places have nearly the same factor, places many correlation lengths apart are unrelated up to the random
    # features' kernel error, about 1 / sqrt(2 * 256).
    near = drift.location_factor(places + torch.tensor([0.1, 0.0])).log()
    far = drift.location_factor(places + torch.tensor([25.0, 0.0])).log()
    assert (near - log_factor).abs().max() < 0.1
    assert abs(torch.corrcoef(torch.stack((log_factor, far)))[0, 1]) < 0.1


def test_the_level_follows_the_ground_the_rover_is_on():
    poses = drive(num_envs=4)
    drift = OdometryDrift(4, (0.5, 2.0), DT, seed=9, device="cpu", field_std=0.5, field_length=5.0)
    for pose in poses:
        drift(pose, torch.zeros(4, dtype=torch.bool))
        torch.testing.assert_close(drift.level, drift.severity * drift.location_factor(pose[:, :2]))


def test_the_environment_random_stream_is_left_alone():
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    run(OdometryDrift(4, (0.25, 4.0), DT, seed=0, device="cpu", field_std=0.5), drive(num_envs=4, steps=10))
    torch.testing.assert_close(torch.rand(3), expected)


def test_invalid_severities_are_rejected():
    for severity in (-1.0, (2.0, 1.0), (0.0, 1.0)):
        with pytest.raises(ValueError):
            OdometryDrift(1, severity, DT, seed=0, device="cpu")
