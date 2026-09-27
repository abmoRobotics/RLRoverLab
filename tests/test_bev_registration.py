"""Checks of the keyframe registration that measures odometry drift from re-observed terrain; no Isaac Sim needed."""

import math

import torch

from rover_envs.envs.navigation.learning.bev.registration import KeyframeRegistration, relative_correction, rotate

# The BEV task's grid: 120 x 120 cells at 5 cm, x in [-2, 4) m forward, y in [-3, 3) m left.
X_RANGE, Y_RANGE, RESOLUTION = (-2.0, 4.0), (-3.0, 3.0), 0.05
CENTRES = torch.stack(torch.meshgrid(
    X_RANGE[0] + (torch.arange(120) + 0.5) * RESOLUTION, Y_RANGE[0] + (torch.arange(120) + 0.5) * RESOLUTION,
    indexing="ij",
), dim=-1)
# Gaussian rocks: world x, y, height and radius in metres.
ROCKS = torch.tensor([[1.8, 0.3, 0.25, 0.20], [2.6, -0.6, 0.30, 0.25], [3.3, 0.8, 0.18, 0.15], [2.2, 1.4, 0.22, 0.3],
                      [3.9, -1.2, 0.28, 0.2], [4.5, 0.1, 0.25, 0.25], [1.4, -1.0, 0.2, 0.15], [4.8, 1.5, 0.3, 0.2]])


def rocky(world: torch.Tensor) -> torch.Tensor:
    x, y = world[..., 0], world[..., 1]
    height = 0.06 * torch.sin(0.9 * x) * torch.cos(0.7 * y)
    for rx, ry, h, r in ROCKS:
        height = height + h * torch.exp(-((x - rx) ** 2 + (y - ry) ** 2) / (2 * r**2))
    return height


def flat(world: torch.Tensor) -> torch.Tensor:
    return torch.full(world.shape[:-1], 0.3)


def frame(true_pose: torch.Tensor, terrain=rocky) -> tuple[torch.Tensor, torch.Tensor]:
    """The depth lift of a camera wedge ahead of each true pose [B, 4]: heights from the BEV origin, point counts."""
    ego = CENTRES.expand(len(true_pose), -1, -1, -1)
    world = true_pose[:, None, None, :2] + rotate(ego, true_pose[:, 3, None, None])
    height = terrain(world) - true_pose[:, 2, None, None]
    x, y = CENTRES[..., 0], CENTRES[..., 1]
    wedge = (x >= 0.8) & (y.abs() <= 0.9 * x)
    return height, torch.where(wedge, 5.0, 0.0).expand_as(height)


def drive(registration: KeyframeRegistration, true_poses, poses, terrain=rocky, resets=None):
    """Feed the frames of true poses [T, B, 4] with odometry poses [T, B, 4]; return the registration outputs."""
    outputs = []
    for k, (true_pose, pose) in enumerate(zip(true_poses, poses)):
        height, count = frame(true_pose, terrain)
        reset = torch.zeros(len(pose), dtype=torch.bool) if resets is None else resets[k]
        outputs.append(registration.step(pose, height, count, reset))
    return outputs


def straight(steps: int = 7, speed: float = 0.21, num_envs: int = 1) -> torch.Tensor:
    """True poses driving along x with a gentle climb: [steps, B, 4]."""
    k = torch.arange(steps, dtype=torch.float32).view(-1, 1)
    pose = torch.cat((0.1 + speed * k, torch.full_like(k, 0.05), 0.02 * k, torch.full_like(k, 0.02)), dim=1)
    return pose.unsqueeze(1).repeat(1, num_envs, 1)


def registration(**kwargs) -> KeyframeRegistration:
    """A registration every metre, with a search wide enough for the drifts below."""
    kwargs = {"keyframe_distance": 1.0, "stages": (
        (0.20, 0.04, math.radians(6.0), math.radians(1.0)), (0.04, 0.01, math.radians(1.0), math.radians(0.25)),
    ), **kwargs}
    return KeyframeRegistration(kwargs.pop("num_envs", 1), X_RANGE, Y_RANGE, RESOLUTION, **kwargs)


def test_exact_odometry_needs_no_correction():
    outputs = drive(registration(), straight(), straight())
    registered = [k for k, out in enumerate(outputs) if out["registered"].any()]
    # A keyframe at the first step, and a registration once the odometry has travelled 1 m.
    assert registered == [5]
    out = outputs[5]
    assert out["interval"].item() >= 1.0 and out["overlap"].item() > 0.5
    correction = out["correction"][0]
    assert correction[:2].norm() < 0.01 and correction[2].abs() < math.radians(0.2) and correction[3].abs() < 0.005


def test_a_known_drift_is_recovered():
    true_poses = straight()
    # The odometry is exact at the keyframe and then drifts: 1.5 deg, (4, -3) cm and 2 cm up by the registration.
    fraction = (torch.arange(len(true_poses)) / 5).view(-1, 1, 1)
    poses = true_poses + fraction * torch.tensor([0.04, -0.03, 0.02, math.radians(1.5)])
    out = drive(registration(), true_poses, poses)[5]
    expected = relative_correction(true_poses[0], true_poses[5], poses[0], poses[5])[0]
    measured = out["correction"][0]
    assert (measured[:2] - expected[:2]).norm() < 0.01, (measured, expected)
    assert (measured[2] - expected[2]).abs() < math.radians(0.25) and (measured[3] - expected[3]).abs() < 0.005
    assert (out["curvature"][0] > 0).all()


def test_flat_ground_leaves_the_horizontal_correction_unobservable():
    true_poses = straight()
    poses = true_poses + torch.linspace(0.0, 0.03, len(true_poses)).view(-1, 1, 1) * torch.tensor([1.0, 0, 1.0, 0])
    out = drive(registration(), true_poses, poses, terrain=flat)[5]
    expected = relative_correction(true_poses[0], true_poses[5], poses[0], poses[5])[0]
    # The height offset is still measured, but the cost does not change along x, y or yaw.
    assert (out["correction"][0, 3] - expected[3]).abs() < 0.001
    assert (out["curvature"][0] < 1e-3).all()


def test_the_correction_matches_the_relative_pose_error():
    # A constant world offset of both odometry poses is no drift between them.
    true_key, true_now = torch.tensor([1.0, 2.0, 0.1, 0.3]), torch.tensor([1.9, 2.4, 0.15, 0.5])
    offset = torch.tensor([0.5, -0.2, 0.3, 0.0])
    torch.testing.assert_close(relative_correction(true_key, true_now, true_key + offset, true_now + offset),
                               torch.zeros(4), rtol=0, atol=1e-6)
    # A yaw error that appears only at the current pose is a pure rotation of the current frame.
    now = true_now.clone()
    now[3] -= 0.1
    torch.testing.assert_close(relative_correction(true_key, true_now, true_key, now),
                               torch.tensor([0.0, 0.0, 0.1, 0.0]), rtol=0, atol=1e-6)


def test_environments_are_independent_and_resets_start_over():
    true_poses = straight(steps=10, num_envs=2)
    resets = [torch.zeros(2, dtype=torch.bool) for _ in true_poses]
    resets[3] = torch.tensor([False, True])
    outputs = drive(registration(num_envs=2), true_poses, true_poses, resets=resets)
    registered = torch.stack([out["registered"] for out in outputs])
    # Environment 0 registers after 1 m; environment 1 starts a new keyframe the step after its reset.
    assert registered[:, 0].nonzero().squeeze(1).tolist() == [5]
    assert registered[:, 1].nonzero().squeeze(1).tolist() == [9]
