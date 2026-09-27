"""Geometry and masking checks for the BEV model that do not need Isaac Sim."""

import math
import os

import pytest
import torch

from rover_envs.envs.navigation.learning.bev.losses import bev_loss, bev_metrics, bev_statistics
from rover_envs.envs.navigation.learning.bev.models import BevUNet, RoverBevModel, lift_to_bev

X_RANGE, Y_RANGE, RESOLUTION = (-1.0, 1.0), (-1.0, 1.0), 0.1
HEIGHT, WIDTH, FOCAL = 48, 64, 100.0


def downward_camera(height: float = 1.0) -> torch.Tensor:
    """Camera `height` m above the BEV origin looking straight down: image right is -y and image down is -x."""
    intrinsics = torch.tensor([[FOCAL, 0.0, WIDTH / 2], [0.0, FOCAL, HEIGHT / 2], [0.0, 0.0, 1.0]])
    rotation = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    return torch.cat((intrinsics.flatten(), rotation.flatten(), torch.tensor([0.0, 0.0, height]))).unsqueeze(0)


def flat_ground_depth(camera_height: float = 1.0, ground: float = -0.3) -> torch.Tensor:
    return torch.full((1, HEIGHT, WIDTH, 1), camera_height - ground)


def test_lift_places_flat_ground_in_the_camera_footprint():
    depth = flat_ground_depth()
    lifted = lift_to_bev(depth, downward_camera(), torch.zeros(1, 4, 3, 4), X_RANGE, Y_RANGE, RESOLUTION)
    assert lifted.shape == (1, 3 + 4, 20, 20)
    observed = lifted[0, 2] > 0
    # The footprint at 1.3 m spans |x| <= 24 / 100 * 1.3 = 0.312 m and |y| <= 32 / 100 * 1.3 = 0.416 m.
    x = torch.arange(20) * RESOLUTION + X_RANGE[0] + RESOLUTION / 2
    y = torch.arange(20) * RESOLUTION + Y_RANGE[0] + RESOLUTION / 2
    inside = (x.abs() < 0.25).unsqueeze(1) & (y.abs() < 0.35).unsqueeze(0)
    outside = (x.abs() > 0.4).unsqueeze(1) | (y.abs() > 0.5).unsqueeze(0)
    assert observed[inside].all() and not observed[outside].any()
    torch.testing.assert_close(lifted[0, 0][observed], torch.full((int(observed.sum()),), -0.3))
    assert (lifted[0, 1][observed] == 0).all()
    assert int(lifted[0, 2].expm1().round().sum()) == HEIGHT * WIDTH


def test_lift_maps_a_pixel_to_its_cell():
    # Only the pixel at row 4, column 50 has depth; its centre is 19.5 px above and 18.5 px right of the image centre.
    depth = torch.full((1, HEIGHT, WIDTH, 1), torch.inf)
    depth[0, 4, 50, 0] = 2.0
    lifted = lift_to_bev(depth, downward_camera(), torch.zeros(1, 1, 3, 4), X_RANGE, Y_RANGE, RESOLUTION)
    x, y = 19.5 / FOCAL * 2.0, -18.5 / FOCAL * 2.0
    cell = (math.floor((x - X_RANGE[0]) / RESOLUTION), math.floor((y - Y_RANGE[0]) / RESOLUTION))
    assert (lifted[0, 2] > 0).nonzero().tolist() == [list(cell)]
    torch.testing.assert_close(lifted[0, 0][cell], torch.tensor(1.0 - 2.0))


def test_lift_ignores_invalid_depth_and_averages_features():
    depth = flat_ground_depth()
    invalid = torch.tensor([0.0, torch.nan, torch.inf, -1.0]).repeat(HEIGHT * WIDTH // 8)
    depth[0, :, : WIDTH // 2] = invalid.view(HEIGHT, WIDTH // 2, 1)
    features = torch.tensor([1.0, -2.0]).view(1, 2, 1, 1).expand(1, 2, 3, 4)
    lifted = lift_to_bev(depth, downward_camera(), features, X_RANGE, Y_RANGE, RESOLUTION)
    observed = lifted[0, 2] > 0
    # The left image half sees +y, which must stay empty.
    y = torch.arange(20) * RESOLUTION + Y_RANGE[0] + RESOLUTION / 2
    assert not observed[:, y > 0.05].any() and observed[:, y < -0.05].any()
    torch.testing.assert_close(lifted[0, 3:][:, observed], torch.tensor([[1.0], [-2.0]]).expand(2, int(observed.sum())))
    assert (lifted[0, 3:][:, ~observed] == 0).all()


def test_lift_gradients_reach_the_features():
    features = torch.randn(1, 2, 3, 4, requires_grad=True)
    lift_to_bev(flat_ground_depth(), downward_camera(), features, X_RANGE, Y_RANGE, RESOLUTION)[:, 3:].sum().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_lift_without_features_returns_the_same_geometry():
    depth, camera, features = flat_ground_depth(), downward_camera(), torch.randn(1, 2, 3, 4)
    geometry = lift_to_bev(depth, camera, None, X_RANGE, Y_RANGE, RESOLUTION)
    assert geometry.shape == (1, 3, 20, 20)
    torch.testing.assert_close(geometry, lift_to_bev(depth, camera, features, X_RANGE, Y_RANGE, RESOLUTION)[:, :3])


def small_model(**kwargs) -> RoverBevModel:
    return RoverBevModel(2, X_RANGE, Y_RANGE, RESOLUTION, feature_dim=4, unet_channels=(8, 16), **kwargs)


def test_geometry_only_model_trains_without_dinov3():
    model = small_model(use_dino=False)
    assert model.dino is None and not any(key.startswith(("dino", "adapter")) for key in model.state_dict())
    assert model.unet.down[0][0].in_channels == 3
    rgb = torch.zeros(1, HEIGHT, WIDTH, 3, dtype=torch.uint8)
    elevation, semantics, lifted = model(rgb, flat_ground_depth(), downward_camera())
    assert elevation.shape == (1, 20, 20) and semantics.shape == (1, 2, 20, 20) and lifted.shape == (1, 3, 20, 20)
    (elevation.sum() + semantics.sum()).backward()
    assert model.unet.down[0][0].weight.grad.abs().sum() > 0


@pytest.mark.skipif(not os.path.isdir(os.path.expanduser("~/models/dinov3-vits16")), reason="needs DINOv3 weights")
def test_dino_only_model_feeds_only_the_features():
    model = small_model(use_geometry=False, image_size=(32, 64))
    assert model.unet.down[0][0].in_channels == 4
    rgb = torch.randint(0, 255, (1, HEIGHT, WIDTH, 3), dtype=torch.uint8)
    elevation, _, lifted = model(rgb, flat_ground_depth(), downward_camera())
    # The lifted geometry is still returned for the metrics.
    assert elevation.shape == (1, 20, 20) and lifted.shape == (1, 3 + 4, 20, 20)


def test_unet_keeps_the_bev_shape():
    assert BevUNet(35, (8, 16, 16, 16))(torch.randn(2, 35, 120, 120)).shape == (2, 8, 120, 120)


def test_loss_and_metrics_only_use_visible_cells():
    target = torch.zeros(1, 3, 2, 2)
    target[0, 0] = torch.tensor([[0.1, torch.nan], [0.3, torch.nan]])
    target[0, 1] = torch.tensor([[0.0, -1.0], [1.0, -1.0]])
    target[0, 2] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    elevation = torch.tensor([[[0.2, 9.0], [0.3, 9.0]]], requires_grad=True)
    semantics = torch.tensor([[[5.0, 0.0], [5.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]).unsqueeze(0)
    lifted = torch.zeros(1, 3, 2, 2)
    lifted[0, 2, 0, 0] = 1.0
    total, elevation_loss, _ = bev_loss(elevation, semantics, target, semantic_weight=1.0)
    total.backward()
    assert torch.isfinite(total) and elevation.grad[0, :, 1].eq(0).all()
    # Huber errors 0.1 (linear branch: 0.1 - beta / 2) and 0, averaged over the two visible cells.
    torch.testing.assert_close(elevation_loss, torch.tensor((0.1 - 0.05 / 2) / 2))
    metrics = bev_metrics(bev_statistics(elevation, semantics, lifted, target), ("ground", "rock"))
    assert math.isclose(metrics["elevation_mae"], 0.05, abs_tol=1e-6)
    assert math.isclose(metrics["lift_elevation_mae"], 0.1, abs_tol=1e-6) and metrics["lift_coverage"] == 0.5
    assert metrics["semantic_accuracy"] == 0.5 and metrics["iou_ground"] == 0.5 and metrics["iou_rock"] == 0.0
