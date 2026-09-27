from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils import math as math_utils

# from isaaclab.command_generators import UniformPoseCommandGenerator

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import Camera, MultiMeshRayCaster


def angle_to_target_observation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Calculate the angle to the target."""

    # Get vector(x,y) from rover to target, in base frame of the rover.
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]

    # Calculate the angle between the rover's heading [1, 0] and the vector to the target.
    angle = torch.atan2(target_vector_b[:, 1], target_vector_b[:, 0])

    return angle.unsqueeze(-1)


def distance_to_target_euclidean(env: ManagerBasedRLEnv, command_name: str):
    """Calculate the euclidean distance to the target."""
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]
    distance: torch.Tensor = torch.norm(target_position, p=2, dim=-1)
    return distance.unsqueeze(-1)


def height_scan_rover(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Calculate the height scan of the rover.

    This function uses a ray caster to generate a height scan of the rover's surroundings.
    The height scan is normalized by the maximum range of the ray caster.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - 0.26878
    # Note: 0.26878 is the distance between the sensor and the rover's base
    return sensor.data.pos_w.torch[:, 2].unsqueeze(1) - sensor.data.ray_hits_w.torch[..., 2] - 0.26878


def angle_diff(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Calculate the angle difference between the rover's heading and the target."""
    # Get the angle to the target
    heading_angle_diff = env.command_manager.get_command(command_name)[:, 3]

    return heading_angle_diff.unsqueeze(-1)


def camera_intrinsics(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Row-major camera intrinsics [N, 9] in pixels, with pixel (u, v) covering [u, u + 1) x [v, v + 1)."""
    camera: Camera = env.scene.sensors[sensor_cfg.name]
    return camera.data.intrinsic_matrices.torch.flatten(1)


def stereo_depth(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    baseline: float,
    disparity_std: float,
    correlation_px: int,
    dropout: float,
) -> torch.Tensor:
    """Rendered depth [N, H, W, 1] in metres with the errors of a stereo camera.

    Gaussian disparity noise with a standard deviation of ``disparity_std`` pixels, correlated over boxes of
    ``correlation_px`` x ``correlation_px`` pixels (odd), gives a depth error of about z^2 * disparity_std /
    (fx * baseline). Blobs of the same size covering a ``dropout`` fraction of the image, and pixels whose disparity
    becomes non-positive, are holes (NaN).
    """
    camera: Camera = env.scene.sensors[sensor_cfg.name]
    depth = camera.data.output["depth"].torch
    focal_baseline = camera.data.intrinsic_matrices.torch[:, 0, 0].view(-1, 1, 1, 1) * baseline

    def correlated_noise() -> torch.Tensor:
        noise = torch.randn(depth.shape[0], 1, *depth.shape[1:3], device=depth.device)
        # A box mean of k x k independent unit Gaussians scaled by k has unit variance again.
        noise = F.avg_pool2d(noise, correlation_px, stride=1, padding=correlation_px // 2) * correlation_px
        return noise.permute(0, 2, 3, 1)

    disparity = focal_baseline / depth + disparity_std * correlated_noise()
    holes = correlated_noise() > torch.special.ndtri(torch.tensor(1.0 - dropout))
    return torch.where((disparity > 0) & ~holes, focal_baseline / disparity, torch.nan)


def bev_frame_pose(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """World pose [N, 4] of the ego BEV frame of the ``sensor_cfg`` sensor: its origin in metres and its yaw in radians.

    This is the pose of the BEV observations taken in the same step; their elevation is measured from this z.
    """
    frame: MultiMeshRayCaster = env.scene.sensors[sensor_cfg.name]
    x, y, z, w = frame.data.quat_w.torch.unbind(1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.cat((frame.data.pos_w.torch, yaw.unsqueeze(1)), dim=1)


def camera_pose_bev(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """Camera pose [N, 12] in the ego BEV frame: the row-major camera-to-BEV rotation, then the position in metres.

    The camera axes follow the ROS optical convention (+x right, +y down, +z forward). The BEV frame is the yaw-aligned
    frame of the ``frame_cfg`` sensor: its origin, +x along its heading, +y left and +z up. The camera must be mounted
    on that sensor's body: its pose is the body's roll and pitch applied to the camera's configured offset, because
    the camera data keeps reporting the initial pose of a camera mounted under a rigid body.
    """
    camera: Camera = env.scene.sensors[sensor_cfg.name]
    frame: MultiMeshRayCaster = env.scene.sensors[frame_cfg.name]
    offset = camera.cfg.offset
    mount_quat = math_utils.convert_camera_frame_orientation_convention(
        torch.tensor([offset.rot], device=env.device), origin=offset.convention, target="ros"
    )
    body_quat = frame.data.quat_w.torch
    tilt = math_utils.quat_mul(math_utils.quat_inv(math_utils.yaw_quat(body_quat)), body_quat)
    rotation = math_utils.matrix_from_quat(math_utils.quat_mul(tilt, mount_quat.expand_as(tilt)))
    position = math_utils.quat_apply(tilt, torch.tensor(offset.pos, device=env.device).expand(len(tilt), 3))
    return torch.cat((rotation.flatten(1), position), dim=-1)


def bev_ground_truth(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, camera_cfg: SceneEntityCfg, occlusion_tolerance: float = 0.03
) -> torch.Tensor:
    """BEV targets [N, 3, X, Y] from a "yx"-ordered grid of downward rays in the yaw-aligned sensor frame.

    Channels:

    0. Elevation of the first hit relative to the sensor origin in metres; NaN where the ray missed.
    1. Semantic class: the index of the hit target in the sensor's ``mesh_prim_paths``; -1 where the ray missed.
    2. Camera visibility: 1 where the hit lies inside the camera image and is not occluded, i.e. at most
       ``occlusion_tolerance`` (relative) behind the rendered depth; 0 elsewhere. Losses should only use these cells.
    """
    sensor: MultiMeshRayCaster = env.scene.sensors[sensor_cfg.name]
    camera: Camera = env.scene.sensors[camera_cfg.name]
    pattern = sensor.cfg.pattern_cfg
    shape = tuple(round(size / pattern.resolution) + 1 for size in pattern.size)
    hits = sensor.data.ray_hits_w.torch
    if pattern.ordering != "yx" or hits.shape[1] != shape[0] * shape[1]:
        raise ValueError(f"Expected a 'yx'-ordered {shape} ray grid, got {hits.shape[1]} '{pattern.ordering}' rays.")
    hit = torch.isfinite(hits).all(dim=-1)
    # Hits relative to the sensor origin in its yaw-aligned frame, the BEV frame.
    yaw = math_utils.matrix_from_quat(math_utils.yaw_quat(sensor.data.quat_w.torch))
    hits = torch.where(hit.unsqueeze(-1), hits, 0.0) - sensor.data.pos_w.torch.unsqueeze(1)
    hits = torch.einsum("nji,nrj->nri", yaw, hits)
    elevation = torch.where(hit, hits[..., 2], torch.nan)
    # Mesh ids are only written on a hit, so misses keep stale values.
    semantics = torch.where(hit, sensor.data.ray_mesh_ids.torch.view_as(hit).float(), -1.0)

    # Project the BEV-frame hits into the camera (ROS optical frame) and compare them with its depth image.
    pose = camera_pose_bev(env, camera_cfg, sensor_cfg)
    points = torch.einsum("nji,nrj->nri", pose[:, :9].view(-1, 3, 3), hits - pose[:, None, 9:])
    intrinsics = camera.data.intrinsic_matrices.torch
    depth = camera.data.output["depth"].torch[..., 0]
    height, width = depth.shape[1:]
    z = points[..., 2].clamp(min=1e-6)
    u = intrinsics[:, 0, 0, None] * points[..., 0] / z + intrinsics[:, 0, 2, None]
    v = intrinsics[:, 1, 1, None] * points[..., 1] / z + intrinsics[:, 1, 2, None]
    in_image = hit & (points[..., 2] > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    pixel = torch.where(in_image, v.long() * width + u.long(), 0)
    visible = in_image & (z <= depth.flatten(1).gather(1, pixel) * (1.0 + occlusion_tolerance))
    return torch.stack((elevation, semantics, visible.float()), dim=1).view(-1, 3, *shape)
