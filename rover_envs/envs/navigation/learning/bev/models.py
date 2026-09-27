"""Instantaneous ego BEV from one RGB-D frame: frozen DINOv3 features, lifted by the depth, refined by a U-Net.

BEV maps follow the BEV task's ``BevCfg``: the yaw-aligned rover Body frame (+x forward, +y left, +z up, metres) with
the layout [x, y], where cell (i, j) covers [x_range[0] + i * resolution, x_range[0] + (i + 1) * resolution) along x
and likewise along y.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rover_envs.envs.navigation.learning.encoders.dinov3 import DinoV3FeatureMap


def lift_to_bev(
    depth: torch.Tensor,
    camera: torch.Tensor,
    features: torch.Tensor | None,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
) -> torch.Tensor:
    """Splat the depth points, and the image features at their pixels, into the BEV grid.

    Args:
        depth: Distance to the image plane [B, H, W, 1] in metres. Non-finite and non-positive pixels are ignored.
        camera: [B, 21]: the row-major intrinsics, with pixel (u, v) covering [u, u + 1) x [v, v + 1), followed by the
            row-major camera-to-BEV rotation and the camera position in the BEV frame. The camera axes follow the ROS
            optical convention: +x right, +y down, +z forward.
        features: Image features [B, C, h, w] covering the whole image, or None to lift the geometry only (C = 0).

    Returns:
        [B, 3 + C, X, Y]: per cell the mean point elevation, the elevation range (max - min) and log(1 + point count),
        followed by the mean feature of its points. Cells without points are zero.
    """
    batch, height, width, _ = depth.shape
    device = depth.device
    nx, ny = round((x_range[1] - x_range[0]) / resolution), round((y_range[1] - y_range[0]) / resolution)
    intrinsics, rotation, position = camera[:, :9].view(-1, 3, 3), camera[:, 9:18].view(-1, 3, 3), camera[:, 18:]

    # Unproject the pixel centres into the BEV frame.
    d = depth[..., 0]
    valid = torch.isfinite(d) & (d > 0)
    d = torch.where(valid, d, 0.0)
    v, u = torch.meshgrid(
        torch.arange(height, device=device) + 0.5, torch.arange(width, device=device) + 0.5, indexing="ij"
    )
    x = (u - intrinsics[:, 0, 2, None, None]) / intrinsics[:, 0, 0, None, None] * d
    y = (v - intrinsics[:, 1, 2, None, None]) / intrinsics[:, 1, 1, None, None] * d
    points = torch.einsum("bij,bhwj->bhwi", rotation, torch.stack((x, y, d), dim=-1)) + position[:, None, None]

    # Flat [B, X, Y] cell index of every valid point inside the grid.
    ix = torch.floor((points[..., 0] - x_range[0]) / resolution).long()
    iy = torch.floor((points[..., 1] - y_range[0]) / resolution).long()
    valid &= (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    cell = ((torch.arange(batch, device=device).view(-1, 1, 1) * nx + ix) * ny + iy)[valid]
    z = points[..., 2][valid]

    cells = batch * nx * ny
    count = torch.zeros(cells, device=device).index_add_(0, cell, torch.ones_like(z))
    z_mean = torch.zeros(cells, device=device).index_add_(0, cell, z) / count.clamp(min=1)
    z_max = torch.full((cells,), -torch.inf, device=device).scatter_reduce_(0, cell, z, "amax")
    z_min = torch.full((cells,), torch.inf, device=device).scatter_reduce_(0, cell, z, "amin")
    z_range = torch.where(count > 0, z_max - z_min, 0.0)
    lifted = torch.stack((z_mean, z_range, count.log1p()), dim=1)

    if features is not None:
        pixel_features = F.interpolate(features, size=(height, width), mode="bilinear", align_corners=False)
        pixel_features = pixel_features.permute(0, 2, 3, 1)[valid]
        cell_features = torch.zeros(cells, features.shape[1], device=device).index_add(0, cell, pixel_features)
        lifted = torch.cat((lifted, cell_features / count.clamp(min=1).unsqueeze(1)), dim=1)
    return lifted.view(batch, nx, ny, -1).permute(0, 3, 1, 2)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.ReLU(inplace=True),
    )


class BevUNet(nn.Module):
    """U-Net with one level per entry of ``channels``; every level after the first halves the resolution."""

    def __init__(self, in_channels: int, channels: tuple[int, ...]):
        super().__init__()
        self.down = nn.ModuleList(_conv_block(c_in, c_out) for c_in, c_out in zip((in_channels, *channels), channels))
        self.up = nn.ModuleList(
            _conv_block(channels[level + 1] + channels[level], channels[level])
            for level in reversed(range(len(channels) - 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for level, block in enumerate(self.down):
            x = block(F.max_pool2d(x, 2) if level else x)
            skips.append(x)
        skips.pop()
        for block in self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skips.pop()), dim=1))
        return x


class RoverBevModel(nn.Module):
    """RGB-D frame -> frozen DINOv3 -> 1x1 adapter -> depth lift into the BEV -> U-Net -> elevation and class heads.

    Only the adapter, the U-Net and the heads are trained. For ablations, ``use_geometry`` and ``use_dino`` select the
    lifted channels the U-Net sees: the geometry channels, the DINOv3 features or both. The depth always places the
    features in the BEV, and without DINOv3 the backbone is not loaded.
    """

    def __init__(
        self,
        num_classes: int,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        resolution: float,
        dinov3_path: str = "~/models/dinov3-vits16",
        image_size: tuple[int, int] = (368, 672),
        feature_dim: int = 32,
        unet_channels: tuple[int, ...] = (64, 128, 256, 256),
        use_geometry: bool = True,
        use_dino: bool = True,
    ):
        super().__init__()
        self.x_range, self.y_range, self.resolution = tuple(x_range), tuple(y_range), resolution
        scale = 2 ** (len(unet_channels) - 1)
        if any(round((high - low) / resolution) % scale for low, high in (self.x_range, self.y_range)):
            raise ValueError(f"The BEV cells per side must be divisible by {scale} for {len(unet_channels)} levels.")
        if not (use_geometry or use_dino):
            raise ValueError("The model needs the geometry channels, the DINOv3 features or both.")
        self.use_geometry = use_geometry
        self.dino = self.adapter = None
        if use_dino:
            self.dino = DinoV3FeatureMap(dinov3_path, image_size)
            self.adapter = nn.Conv2d(self.dino.out_dim, feature_dim, kernel_size=1)
        self.unet = BevUNet(3 * use_geometry + feature_dim * use_dino, tuple(unet_channels))
        self.elevation_head = nn.Conv2d(unet_channels[0], 1, kernel_size=1)
        self.semantic_head = nn.Conv2d(unet_channels[0], num_classes, kernel_size=1)

    def forward(
        self, rgb: torch.Tensor, depth: torch.Tensor, camera: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict the BEV of RGB frames [B, H, W, 3] (uint8), their depth [B, H, W, 1] and camera [B, 21].

        Returns:
            The elevation [B, X, Y] in metres, the class logits [B, K, X, Y] and the lifted input [B, 3 + C, X, Y]
            (see ``lift_to_bev``). The lifted geometry is always returned, also when the U-Net does not see it.
        """
        features = self.adapter(self.dino(rgb)) if self.dino is not None else None
        lifted = lift_to_bev(depth, camera, features, self.x_range, self.y_range, self.resolution)
        x = self.unet(lifted if self.use_geometry else lifted[:, 3:])
        return self.elevation_head(x).squeeze(1), self.semantic_head(x), lifted


def load_bev_model(path: str, device: str | torch.device = "cpu") -> RoverBevModel:
    """A frozen ``RoverBevModel`` from a ``train_bev.py`` checkpoint.

    The checkpoint may leave out the frozen DINOv3 backbone, as ``save_slim_bev_model`` does; its weights then come from
    ``dinov3_path``, like at training time.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = RoverBevModel(**checkpoint["model_kwargs"]).to(device).eval()
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if unexpected or any(not key.startswith("dino.") for key in missing):
        raise ValueError(f"{path} does not match the model: missing {missing}, unexpected {unexpected}.")
    return model.requires_grad_(False)


def save_slim_bev_model(checkpoint_path: str, path: str) -> None:
    """Save a ``train_bev.py`` checkpoint without the frozen DINOv3 backbone and the optimizer, for deployment."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {key: value for key, value in checkpoint["model_state_dict"].items() if not key.startswith("dino.")}
    torch.save({"model_state_dict": state, "model_kwargs": checkpoint["model_kwargs"], "source": checkpoint_path}, path)
