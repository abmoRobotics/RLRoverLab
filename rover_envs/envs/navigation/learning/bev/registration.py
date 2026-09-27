"""Online measurement of the odometry drift from re-observed terrain (Phase 3.3).

The rover cannot know how much its odometry drifts, but it can see it: terrain observed again after driving on should
look the same. Every ``keyframe_distance`` metres of odometry, the current depth lift is registered against the lift of
the keyframe, the frame that far back along the route. The rigid correction that aligns the two is the drift the
odometry accumulated in between. It is zero with exact odometry.

Heights are the lift's mean point elevation, from the BEV frame origin; with the pose's z they are world heights. Poses
are the estimated BEV frame poses [B, 4] (x, y, z, yaw) that the memory also receives.

The correction (x, y, yaw, z) is expressed in the current BEV frame: a current ego point q truly lies at
R(yaw) q + (x, y) in the frame the odometry believes the rover is in, and its height is off by z. Equivalently, with
the true and odometry poses T and T' of the keyframe (k) and now (t), the correction is (T'_k^-1 T'_t)^-1 (T_k^-1 T_t).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def rotate(xy: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate 2D vectors [..., 2] by ``angle`` [...]."""
    cos, sin = angle.cos(), angle.sin()
    return torch.stack((cos * xy[..., 0] - sin * xy[..., 1], sin * xy[..., 0] + cos * xy[..., 1]), dim=-1)


def relative_correction(true_key: torch.Tensor, true_now: torch.Tensor, key: torch.Tensor, now: torch.Tensor):
    """The correction [..., 4] (x, y, yaw, z) that ``KeyframeRegistration`` should find, from true and odometry
    poses [..., 4] of the keyframe and now. For evaluation in simulation."""
    true_yaw, yaw = true_now[..., 3] - true_key[..., 3], now[..., 3] - key[..., 3]
    true_shift = rotate(true_now[..., :2] - true_key[..., :2], -true_key[..., 3])
    shift = rotate(now[..., :2] - key[..., :2], -key[..., 3])
    turn = torch.remainder(true_yaw - yaw + math.pi, 2 * math.pi) - math.pi
    height = (key[..., 2] - true_key[..., 2]) - (now[..., 2] - true_now[..., 2])
    return torch.cat((rotate(true_shift - shift, -yaw), turn.unsqueeze(-1), height.unsqueeze(-1)), dim=-1)


class KeyframeRegistration:
    """Measures the drift over each ``keyframe_distance`` metres of odometry, for a batch of environments.

    The search is a grid over (x, y, yaw) per stage, centred on the previous stage's best: each stage is
    (translation half-width, translation step, yaw half-width, yaw step), in metres and radians. It uses up to
    ``samples`` current cells with at least ``min_points`` depth points within ``max_range``. All hypotheses of a stage
    are scored on the same cells: those that land among the keyframe's observed cells under every hypothesis, so that
    no hypothesis wins by gaining or losing overlap. The height offset of a hypothesis is the median residual, and its
    cost the mean absolute residual truncated at ``truncation``. The best hypothesis of the last stage is refined by a
    parabola along each axis, whose curvature tells how well the axis is observed: flat ground leaves the horizontal
    correction unobservable, and a low curvature reports that.
    """

    def __init__(
        self,
        num_envs: int,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        resolution: float,
        keyframe_distance: float = 0.5,
        min_points: int = 3,
        max_range: float = 4.0,
        samples: int = 1024,
        truncation: float = 0.05,
        min_overlap: float = 0.2,
        stages: tuple[tuple[float, float, float, float], ...] = (
            (0.12, 0.03, math.radians(3.0), math.radians(0.75)),
            (0.03, 0.01, math.radians(0.75), math.radians(0.25)),
        ),
        chunk: int = 64,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        self.x_range, self.y_range, self.resolution = tuple(x_range), tuple(y_range), resolution
        self.shape = (round((x_range[1] - x_range[0]) / resolution), round((y_range[1] - y_range[0]) / resolution))
        self.keyframe_distance, self.min_points, self.max_range = keyframe_distance, min_points, max_range
        self.samples, self.truncation, self.min_overlap = samples, truncation, min_overlap
        self.stages, self.chunk = stages, chunk
        self.generator = torch.Generator(device).manual_seed(seed)
        x = x_range[0] + (torch.arange(self.shape[0], device=device) + 0.5) * resolution
        y = y_range[0] + (torch.arange(self.shape[1], device=device) + 0.5) * resolution
        self.cell_xy = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1).view(-1, 2)
        self.key_height = torch.zeros(num_envs, *self.shape, device=device)
        self.key_observed = torch.zeros(num_envs, *self.shape, dtype=torch.bool, device=device)
        self.key_pose = torch.zeros(num_envs, 4, device=device)
        self.key_odometer = torch.zeros(num_envs, device=device)
        self.has_key = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.odometer = torch.zeros(num_envs, device=device)
        self.last_xy = torch.full((num_envs, 2), torch.nan, device=device)

    def step(
        self, pose: torch.Tensor, height: torch.Tensor, count: torch.Tensor, reset: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Advance one step with the current frame.

        Args:
            pose: Odometry pose [B, 4] of the BEV frame.
            height: The depth lift's mean point elevation [B, X, Y] in metres from the BEV frame origin.
            count: Depth points per cell [B, X, Y].
            reset: [B], true where the environment restarted in this step; that frame is not used.

        Returns:
            ``registered`` [B]: true where a correction was measured in this step. For those: ``correction`` [B, 4]
            (x, y, yaw, z), ``interval`` [B], the odometry distance since the keyframe, ``cost`` [B], ``overlap`` [B],
            the share of the current cells the keyframe observed too, and ``curvature`` [B, 3], the cost's curvature
            along x, y (per m^2) and yaw (per rad^2). Other environments hold NaN. ``keyframe`` [B]: true where the
            current frame became the new keyframe.
        """
        batch = pose.shape[0]
        self.has_key &= ~reset
        moved = torch.nan_to_num((pose[:, :2] - self.last_xy).norm(dim=1))
        self.odometer = torch.where(reset, 0.0, self.odometer + moved)
        self.last_xy = pose[:, :2].clone()
        observed = (count >= self.min_points) & torch.isfinite(height)

        interval = self.odometer - self.key_odometer
        due = self.has_key & ~reset & (interval >= self.keyframe_distance)
        nan = torch.full((batch,), torch.nan, device=pose.device)
        out = {
            "registered": torch.zeros(batch, dtype=torch.bool, device=pose.device),
            "correction": nan.unsqueeze(1).repeat(1, 4),
            "interval": torch.where(due, interval, torch.nan),
            "cost": nan.clone(),
            "overlap": nan.clone(),
            "curvature": nan.unsqueeze(1).repeat(1, 3),
        }
        if due.any():
            envs = due.nonzero().squeeze(1)
            correction, cost, overlap, curvature = self._register(envs, pose[envs], height[envs], observed[envs])
            usable = overlap >= self.min_overlap
            out["registered"][envs] = usable
            out["correction"][envs] = torch.where(usable.unsqueeze(1), correction, torch.nan)
            out["cost"][envs], out["overlap"][envs] = cost, overlap
            out["curvature"][envs] = torch.where(usable.unsqueeze(1), curvature, torch.nan)

        # The current frame becomes the keyframe where one was due or missing.
        new_key = (due | ~self.has_key) & ~reset
        self.key_height[new_key] = torch.nan_to_num(height[new_key])
        self.key_observed[new_key] = observed[new_key]
        self.key_pose[new_key] = pose[new_key]
        self.key_odometer[new_key] = self.odometer[new_key]
        self.has_key |= new_key
        out["keyframe"] = new_key
        return out

    def _register(self, envs: torch.Tensor, pose: torch.Tensor, height: torch.Tensor, observed: torch.Tensor):
        count, device = len(envs), pose.device
        key_pose = self.key_pose[envs]
        # The current frame in the keyframe's frame, as the odometry has it.
        turn = pose[:, 3] - key_pose[:, 3]
        shift = rotate(pose[:, :2] - key_pose[:, :2], -key_pose[:, 3])
        climb = pose[:, 2] - key_pose[:, 2]
        # A random sample of the current cells that are observed well enough and within range.
        usable = observed.flatten(1) & (self.cell_xy.norm(dim=1) <= self.max_range)
        draw = torch.rand(usable.shape, generator=self.generator, device=device).masked_fill(~usable, -1.0)
        top = draw.topk(min(self.samples, usable.shape[1]), dim=1)
        points, valid = self.cell_xy[top.indices], top.values >= 0
        heights = height.flatten(1).gather(1, top.indices)
        observed_key = self.key_observed[envs].float()
        maps = torch.stack((self.key_height[envs] * observed_key, observed_key), dim=1)
        context = (maps, points, heights, valid, shift, turn, climb)

        rows = torch.arange(count, device=device)
        best = torch.zeros(count, 3, device=device)
        for half_width, step, yaw_half_width, yaw_step in self.stages:
            hypotheses = best.unsqueeze(1) + self._grid(half_width, step, yaw_half_width, yaw_step, device)
            common = self._common(hypotheses, valid, context)
            cost, offset = self._cost(hypotheses, common, context)
            index = cost.argmin(dim=1)
            best, best_offset = hypotheses[rows, index], offset[rows, index]

        # Parabolic refinement and curvature along each axis of the last stage's grid, on one common set of cells.
        steps = torch.tensor((step, step, yaw_step), device=device)
        axes = torch.diag(steps)
        probes = best.unsqueeze(1) + torch.cat((torch.zeros(1, 3, device=device), axes, -axes))
        common &= self._common(probes, valid, context)
        probe_cost = self._cost(probes, common, context)[0]
        centre, plus, minus = probe_cost[:, :1], probe_cost[:, 1:4], probe_cost[:, 4:]
        bend = plus + minus - 2 * centre
        shift_along = torch.where(bend > 0, 0.5 * (minus - plus) / bend.clamp(min=1e-12), 0.0).clamp(-0.5, 0.5)
        best = best + shift_along * steps
        correction = torch.cat((best, best_offset.unsqueeze(1)), dim=1)
        overlap = common.sum(1) / valid.sum(1).clamp(min=1)
        return correction, centre.squeeze(1), overlap, bend / steps**2

    @staticmethod
    def _grid(half_width: float, step: float, yaw_half_width: float, yaw_step: float, device) -> torch.Tensor:
        xy = torch.arange(-round(half_width / step), round(half_width / step) + 1, device=device) * step
        yaw = torch.arange(-round(yaw_half_width / yaw_step), round(yaw_half_width / yaw_step) + 1, device=device)
        return torch.stack(torch.meshgrid(xy, xy, yaw * yaw_step, indexing="ij"), dim=-1).view(-1, 3)

    def _sample(self, hypotheses, maps, points, heights, shift, turn, climb):
        """Height residuals [E, H, N] of the current cells under the hypotheses [E, H, 3] (x, y, yaw corrections), and
        whether the keyframe observed the cells [E, H, N] they land on."""
        (x_min, _), (y_min, _), (nx, ny) = self.x_range, self.y_range, self.shape
        # Corrected current point R(yaw) q + (x, y), then into the keyframe frame by the odometry motion.
        yaw = (turn.unsqueeze(1) + hypotheses[..., 2]).unsqueeze(-1)
        moved = rotate(hypotheses[..., :2], turn.unsqueeze(1)) + shift.unsqueeze(1)
        px, py = points[..., 0].unsqueeze(1), points[..., 1].unsqueeze(1)
        x = yaw.cos() * px - yaw.sin() * py + moved[..., :1]
        y = yaw.sin() * px + yaw.cos() * py + moved[..., 1:]
        # Maps are [x, y]: grid_sample's first coordinate indexes their last dimension, y.
        grid = torch.stack(
            (2 * (y - y_min) / (self.resolution * ny) - 1, 2 * (x - x_min) / (self.resolution * nx) - 1), dim=-1
        )
        sampled = F.grid_sample(maps, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        residual = heights.unsqueeze(1) + climb.view(-1, 1, 1) - sampled[:, 0] / sampled[:, 1].clamp(min=1e-6)
        return residual, sampled[:, 1] > 0.999

    def _common(self, hypotheses, valid, context) -> torch.Tensor:
        """The valid current cells [E, N] that land on observed keyframe cells under every hypothesis."""
        maps, points, heights, _, shift, turn, climb = context
        common = valid.clone()
        for chunk in hypotheses.split(self.chunk, dim=1):
            common &= self._sample(chunk, maps, points, heights, shift, turn, climb)[1].all(dim=1)
        return common

    def _cost(self, hypotheses, common, context) -> tuple[torch.Tensor, torch.Tensor]:
        """Cost and height offset [E, H] of the hypotheses [E, H, 3] on the ``common`` cells [E, N]."""
        maps, points, heights, _, shift, turn, climb = context
        costs, offsets = [], []
        for chunk in hypotheses.split(self.chunk, dim=1):
            residual = self._sample(chunk, maps, points, heights, shift, turn, climb)[0]
            used = common.unsqueeze(1).expand_as(residual)
            offset = -residual.where(used, torch.nan).nanmedian(dim=-1).values
            error = (residual + offset.unsqueeze(-1)).abs().clamp(max=self.truncation)
            costs.append(error.where(used, 0.0).sum(-1) / common.sum(-1, keepdim=True).clamp(min=1))
            offsets.append(offset)
        return torch.cat(costs, dim=1), torch.cat(offsets, dim=1)
