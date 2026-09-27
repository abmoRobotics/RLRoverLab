"""Online measurement of the odometry drift from re-observed terrain (Phase 3.3).

The rover does not know how much its odometry drifts, but it sees the effect: while it drives, the terrain it keeps
observing should stay where it first saw it. ``BevMemory`` keeps the first observation of every world cell. Without
drift, a cell's current and first observations differ only by perception noise, however far the rover has driven in
between. With drift the difference grows with that distance, and its growth measures the drift level of the ground the
rover is on. No ground truth enters, and the perception noise floor is measured the same way, on cells first seen only
just before.

How much a given drift moves the elevation depends on the terrain: a horizontal shift changes the elevation of a slope
but not of flat ground. The monitor therefore compares the observed excess variance with the excess that a nominal
odometry noise model predicts for the same cells at 1x. The square root of their ratio is the drift level in units of
that model, which only needs the model's shape, not its scale.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from rover_envs.envs.navigation.learning.bev.memory import CURRENT

# Nominal odometry noise at 1x as random walks per metre travelled: translation (m^2/m), heading (rad^2/m) and height
# (m^2/m). These are the random-walk equivalents of ``OdometryDrift`` at about 0.5 m/s, whose correlated biases act as
# random walks over their ~2 m correlation length. Only their ratios matter to the level estimate's ranking.
NOMINAL_RATES = (8.5e-4, 3.0e-4, 2.6e-4)


def local_relief(elevation: torch.Tensor) -> torch.Tensor:
    """Largest elevation difference [B, X, Y] between each cell and its 8 neighbours, ignoring unknown (NaN) cells."""
    batch, nx, ny = elevation.shape
    neighbours = F.unfold(F.pad(elevation.unsqueeze(1), (1, 1, 1, 1), value=math.nan), 3)
    return (neighbours - elevation.view(batch, 1, -1)).abs().nan_to_num(0.0).amax(dim=1).view(batch, nx, ny)


def nominal_effect(slope: torch.Tensor, distance: torch.Tensor, range_: torch.Tensor) -> torch.Tensor:
    """Elevation variance (m^2) that the nominal 1x drift adds to a cell over ``distance`` metres of travel.

    A horizontal displacement changes the elevation by the ``slope`` times its component along the slope, which has half
    its variance. Heading errors displace a cell in proportion to its distance from the path; for a rover that drove
    ``distance`` metres towards a cell now ``range_`` metres away, the mean squared lever arm is
    range^2 + range * distance + distance^2 / 3.
    """
    translation, heading, height = NOMINAL_RATES
    lever = range_**2 + range_ * distance + distance**2 / 3
    horizontal = translation * distance + heading * distance * lever
    return slope**2 * horizontal / 2 + height * distance


def revisit_statistics(
    persistent: dict[str, torch.Tensor],
    resolution: float,
    near: tuple[float, float],
    far: tuple[float, float],
    truncation: float,
) -> torch.Tensor:
    """This frame's evidence [B, 5] from current cells observed before: the sum of squared current-minus-first
    elevation differences (each truncated at ``truncation``) and the count of the cells first seen ``near`` metres of
    travel ago, then the same for ``far`` and the sum of their ``nominal_effect``."""
    current = (persistent["status"] == CURRENT) & torch.isfinite(persistent["first_elevation"])
    difference = (persistent["elevation"] - persistent["first_elevation"]).clamp(-truncation, truncation) ** 2
    distance = persistent["first_distance"]
    statistics = []
    for k, (low, high) in enumerate((near, far)):
        cells = current & (distance > low) & (distance <= high)
        statistics += [torch.where(cells, difference, 0.0).sum(dim=(1, 2)), cells.sum(dim=(1, 2)).float()]
        if k:
            slope = local_relief(persistent["elevation"]) / resolution
            effect = nominal_effect(slope, distance.nan_to_num(), persistent["range"].nan_to_num())
            statistics.append(torch.where(cells, effect.clamp(max=truncation**2), 0.0).sum(dim=(1, 2)))
    return torch.stack(statistics, dim=1)


class DriftMonitor:
    """Per-environment running estimate of the drift level of the ground the rover is on, from re-observed terrain.

    Every step adds the ``revisit_statistics`` of the persistent BEV to running sums that forget with the distance
    travelled, over ``memory_length`` metres. The mean squared difference of cells first seen ``near`` is the perception
    floor, ``floor`` (m^2). The far cells' mean squared difference minus the floor is the drift's contribution,
    ``excess`` (m^2). The square root of the excess over their mean nominal effect is the drift ``level``, 1 at the
    nominal noise.
    Differences are truncated at ``truncation`` metres, so rare view changes, e.g. at the edge of a rock seen from far
    and then from near, do not dominate. ``weight`` [B] is the number of far cells behind the estimate.
    """

    def __init__(
        self,
        num_envs: int,
        resolution: float,
        near: tuple[float, float] = (0.0, 0.25),
        far: tuple[float, float] = (1.0, 3.0),
        memory_length: float = 5.0,
        truncation: float = 0.1,
        device: str | torch.device = "cpu",
    ):
        self.resolution, self.near, self.far = resolution, near, far
        self.memory_length, self.truncation = memory_length, truncation
        self.sums = torch.zeros(num_envs, 5, device=device)

    def reset(self, envs: torch.Tensor) -> None:
        """Forget everything about the environments where ``envs`` [B] is true."""
        self.sums[envs] = 0.0

    def step(self, persistent: dict[str, torch.Tensor], moved: torch.Tensor) -> dict[str, torch.Tensor]:
        """Add the current frame's evidence; ``moved`` [B] is the distance travelled since the previous step.

        Returns ``floor``, ``excess`` and ``level`` [B], NaN until there is evidence for them, ``weight`` [B], and this
        step's ``statistics`` [B, 5] (see ``revisit_statistics``).
        """
        statistics = revisit_statistics(persistent, self.resolution, self.near, self.far, self.truncation)
        self.sums = torch.exp(-moved / self.memory_length).unsqueeze(1) * self.sums + statistics
        near_sum, near_count, far_sum, far_count, far_effect = self.sums.unbind(1)
        floor = torch.where(near_count > 0, near_sum / near_count.clamp(min=1e-9), torch.nan)
        spread = torch.where(far_count > 0, far_sum / far_count.clamp(min=1e-9), torch.nan)
        excess = (spread - floor).clamp(min=0.0)
        level = (excess * far_count / far_effect.clamp(min=1e-12)).sqrt()
        return {"floor": floor, "excess": excess, "level": level, "weight": far_count, "statistics": statistics}
