"""Checks of the online drift measurement from re-observed terrain; no Isaac Sim needed."""

import math

import torch

from rover_envs.envs.navigation.learning.bev.drift import (NOMINAL_RATES, DriftMonitor, local_relief, nominal_effect,
                                                           revisit_statistics)
from rover_envs.envs.navigation.learning.bev.memory import CURRENT, REMEMBERED, BevMemory

CELLS, RESOLUTION = 20, 0.1


def frame(near_difference: float, far_difference: float, num_envs: int = 1) -> dict[str, torch.Tensor]:
    """A flat persistent BEV whose current cells were first seen 0.1 m (left half) or 2 m (right half) back."""
    status = torch.full((num_envs, CELLS, CELLS), CURRENT)
    status[:, :, -2:] = REMEMBERED  # not current: ignored
    distance = torch.full((num_envs, CELLS, CELLS), 0.1)
    distance[:, :, CELLS // 2:] = 2.0
    first = torch.where(distance < 1.0, near_difference, far_difference)
    # Alternate the sign, so the differences have zero mean like perception noise.
    first = first * torch.where(torch.arange(CELLS) % 2 == 0, 1.0, -1.0)
    return {"status": status, "elevation": torch.zeros(num_envs, CELLS, CELLS), "first_elevation": first,
            "first_distance": distance, "range": torch.full((num_envs, CELLS, CELLS), 1.0)}


def monitor(**kwargs) -> DriftMonitor:
    return DriftMonitor(kwargs.pop("num_envs", 1), RESOLUTION, **kwargs)


def test_local_relief_is_the_largest_difference_to_a_known_neighbour():
    elevation = torch.zeros(1, 5, 5)
    elevation[0, 2, 2] = 0.3
    elevation[0, 0, 0] = torch.nan
    relief = local_relief(elevation)
    assert relief[0, 2, 2] == 0.3 and relief[0, 1, 1] == 0.3 and relief[0, 3, 4] == 0.0
    assert relief[0, 0, 0] == 0.0 and relief[0, 0, 1] == 0.0


def test_the_nominal_effect_grows_with_distance_slope_and_lever_arm():
    translation, heading, height = NOMINAL_RATES
    flat = nominal_effect(torch.tensor(0.0), torch.tensor(2.0), torch.tensor(1.0))
    torch.testing.assert_close(flat, torch.tensor(2 * height))
    sloped = nominal_effect(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(1.0))
    lever = 1 + 2 + 4 / 3
    torch.testing.assert_close(sloped, torch.tensor((translation * 2 + heading * 2 * lever) / 2 + 2 * height))


def test_revisit_statistics_split_cells_by_the_distance_since_first_seen():
    statistics = revisit_statistics(frame(0.01, 0.03), RESOLUTION, near=(0.0, 0.25), far=(1.0, 3.0), truncation=0.1)
    near_cells, far_cells = CELLS * CELLS // 2, CELLS * (CELLS // 2 - 2)
    # Flat ground: the nominal effect of a far cell is the height drift over 2 m.
    expected = [near_cells * 1e-4, near_cells, far_cells * 9e-4, far_cells, far_cells * 2 * NOMINAL_RATES[2]]
    torch.testing.assert_close(statistics[0], torch.tensor(expected))
    # Differences beyond the truncation count as the truncation.
    truncated = revisit_statistics(frame(0.01, 0.3), RESOLUTION, near=(0.0, 0.25), far=(1.0, 3.0), truncation=0.1)
    torch.testing.assert_close(truncated[0, 2], torch.tensor(far_cells * 0.01))


def test_the_level_is_the_excess_over_the_nominal_effect():
    out = monitor().step(frame(0.01, 0.03), moved=torch.tensor([0.1]))
    torch.testing.assert_close(out["floor"], torch.tensor([1e-4]))
    torch.testing.assert_close(out["excess"], torch.tensor([8e-4]))
    # The nominal 1x drift would add 2 m of height drift variance: the measured excess is this many times that.
    torch.testing.assert_close(out["level"], torch.tensor([math.sqrt(8e-4 / (2 * NOMINAL_RATES[2]))]))


def test_the_monitor_forgets_with_the_distance_travelled():
    drift = monitor(memory_length=2.0)
    drift.step(frame(0.01, 0.05), moved=torch.tensor([0.0]))
    # 20 m later on better ground, the old evidence weighs exp(-10) of what it did.
    for _ in range(100):
        out = drift.step(frame(0.01, 0.02), moved=torch.tensor([0.2]))
    torch.testing.assert_close(out["excess"], torch.tensor([3e-4]), rtol=1e-3, atol=0)
    # A rover that stands still keeps what it measured.
    before = out["level"].clone()
    out = drift.step(frame(0.01, 0.02), moved=torch.tensor([0.0]))
    torch.testing.assert_close(out["level"], before)


def test_resets_forget_and_estimates_are_undefined_without_evidence():
    drift = monitor(num_envs=2)
    drift.step(frame(0.01, 0.03, num_envs=2), moved=torch.tensor([0.1, 0.1]))
    drift.reset(torch.tensor([True, False]))
    empty = frame(0.01, 0.03, num_envs=2)
    empty["status"][:] = REMEMBERED
    out = drift.step(empty, moved=torch.tensor([0.1, 0.1]))
    assert out["level"][0].isnan() and torch.isfinite(out["level"][1]) and out["weight"][0] == 0


def test_drift_shows_as_terrain_that_moves_between_observations():
    """Drive a curve over rolling terrain seeing a wedge ahead, with exact and with drifting headings."""

    def terrain(wx, wy):
        return 0.15 * torch.sin(2.0 * wx) * torch.cos(1.5 * wy)

    centres = -1.0 + (torch.arange(CELLS) + 0.5) * RESOLUTION
    ego_x, ego_y = torch.meshgrid(centres, centres, indexing="ij")
    wedge = (ego_x > 0.1) & (ego_y.abs() < ego_x)
    level = {}
    for heading_drift in (0.0, 1.0):
        memory = BevMemory(1, 1, (-1.0, 1.0), (-1.0, 1.0), RESOLUTION, 0.2, size=40)
        drift = monitor(far=(0.4, 1.0))
        x, y, yaw, estimated_yaw = 0.0, 0.0, 0.2, 0.2
        for _ in range(40):
            # The elevation is observed in the true frame; the memory places it with the estimated heading.
            world_x = x + math.cos(yaw) * ego_x - math.sin(yaw) * ego_y
            world_y = y + math.sin(yaw) * ego_x + math.cos(yaw) * ego_y
            pose = torch.tensor([[x, y, 0.0, estimated_yaw]])
            persistent = memory.step(pose, terrain(world_x, world_y).unsqueeze(0), torch.zeros(1, 1, CELLS, CELLS),
                                     wedge.unsqueeze(0))
            out = drift.step(persistent, moved=torch.tensor([0.05]))
            x, y = x + 0.05 * math.cos(yaw), y + 0.05 * math.sin(yaw)
            yaw, estimated_yaw = yaw + 0.02, estimated_yaw + 0.02 + heading_drift * 0.05
        level[heading_drift] = out["level"].item()
    # Exact poses leave resampling differences at these 10 cm cells, which do not grow with the distance travelled;
    # drift does, well above the nominal level (the truncation caps this toy's large differences).
    assert level[1.0] > 1 and level[0.0] < 0.1 * level[1.0], level
