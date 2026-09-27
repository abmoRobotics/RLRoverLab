"""Checks of the world-aligned BEV memory on small synthetic BEVs; no Isaac Sim needed."""

import math

import torch

from rover_envs.envs.navigation.learning.bev.memory import CURRENT, REMEMBERED, UNKNOWN, BevMemory

# A 20 x 20 ego BEV at 10 cm around the BEV origin, in a 4 m memory window.
X_RANGE, Y_RANGE, RESOLUTION, DT = (-1.0, 1.0), (-1.0, 1.0), 0.1, 0.2
CELLS = 20
CENTRES = -1.0 + (torch.arange(CELLS) + 0.5) * RESOLUTION


def memory(num_envs: int = 1, **kwargs) -> BevMemory:
    return BevMemory(num_envs, 2, X_RANGE, Y_RANGE, RESOLUTION, DT, size=40, **kwargs)


def pose(x=0.0, y=0.0, z=0.0, yaw=0.0, num_envs=1) -> torch.Tensor:
    return torch.tensor([[x, y, z, yaw]]).repeat(num_envs, 1)


def step(bev: BevMemory, at: torch.Tensor, elevation=None, evidence=None, semantics=None) -> dict[str, torch.Tensor]:
    num_envs = at.shape[0]
    elevation = torch.zeros(num_envs, CELLS, CELLS) if elevation is None else elevation
    semantics = torch.stack((elevation, -elevation), dim=1) if semantics is None else semantics
    evidence = torch.zeros(num_envs, CELLS, CELLS, dtype=torch.bool) if evidence is None else evidence
    return bev.step(at, elevation, semantics, evidence)


def everywhere(num_envs: int = 1) -> torch.Tensor:
    return torch.ones(num_envs, CELLS, CELLS, dtype=torch.bool)


def test_current_cells_pass_through_and_unobserved_cells_are_unknown():
    bev, elevation = memory(), torch.randn(1, CELLS, CELLS)
    evidence = torch.zeros(1, CELLS, CELLS, dtype=torch.bool)
    evidence[:, 10:] = True
    out = step(bev, pose(0.3, -0.2, 0.5), elevation, evidence)
    assert (out["status"][evidence] == CURRENT).all() and (out["status"][~evidence] == UNKNOWN).all()
    assert torch.equal(out["elevation"][evidence], elevation[evidence]) and out["elevation"][~evidence].isnan().all()
    assert (out["age"][evidence] == 0).all() and (out["distance"][evidence] == 0).all()


def test_write_then_read_at_the_same_pose():
    bev, elevation = memory(), torch.randn(1, CELLS, CELLS)
    step(bev, pose(0.3, -0.2, 0.5), elevation, everywhere())
    out = step(bev, pose(0.3, -0.2, 0.5))
    assert (out["status"] == REMEMBERED).all()
    torch.testing.assert_close(out["elevation"], elevation)
    torch.testing.assert_close(out["semantics"], torch.stack((elevation, -elevation), dim=1))


def test_translation_keeps_terrain_in_place():
    bev, elevation = memory(), torch.zeros(1, CELLS, CELLS)
    elevation[0, 12, 7] = 1.0
    step(bev, pose(0.3, -0.2), elevation, everywhere())
    out = step(bev, pose(0.3 + 0.5, -0.2 - 0.3))  # moved 5 cells forward and 3 cells to the right
    assert (out["elevation"] == 1.0).nonzero().tolist() == [[0, 12 - 5, 7 + 3]]
    # Everything left over from the first footprint is remembered, the rest is unknown.
    assert (out["status"][0, :15, 3:] == REMEMBERED).all() and (out["status"][0, 15:] == UNKNOWN).all()


def test_rotation_keeps_terrain_in_place():
    bev, elevation = memory(), torch.zeros(1, CELLS, CELLS)
    elevation[0, 16, 13] = 1.0  # at ego (0.65, 0.35) m
    step(bev, pose(0.4, 0.4), elevation, everywhere())
    out = step(bev, pose(0.4, 0.4, yaw=math.pi / 2))
    # Turning left by 90 degrees moves a point at ego (x, y) to (y, -x) = (0.35, -0.65) m.
    assert (out["elevation"] == 1.0).nonzero().tolist() == [[0, 13, 3]]


def test_rotation_by_an_arbitrary_angle_resamples_within_a_cell():
    # A plane seen from yaw 0 and read back after turning by 30 degrees on the spot.
    gradient = torch.tensor([0.2, -0.1])
    ego = torch.stack(torch.meshgrid(CENTRES, CENTRES, indexing="ij"), dim=-1)
    bev = memory()
    step(bev, pose(0.4, 0.4), (ego @ gradient).unsqueeze(0), everywhere())
    yaw = math.radians(30)
    out = step(bev, pose(0.4, 0.4, yaw=yaw))
    rotation = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    expected = (ego @ rotation.T) @ gradient
    remembered = out["status"][0] == REMEMBERED
    assert remembered.sum() > 250
    # Two nearest-cell resamplings move each sample by at most one cell diagonal.
    assert (out["elevation"][0] - expected)[remembered].abs().max() <= gradient.norm() * RESOLUTION * math.sqrt(2)


def test_heights_follow_the_rover_up_a_slope():
    bev = memory()
    step(bev, pose(z=0.0), torch.full((1, CELLS, CELLS), -0.3), everywhere())
    out = step(bev, pose(z=0.2))
    torch.testing.assert_close(out["elevation"], torch.full((1, CELLS, CELLS), -0.5))


def test_age_and_distance_since_observation():
    bev = memory()
    step(bev, pose(0.3, 0.3), torch.zeros(1, CELLS, CELLS), everywhere())
    for moved in (0.1, 0.2, 0.3):
        out = step(bev, pose(0.3 + moved, 0.3))
    remembered = out["status"] == REMEMBERED
    torch.testing.assert_close(out["age"][remembered], torch.full((int(remembered.sum()),), 3 * DT))
    torch.testing.assert_close(out["distance"][remembered], torch.full((int(remembered.sum()),), 0.3))


def test_environments_are_isolated_and_reset_independently():
    bev = memory(num_envs=2)
    evidence = everywhere(2)
    evidence[1] = False
    step(bev, pose(num_envs=2), torch.ones(2, CELLS, CELLS), evidence)
    out = step(bev, pose(num_envs=2))
    assert (out["status"][0] == REMEMBERED).all() and (out["status"][1] == UNKNOWN).all()

    step(bev, pose(num_envs=2), torch.ones(2, CELLS, CELLS), everywhere(2))
    bev.reset(torch.tensor([True, False]))
    out = step(bev, pose(num_envs=2))
    assert (out["status"][0] == UNKNOWN).all() and (out["status"][1] == REMEMBERED).all()
    torch.testing.assert_close(bev.clock, torch.tensor([DT, 4 * DT]))


def test_closest_write_rule_keeps_the_nearest_observation():
    # The world cell at (0.95, 0.05) m is seen from 0.95 m away, then from 0.45 m and then from 0.95 m again.
    for rule, expected in (("latest", 3.0), ("closest", 2.0)):
        bev = memory(write_rule=rule)
        for x, value in ((0.0, 1.0), (0.5, 2.0), (0.0, 3.0)):
            evidence = torch.zeros(1, CELLS, CELLS, dtype=torch.bool)
            evidence[0, int((0.95 - x + 1.0) / RESOLUTION), 10] = True
            step(bev, pose(x, 0.0), torch.full((1, CELLS, CELLS), value), evidence)
        out = step(bev, pose(0.0, 0.0))
        assert out["elevation"][0, 19, 10].item() == expected, rule


def test_remembered_terrain_stays_aligned_while_driving():
    """Drive a curve over smooth terrain, seeing only a wedge ahead, and compare the memory with the terrain."""

    def terrain(wx, wy):
        return 0.1 * torch.sin(wx) + 0.05 * torch.cos(1.3 * wy)

    max_slope = 0.1 + 0.05 * 1.3
    bev = memory(write_rule="latest")
    ego_x, ego_y = torch.meshgrid(CENTRES, CENTRES, indexing="ij")
    wedge = (ego_x > 0.3) & (ego_y.abs() < ego_x)
    x, y, z, yaw = 0.123, -0.456, 0.02, 0.3
    for _ in range(25):
        world_x = x + math.cos(yaw) * ego_x - math.sin(yaw) * ego_y
        world_y = y + math.sin(yaw) * ego_x + math.cos(yaw) * ego_y
        out = step(bev, pose(x, y, z, yaw), (terrain(world_x, world_y) - z).unsqueeze(0), wedge.unsqueeze(0))
        x, y, z, yaw = x + 0.07 * math.cos(yaw), y + 0.07 * math.sin(yaw), z + 0.013, yaw + 0.11

    out = step(bev, pose(x, y, z, yaw))
    remembered = out["status"][0] == REMEMBERED
    world_x = x + math.cos(yaw) * ego_x - math.sin(yaw) * ego_y
    world_y = y + math.sin(yaw) * ego_x + math.cos(yaw) * ego_y
    error = (out["elevation"][0] - (terrain(world_x, world_y) - z))[remembered].abs()
    assert remembered.sum() > 100 and error.max() <= max_slope * RESOLUTION * math.sqrt(2)
    assert (out["distance"][0][remembered] > 0).all()
