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


def observe_one_cell(bev: BevMemory, probabilities: list[tuple[float, float]], at=None) -> dict[str, torch.Tensor]:
    """Observe ego cell (12, 7) once per class distribution in ``probabilities``, then read it back."""
    at = pose(0.3, -0.2) if at is None else at
    evidence = torch.zeros(1, CELLS, CELLS, dtype=torch.bool)
    evidence[0, 12, 7] = True
    for p in probabilities:
        step(bev, at, evidence=evidence, semantics=torch.tensor(p).view(1, 2, 1, 1).expand(1, 2, CELLS, CELLS))
    return step(bev, at)


def test_class_fusion_combines_repeated_observations():
    observations = [(0.9, 0.1), (0.2, 0.8), (0.4, 0.6)]
    # The mean of the probabilities, the normalized product (0.072 : 0.048), and the latest observation.
    for fusion, expected in (("mean", (0.5, 0.5)), ("log_odds", (0.6, 0.4)), ("latest", (0.4, 0.6))):
        out = observe_one_cell(memory(class_fusion=fusion), observations)
        torch.testing.assert_close(out["semantics"][0, :, 12, 7], torch.tensor(expected), msg=fusion)


def test_a_fusion_window_lets_the_mean_follow_changes():
    observations = [(1.0, 0.0)] * 5 + [(0.0, 1.0)] * 2
    for window, expected in ((0, (5 / 7, 2 / 7)), (2, (0.25, 0.75))):
        out = observe_one_cell(memory(class_fusion="mean", fusion_window=window), observations)
        torch.testing.assert_close(out["semantics"][0, :, 12, 7], torch.tensor(expected), msg=f"window {window}")


def test_log_odds_fusion_is_clamped_so_it_can_change_its_mind():
    bev = memory(class_fusion="log_odds", min_probability=0.02)
    out = observe_one_cell(bev, [(0.99, 0.01)] * 20)
    torch.testing.assert_close(out["semantics"][0, :, 12, 7], torch.tensor([0.98, 0.02]), rtol=0, atol=1e-3)
    # Two confident contrary observations suffice: 0.98 * 0.1 : 0.02 * 0.9, then 0.847 * 0.1 : 0.153 * 0.9.
    out = observe_one_cell(bev, [(0.1, 0.9)] * 2)
    assert out["semantics"][0, 1, 12, 7] > 0.5


def test_class_fusion_restarts_when_another_world_cell_takes_the_slot():
    bev = memory(class_fusion="mean")
    observe_one_cell(bev, [(1.0, 0.0)], at=pose(0.3, -0.2))
    # 4 m further on, the same ego cell is another world cell in the same slot of the 40-slot window.
    out = observe_one_cell(bev, [(0.0, 1.0)], at=pose(0.3 + 4.0, -0.2))
    torch.testing.assert_close(out["semantics"][0, :, 12, 7], torch.tensor([0.0, 1.0]))


def test_class_fusion_uses_observations_the_closest_rule_skips_for_elevation():
    # As in the closest-rule test: the cell at (0.95, 0.05) m is seen from 0.95 m, 0.45 m and 0.95 m away.
    bev = memory(write_rule="closest", class_fusion="mean")
    for x, value, rock in ((0.0, 1.0, 0.0), (0.5, 2.0, 0.0), (0.0, 3.0, 1.0)):
        evidence = torch.zeros(1, CELLS, CELLS, dtype=torch.bool)
        evidence[0, int((0.95 - x + 1.0) / RESOLUTION), 10] = True
        semantics = torch.tensor([1.0 - rock, rock]).view(1, 2, 1, 1).expand(1, 2, CELLS, CELLS)
        step(bev, pose(x, 0.0), torch.full((1, CELLS, CELLS), value), evidence, semantics)
    out = step(bev, pose(0.0, 0.0))
    assert out["elevation"][0, 19, 10].item() == 2.0
    torch.testing.assert_close(out["semantics"][0, :, 19, 10], torch.tensor([2 / 3, 1 / 3]))


def test_the_observation_range_is_reported():
    bev = memory()
    out = step(bev, pose(0.3, -0.2), evidence=everywhere())
    ego_range = torch.hypot(*torch.meshgrid(CENTRES, CENTRES, indexing="ij"))
    torch.testing.assert_close(out["range"][0], ego_range)
    out = step(bev, pose(0.3 + 0.5, -0.2))
    # The cell now at ego (0.45, 0.05) m was observed at (0.95, 0.05) m.
    torch.testing.assert_close(out["range"][0, 14, 10], torch.tensor(math.hypot(0.95, 0.05)))
    assert out["range"][0][out["status"][0] == UNKNOWN].isnan().all()


def test_the_first_observation_is_kept_while_the_latest_is_updated():
    # The world cell at (0.95, 0.05) m is seen with elevation 1 from x = 0, then with 2 from x = 0.5, 0.2 m higher.
    bev = memory()
    for x, z, value in ((0.0, 0.0, 1.0), (0.5, 0.2, 2.0)):
        evidence = torch.zeros(1, CELLS, CELLS, dtype=torch.bool)
        evidence[0, int((0.95 - x + 1.0) / RESOLUTION), 10] = True
        step(bev, pose(x, 0.0, z), torch.full((1, CELLS, CELLS), value), evidence)
    out = step(bev, pose(0.5, 0.0, 0.2))
    # The cell is now at ego (0.45, 0.05) m; its first observation was 1 m in the world, 0.8 m below this frame.
    assert out["elevation"][0, 14, 10].item() == 2.0
    torch.testing.assert_close(out["first_elevation"][0, 14, 10], torch.tensor(0.8))
    torch.testing.assert_close(out["first_distance"][0, 14, 10], torch.tensor(0.5))
    torch.testing.assert_close(out["first_age"][0, 14, 10], torch.tensor(2 * DT))
    torch.testing.assert_close(out["first_range"][0, 14, 10], torch.tensor(math.hypot(0.95, 0.05)))


def test_first_observations_restart_with_the_slot_and_are_unknown_for_new_cells():
    bev = memory()
    out = step(bev, pose(0.3, -0.2), torch.ones(1, CELLS, CELLS), everywhere())
    assert out["first_elevation"].isnan().all() and out["first_distance"].isnan().all()
    # 4 m on, every ego cell is another world cell in the same slot of the 40-slot window.
    step(bev, pose(0.3 + 4.0, -0.2), torch.full((1, CELLS, CELLS), 2.0), everywhere())
    out = step(bev, pose(0.3 + 4.0, -0.2))
    torch.testing.assert_close(out["first_elevation"][0], torch.full((CELLS, CELLS), 2.0))


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
