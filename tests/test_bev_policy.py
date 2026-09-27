"""Checks of the Phase-4 policy BEV: pipeline, RSL-RL environment wrapper and BEV student; no Isaac Sim needed."""

import math

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from rover_envs.envs.navigation.learning.bev.pipeline import VARIANTS, PersistentBev, ReliabilityMap
from rover_envs.envs.navigation.learning.rsl_rl.bev_env import PersistentBevVecEnv
from rover_envs.envs.navigation.learning.rsl_rl.models import RoverBevStudentModel

# A 20 x 20 BEV at 10 cm around the BEV origin.
X_RANGE, Y_RANGE, RESOLUTION, DT, CELLS = (-1.0, 1.0), (-1.0, 1.0), 0.1, 0.2, 20
CENTRES = -1.0 + (torch.arange(CELLS) + 0.5) * RESOLUTION
EGO_X, EGO_Y = torch.meshgrid(CENTRES, CENTRES, indexing="ij")
WEDGE = (EGO_X > 0.2) & (EGO_Y.abs() < EGO_X)


class StubModel(torch.nn.Module):
    """Sees flat ground 0.3 m below the BEV origin in a wedge ahead; the network is 1 cm higher than the lift."""

    def forward(self, rgb, depth, camera):
        batch = rgb.shape[0]
        lifted = torch.zeros(batch, 3, CELLS, CELLS)
        lifted[:, 0] = torch.where(WEDGE, -0.3, 0.0)
        lifted[:, 2] = torch.where(WEDGE, math.log1p(5), 0.0)
        logits = torch.stack((torch.zeros(CELLS, CELLS), torch.where(EGO_Y > 0, 3.0, -3.0))).expand(batch, -1, -1, -1)
        return lifted[:, 0] + 0.01, logits, lifted


def reliability_map(tmp_path) -> ReliabilityMap:
    path = tmp_path / "map.npz"
    np.savez(path, edges=np.array([0.0, 1e-4, 1e-3, 1.0]), values=np.array([0.01, 0.02, 0.05]), weight=1.0, prior=2.0)
    return ReliabilityMap(str(path))


def pipeline(tmp_path, num_envs=2, **kwargs) -> PersistentBev:
    kwargs = {"variant": "full", "reliability": reliability_map(tmp_path), "severity": 0.0, **kwargs}
    return PersistentBev(StubModel(), num_envs, X_RANGE, Y_RANGE, RESOLUTION, ("ground", "rock"), DT, **kwargs)


def frame(num_envs=2):
    rgb, depth, camera = torch.zeros(num_envs, 4, 4, 3), torch.ones(num_envs, 4, 4, 1), torch.zeros(num_envs, 21)
    return rgb, depth, camera


def pose(x=0.0, num_envs=2):
    return torch.tensor([[x, 0.0, 0.0, 0.0]]).repeat(num_envs, 1)


def no_reset(num_envs=2):
    return torch.zeros(num_envs, dtype=torch.bool)


def test_the_reliability_map_looks_up_its_monotone_table(tmp_path):
    reliability = reliability_map(tmp_path)
    # Scores below the first edge or above the last take the first or last value.
    scores = torch.tensor([-1.0, 5e-5, 5e-4, 0.5, 7.0])
    index = (torch.searchsorted(reliability.edges, scores, right=True) - 1).clamp(0, 2)
    torch.testing.assert_close(reliability.values[index], torch.tensor([0.01, 0.01, 0.02, 0.05, 0.05]))


def test_every_variant_has_its_channels_and_no_nan(tmp_path):
    for variant, channels in VARIANTS.items():
        bev = pipeline(tmp_path, variant=variant).step(*frame(), pose(), no_reset())
        assert bev.shape == (2, len(channels), CELLS, CELLS) and torch.isfinite(bev).all(), variant


def test_current_remembered_and_unknown_cells(tmp_path):
    policy_bev = pipeline(tmp_path)
    policy_bev.step(*frame(), pose(0.0), no_reset())
    bev = policy_bev.step(*frame(), pose(0.5), no_reset())
    elevation, rock, known, current, error = bev[0]
    # The wedge is current; the cells the rover drove past are remembered; the rest is unknown and all zero.
    assert (current[WEDGE] == 1).all() and (known[WEDGE] == 1).all()
    remembered = (known == 1) & (current == 0)
    assert remembered.any() and ((elevation - -0.3).abs()[known == 1] < 1e-5).all()
    assert (bev[0][:, known == 0] == 0).all()
    # Rock probability follows the class logits; the expected error is positive where known, in decimetres.
    assert (rock[(known == 1) & (EGO_Y > 0)] > 0.9).all() and (rock[(known == 1) & (EGO_Y < 0)] < 0.1).all()
    assert (error[known == 1] > 0.1 - 1e-6).all() and (error[known == 1] < 0.5 + 1e-6).all()


def test_a_reset_empties_the_bev_of_that_environment_only(tmp_path):
    policy_bev = pipeline(tmp_path)
    policy_bev.step(*frame(), pose(0.0), no_reset())
    bev = policy_bev.step(*frame(), pose(0.3), torch.tensor([True, False]))
    assert (bev[0] == 0).all() and bev[1, 2].sum() > 0


def test_a_blind_camera_leaves_only_the_memory(tmp_path):
    policy_bev = pipeline(tmp_path)
    policy_bev.step(*frame(), pose(0.0), no_reset())
    bev = policy_bev.step(*frame(), pose(0.3), no_reset(), blind=torch.tensor([True, False]))
    # The blind environment sees nothing new but still remembers; the other one observes as usual.
    assert bev[0, 3].sum() == 0 and bev[0, 2].sum() > 0 and bev[1, 3].sum() > 0


def test_the_instantaneous_variant_only_knows_the_current_frame(tmp_path):
    policy_bev = pipeline(tmp_path, variant="instantaneous", reliability=None)
    policy_bev.step(*frame(), pose(0.0), no_reset())
    bev = policy_bev.step(*frame(), pose(0.5), no_reset())
    torch.testing.assert_close(bev[0, 2], WEDGE.float())


def test_the_full_variant_needs_a_reliability_map():
    with pytest.raises(ValueError):
        PersistentBev(StubModel(), 1, X_RANGE, Y_RANGE, RESOLUTION, ("ground", "rock"), DT, "full", None, 0.0)


class FakeEnv:
    """The groups of a BEV distillation task, with env 1 resetting on the second step."""

    num_envs, num_actions, device, max_episode_length, cfg = 2, 2, "cpu", 100, {}

    def __init__(self):
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.steps = 0

    def get_observations(self):
        rgb, depth, camera = frame()
        return TensorDict({
            "rgb": rgb, "depth": depth, "camera": camera, "pose": pose(0.1 * self.steps),
            "policy": torch.ones(2, 7), "proprio": torch.ones(2, 3), "bev_target": torch.zeros(2, 3, CELLS, CELLS),
        }, batch_size=[2])

    def step(self, actions):
        self.steps += 1
        dones = torch.tensor([0, int(self.steps == 2)])
        return self.get_observations(), torch.zeros(2), dones, {}


def test_the_wrapper_hands_the_runner_only_the_policy_groups_and_the_bev(tmp_path):
    env = PersistentBevVecEnv(FakeEnv(), pipeline(tmp_path))
    assert set(env.get_observations().keys()) == {"policy", "proprio", "bev"}
    env.step(torch.zeros(2, 2))
    obs, _, dones, _ = env.step(torch.zeros(2, 2))
    assert dones.tolist() == [0, 1] and (obs["bev"][1] == 0).all() and obs["bev"][0, 2].sum() > 0
    # The runner randomizes episode lengths through the wrapper.
    env.episode_length_buf = torch.tensor([5, 7])
    assert env.env.episode_length_buf.tolist() == [5, 7]


def test_the_bev_student_acts_recurrently_and_exports(tmp_path):
    obs = TensorDict({"proprio": torch.randn(3, 3), "bev": torch.randn(3, 5, 120, 120)}, batch_size=[3])
    student = RoverBevStudentModel(
        obs, {"student": ["proprio", "bev"]}, "student", 2, [64, 32], "elu", rnn_hidden_dim=32, rnn_num_layers=2
    )
    first = student(obs)
    assert first.shape == (3, 2) and (first.abs() < 1).all()
    # The hidden state carries over steps until reset.
    assert not torch.allclose(student(obs), first)
    student.reset(torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(student(obs), first)
    exported = student.as_onnx()
    actions, hidden = exported(*exported.get_dummy_inputs())
    assert actions.shape == (1, 2) and hidden.shape == (2, 1, 32)
