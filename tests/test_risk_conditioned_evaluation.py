from __future__ import annotations

import math
import unittest

import torch

from rover_envs.integrations.clonelab.risk_conditioned_policy import (
    RISK_PREFERENCE_KEY,
    RiskConditionedCloneLabActorPolicy,
)
from rover_envs.integrations.clonelab.risk_evaluation_metrics import (
    RiskEvaluationMetrics,
    clearance_cost,
)


class _DroppingAdapter:
    def to_state(self, state):
        return {"proprioceptive": state["proprioceptive"]}


class _FakePolicy:
    def __init__(self):
        self.device = "cpu"
        self.online_adapter = _DroppingAdapter()
        self.proprioceptive_keys = ("angle_diff", "distance", "heading")
        self.actor = torch.nn.Identity()
        self.last_state = None

    def act(self, state, deterministic=True):
        state = {key: value.to(self.device) for key, value in state.items()}
        state = self.online_adapter.to_state(state)
        self.last_state = state
        return torch.zeros((state["proprioceptive"].shape[0], 2))

    def reset(self, batch_size):
        return None

    def reset_done(self, done):
        return None


class RiskConditionedPolicyTests(unittest.TestCase):
    def test_alpha_is_injected_after_visual_adapter(self):
        base = _FakePolicy()
        policy = RiskConditionedCloneLabActorPolicy(base, risk_preference=0.75)
        policy.act(
            {
                "proprioceptive": torch.zeros((3, 3)),
                "image": torch.zeros((3, 3, 8, 8)),
            }
        )

        self.assertIn(RISK_PREFERENCE_KEY, base.last_state)
        self.assertEqual(tuple(base.last_state[RISK_PREFERENCE_KEY].shape), (3, 1))
        torch.testing.assert_close(
            base.last_state[RISK_PREFERENCE_KEY],
            torch.full((3, 1), 0.75),
        )

    def test_invalid_alpha_is_rejected(self):
        with self.assertRaises(ValueError):
            RiskConditionedCloneLabActorPolicy(_FakePolicy(), risk_preference=1.01)


class RiskEvaluationMetricsTests(unittest.TestCase):
    def test_episode_metrics_and_training_cost_definition(self):
        tracker = RiskEvaluationMetrics(
            num_envs=2,
            step_dt=0.2,
            device="cpu",
            target_episodes=2,
        )
        tracker.start_episodes(
            positions_xy=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            goal_distances=torch.tensor([2.0, 2.0]),
        )

        completed = tracker.update(
            final_positions_xy=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            decision_clearance=torch.tensor([5.0, 1.0]),
            final_clearance=torch.tensor([0.5, 0.8]),
            rewards=torch.tensor([10.0, -1.0]),
            done=torch.tensor([True, False]),
            termination_masks={
                "is_success": torch.tensor([True, False]),
                "collision": torch.tensor([False, False]),
                "time_limit": torch.tensor([False, False]),
            },
        )

        self.assertEqual(len(completed), 1)
        episode = completed[0]
        self.assertTrue(episode.success)
        self.assertFalse(episode.collision)
        self.assertAlmostEqual(episode.minimum_clearance_m, 0.5)
        self.assertAlmostEqual(episode.path_efficiency, 1.0)
        self.assertAlmostEqual(episode.time_to_goal_s, 0.2)
        self.assertAlmostEqual(episode.risk_exposure, 0.01, places=6)
        self.assertAlmostEqual(episode.fraction_steps_below_0p55m, 0.0)
        self.assertAlmostEqual(episode.fraction_steps_below_0p75m, 0.0)

        tracker.start_episodes(
            positions_xy=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            goal_distances=torch.tensor([1.0, 2.0]),
            env_mask=torch.tensor([True, False]),
        )
        tracker.update(
            final_positions_xy=torch.tensor([[0.1, 0.0], [2.0, 0.0]]),
            decision_clearance=torch.tensor([5.0, 0.0]),
            final_clearance=torch.tensor([5.0, 0.0]),
            rewards=torch.tensor([0.0, -5.0]),
            done=torch.tensor([False, True]),
            termination_masks={
                "is_success": torch.tensor([False, False]),
                "collision": torch.tensor([False, True]),
                "time_limit": torch.tensor([False, False]),
            },
        )

        summary = tracker.summary()
        self.assertEqual(summary["completed_episodes"], 2)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertAlmostEqual(summary["collision_rate"], 0.5)
        self.assertAlmostEqual(summary["fraction_steps_below_0p55m"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["fraction_steps_below_0p75m"], 1.0 / 3.0)
        self.assertEqual(summary["termination_counts"], {"collision": 1, "is_success": 1})
        self.assertTrue(tracker.target_reached)

    def test_clearance_cost_endpoints(self):
        values = clearance_cost(torch.tensor([0.0, 5.0]))
        self.assertTrue(math.isclose(float(values[0]), 1.0, rel_tol=0.0, abs_tol=1e-7))
        self.assertTrue(math.isclose(float(values[1]), 0.01, rel_tol=0.0, abs_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
