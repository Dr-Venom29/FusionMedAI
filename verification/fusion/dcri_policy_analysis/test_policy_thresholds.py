"""
FusionMedAI - Phase C11.14: Unit Tests for Policy Thresholds and Invariants
"""

import unittest
import numpy as np

from src.fusion.dcri_policy.policy_config import (
    ACTION_ROUTINE_REVIEW,
    ACTION_ADDITIONAL_ASSESSMENT,
    ACTION_ESCALATION,
    TAU_1_DEFAULT,
    TAU_2_DEFAULT,
    TAU_1_GRID,
    TAU_2_GRID,
)
from src.fusion.dcri_policy.policy_engine import assign_action


class TestPolicyThresholds(unittest.TestCase):
    """Tests threshold assignment boundaries, negative handling, and grid properties."""

    def test_threshold_equality_boundaries(self):
        """Validates that boundary equality strictly respects [tau_1, tau_2) half-open intervals."""
        tau_1 = 0.20
        tau_2 = 0.40

        # Just below tau_1
        self.assertEqual(assign_action(0.199999, tau_1, tau_2), ACTION_ROUTINE_REVIEW)
        # Exactly on tau_1 -> must be ADDITIONAL_ASSESSMENT
        self.assertEqual(assign_action(0.200000, tau_1, tau_2), ACTION_ADDITIONAL_ASSESSMENT)
        # Just above tau_1
        self.assertEqual(assign_action(0.200001, tau_1, tau_2), ACTION_ADDITIONAL_ASSESSMENT)

        # Just below tau_2
        self.assertEqual(assign_action(0.399999, tau_1, tau_2), ACTION_ADDITIONAL_ASSESSMENT)
        # Exactly on tau_2 -> must be ESCALATION
        self.assertEqual(assign_action(0.400000, tau_1, tau_2), ACTION_ESCALATION)
        # Just above tau_2
        self.assertEqual(assign_action(0.400001, tau_1, tau_2), ACTION_ESCALATION)

    def test_negative_score_handling(self):
        """Validates that negative DCRI scores fall strictly into Routine Review (tier 0)."""
        tau_1 = 0.20
        tau_2 = 0.40

        for neg_val in [-0.001, -0.05, -0.226, -0.50, -1.00, -3.00]:
            self.assertEqual(
                assign_action(neg_val, tau_1, tau_2),
                ACTION_ROUTINE_REVIEW,
                f"Negative score {neg_val} did not assign to Routine Review",
            )

    def test_invalid_inputs_raise(self):
        """Validates that NaN, Inf, non-finite thresholds, or inverted thresholds raise ValueError."""
        # Non-finite scores
        with self.assertRaises(ValueError):
            assign_action(float("nan"), 0.20, 0.40)
        with self.assertRaises(ValueError):
            assign_action(float("inf"), 0.20, 0.40)
        with self.assertRaises(ValueError):
            assign_action(float("-inf"), 0.20, 0.40)

        # Non-finite thresholds
        with self.assertRaises(ValueError):
            assign_action(0.25, float("nan"), 0.40)
        with self.assertRaises(ValueError):
            assign_action(0.25, 0.20, float("nan"))
        with self.assertRaises(ValueError):
            assign_action(0.25, float("inf"), 0.40)
        with self.assertRaises(ValueError):
            assign_action(0.25, 0.20, float("inf"))
        with self.assertRaises(ValueError):
            assign_action(0.25, float("-inf"), 0.40)

        # Inverted or equal thresholds
        with self.assertRaises(ValueError):
            assign_action(0.25, 0.40, 0.20)  # Inverted tau_1 > tau_2
        with self.assertRaises(ValueError):
            assign_action(0.25, 0.30, 0.30)  # Equal tau_1 == tau_2

    def test_grid_ordering_properties(self):
        """Validates that tau_1 and tau_2 grids are strictly monotonic."""
        self.assertTrue(all(TAU_1_GRID[i] < TAU_1_GRID[i+1] for i in range(len(TAU_1_GRID)-1)))
        self.assertTrue(all(TAU_2_GRID[i] < TAU_2_GRID[i+1] for i in range(len(TAU_2_GRID)-1)))


if __name__ == "__main__":
    unittest.main()
