"""
verification/fusion/dcri_selection/test_monotonic_penalty.py
Tests for Monotonic Penalty Invariants, Zero-Penalty Identity, and Analytical Derivatives.
"""

import unittest
import sys
from pathlib import Path
import numpy as np

# Ensure root is in path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.fusion.dcri_selection.selection_config import SEED, N_PACKETS
from src.fusion.dcri_selection.candidate_grid import get_candidate_grid
from src.fusion.dcri_selection.selection_runner import DeltaSelectionExperimentRunner


class TestMonotonicPenaltyInvariants(unittest.TestCase):
    """Unit tests validating mathematical penalty invariants across candidate deltas."""

    @classmethod
    def setUpClass(cls):
        cls.runner = DeltaSelectionExperimentRunner(repo_root=root_dir, seed=SEED)
        cls.cohort = cls.runner.load_cohort()
        cls.grid = get_candidate_grid()
        cls.base_states = cls.runner.evaluator.precompute_packet_base_state(cls.cohort)
        cls.evals = [cls.runner.evaluator.evaluate_candidate_delta(g, cls.base_states) for g in cls.grid]

    def test_zero_penalty_identity(self):
        """Asserts DCRI_0 == R_fusion bitwise within 1e-14 across all active packets."""
        eval_0 = self.evals[0]
        self.assertEqual(eval_0["delta"], 0.0)
        for p in eval_0["packet_evaluations"]:
            self.assertAlmostEqual(p["dcri"], p["r_fusion"], delta=1e-14)
            self.assertAlmostEqual(p["penalty"], 0.0, delta=1e-14)

    def test_strict_monotonic_decay_per_packet(self):
        """Asserts that for delta_2 > delta_1, DCRI_delta2 <= DCRI_delta1 on every single packet."""
        for i in range(len(self.grid) - 1):
            eval_a = self.evals[i]
            eval_b = self.evals[i+1]
            self.assertLess(eval_a["delta"], eval_b["delta"])
            
            for pa, pb in zip(eval_a["packet_evaluations"], eval_b["packet_evaluations"]):
                self.assertLessEqual(
                    pb["dcri"],
                    pa["dcri"] + 1e-14,
                    f"Monotonicity violated on packet {pa['packet_id']}: {pb['dcri']} > {pa['dcri']}"
                )

    def test_analytical_derivative_identity(self):
        """Asserts dDCRI/ddelta == -U_sum matches finite step slope within numerical precision."""
        mean_u_sum = self.evals[0]["mean_u_sum"]
        for i in range(len(self.grid) - 1):
            d1 = self.grid[i].delta
            d2 = self.grid[i+1].delta
            m1 = self.evals[i]["mean_dcri"]
            m2 = self.evals[i+1]["mean_dcri"]
            finite_slope = (m2 - m1) / (d2 - d1)
            self.assertAlmostEqual(finite_slope, -mean_u_sum, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
