"""
FusionMedAI - Phase C11.14: Unit Tests for Regime Policy Invariance and Availability
"""

import unittest
from pathlib import Path

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.dcri_policy.policy_config import (
    ACTIVE_REGIMES,
    DELTA_FROZEN,
    TAU_1_DEFAULT,
    TAU_2_DEFAULT,
)
from src.fusion.dcri_policy.regime_policy_evaluator import RegimePolicyEvaluator


class TestRegimePolicyInvariance(unittest.TestCase):
    """Tests regime availability masking, simplex conservation, and EMPTY fail-closed."""

    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[3]
        cls.cohort = load_frozen_cohort(repo_root=repo_root, n_packets=500, seed=115)
        cls.evaluator = RegimePolicyEvaluator(
            delta=DELTA_FROZEN,
            tau_1=TAU_1_DEFAULT,
            tau_2=TAU_2_DEFAULT,
        )
        cls.regime_results = cls.evaluator.evaluate_regimes(cls.cohort)

    def test_all_seven_active_regimes_present(self):
        """Validates that all 7 active regimes are evaluated and present."""
        for reg in ACTIVE_REGIMES:
            self.assertIn(reg, self.regime_results)
            self.assertEqual(self.regime_results[reg]["n_packets"], 500)

    def test_empty_regime_fail_closed_contract(self):
        """Validates that EMPTY regime fails closed with NO_MODALITY_AVAILABLE."""
        self.assertIn("EMPTY", self.regime_results)
        empty = self.regime_results["EMPTY"]
        self.assertEqual(empty["fail_closed_status"], "NO_MODALITY_AVAILABLE")
        self.assertFalse(empty["decision_output_available"])
        self.assertEqual(empty["sentinel_risk"], 0.0)
        self.assertTrue(all(w == 0.0 for w in empty["routing_weights"].values()))

    def test_regime_non_negative_reclassification(self):
        """Validates that across all regimes, no packet is upgraded (downgraded_count == total_count)."""
        for reg in ACTIVE_REGIMES:
            res = self.regime_results[reg]["reclassifications"]
            self.assertEqual(
                res["total_count"],
                res["downgraded_count"],
                f"Regime {reg} had non-downgrade reclassifications",
            )

    def test_regime_simplex_and_percentages(self):
        """Validates that tier percentages sum to 100.0%."""
        for reg in ACTIVE_REGIMES:
            r = self.regime_results[reg]
            pct_a = sum(r["policy_a_dcri"]["percentages"].values())
            self.assertAlmostEqual(pct_a, 100.0, places=3, msg=f"Regime {reg} policy A percentages do not sum to 100")

            pct_b = sum(r["policy_b_fused_risk"]["percentages"].values())
            self.assertAlmostEqual(pct_b, 100.0, places=3, msg=f"Regime {reg} policy B percentages do not sum to 100")


if __name__ == "__main__":
    unittest.main()
