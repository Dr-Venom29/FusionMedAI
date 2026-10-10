"""
FusionMedAI - Phase C11.15: Test Suite for Isolated Input Monotonicity
Validates S15-02, S15-03, S15-04, S15-05.
"""

import unittest
from src.fusion.router_sanity.isolated_monotonicity import IsolatedMonotonicityEvaluator


class TestIsolatedMonotonicity(unittest.TestCase):
    """Unit test suite for single-input response directionality."""

    @classmethod
    def setUpClass(cls):
        cls.evaluator = IsolatedMonotonicityEvaluator()

    def test_s15_02_confidence_monotonicity(self):
        """S15-02: Increasing confidence C_i must strictly increase w_i in active multi-channel regimes."""
        res = self.evaluator.evaluate_confidence_monotonicity()
        self.assertEqual(res["status"], "PASS", f"Confidence monotonicity failed: {res}")
        self.assertEqual(res["passed_trials"], res["total_trials"])
        self.assertGreater(res["total_trials"], 0)

    def test_s15_03_reliability_monotonicity_fixture(self):
        """S15-03: Increasing reliability R_i in mathematical kernel fixture must strictly increase w_i."""
        res = self.evaluator.evaluate_reliability_monotonicity_fixture()
        self.assertEqual(res["status"], "PASS", f"Reliability monotonicity failed: {res}")
        self.assertEqual(res["passed_trials"], res["total_trials"])
        self.assertGreater(res["total_trials"], 0)

    def test_s15_04_uncertainty_monotonicity(self):
        """S15-04: Increasing uncertainty U_i must strictly decrease w_i in active multi-channel regimes."""
        res = self.evaluator.evaluate_uncertainty_monotonicity()
        self.assertEqual(res["status"], "PASS", f"Uncertainty monotonicity failed: {res}")
        self.assertEqual(res["passed_trials"], res["total_trials"])
        self.assertGreater(res["total_trials"], 0)

    def test_s15_05_quality_monotonicity(self):
        """S15-05: Increasing quality Q_i must strictly increase w_i in active multi-channel regimes."""
        res = self.evaluator.evaluate_quality_monotonicity()
        self.assertEqual(res["status"], "PASS", f"Quality monotonicity failed: {res}")
        self.assertEqual(res["passed_trials"], res["total_trials"])
        self.assertGreater(res["total_trials"], 0)

    def test_single_modality_confidence_invariant_unity(self):
        """In single-modality regimes (R, F, C), confidence changes maintain w_i = 1.0."""
        res = self.evaluator.evaluate_confidence_monotonicity()
        single_trials = [t for t in res["trials"] if t["cardinality"] == 1]
        for t in single_trials:
            self.assertAlmostEqual(t["weight_after"], 1.0, places=7)
            self.assertAlmostEqual(t["delta_weight"], 0.0, places=7)

    def test_single_modality_reliability_invariant_unity(self):
        """In single-modality regimes (R, F, C), reliability kernel changes maintain w_i = 1.0."""
        res = self.evaluator.evaluate_reliability_monotonicity_fixture()
        single_trials = [t for t in res["trials"] if t["cardinality"] == 1]
        for t in single_trials:
            self.assertAlmostEqual(t["weight_after"], 1.0, places=7)
            self.assertAlmostEqual(t["delta_weight"], 0.0, places=7)

    def test_single_modality_uncertainty_invariant_unity(self):
        """In single-modality regimes (R, F, C), uncertainty changes maintain w_i = 1.0."""
        res = self.evaluator.evaluate_uncertainty_monotonicity()
        single_trials = [t for t in res["trials"] if t["cardinality"] == 1]
        for t in single_trials:
            self.assertAlmostEqual(t["weight_after"], 1.0, places=7)
            self.assertAlmostEqual(t["delta_weight"], 0.0, places=7)

    def test_single_modality_quality_invariant_unity(self):
        """In single-modality regimes (R, F, C), quality changes maintain w_i = 1.0."""
        res = self.evaluator.evaluate_quality_monotonicity()
        single_trials = [t for t in res["trials"] if t["cardinality"] == 1]
        for t in single_trials:
            self.assertAlmostEqual(t["weight_after"], 1.0, places=7)
            self.assertAlmostEqual(t["delta_weight"], 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
