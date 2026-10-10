"""
FusionMedAI - Phase C11.15: Test Suite for Router Mathematical Invariants
Validates S15-06, S15-07, S15-08, S15-09, S15-10.
"""

import unittest
from src.fusion.router_sanity.invariant_checker import RouterInvariantChecker


class TestRouterInvariants(unittest.TestCase):
    """Unit test suite for foundational mathematical invariants."""

    @classmethod
    def setUpClass(cls):
        cls.checker = RouterInvariantChecker()

    def test_s15_06_simplex_conservation(self):
        """S15-06: Sum of weights must equal 1.0 within 1e-12 and all weights non-negative."""
        res = self.checker.check_simplex_conservation(n_trials=100)
        self.assertTrue(res["passed"], f"Simplex check failed: {res}")
        self.assertLess(res["max_sum_deviation"], 1e-10)
        self.assertGreaterEqual(res["min_weight_observed"], 0.0)

    def test_s15_07_hard_availability_masking(self):
        """S15-07: Inactive modalities must strictly receive w_i == 0.0 exact."""
        res = self.checker.check_hard_masking(n_trials=50)
        self.assertTrue(res["passed"], f"Hard masking check failed: {res}")
        self.assertEqual(res["violation_count"], 0)

    def test_s15_08_softmax_shift_invariance(self):
        """S15-08: Softmax shift invariance softmax(z + c) == softmax(z)."""
        res = self.checker.check_shift_invariance()
        self.assertTrue(res["passed"], f"Shift invariance check failed: {res}")
        self.assertTrue(res["production_route_shift_passed"])
        self.assertTrue(res["reference_kernel_shift_passed"])
        self.assertLess(res["max_shift_deviation"], 1e-12)

    def test_s15_09_kernel_permutation_invariance(self):
        """S15-09: Reference normalization kernel order independence."""
        res = self.checker.check_kernel_permutation_invariance()
        self.assertTrue(res["passed"], f"Kernel permutation invariance check failed: {res}")
        self.assertTrue(res["reference_kernel_perm_passed"])
        self.assertLess(res["max_permutation_deviation"], 1e-12)

    def test_s15_10_numerical_stability(self):
        """S15-10: Finite outputs and stable normalization under extreme logit differentials."""
        res = self.checker.check_numerical_stability()
        self.assertTrue(res["passed"], f"Numerical stability check failed: {res}")
        self.assertTrue(res["production_route_boundary_passed"])
        self.assertTrue(res["reference_kernel_extreme_logits_passed"])

    def test_determinism(self):
        """Verifies exact deterministic reproducibility across repeated calls."""
        res = self.checker.check_determinism(n_repeats=25)
        self.assertTrue(res["passed"])


if __name__ == "__main__":
    unittest.main()
