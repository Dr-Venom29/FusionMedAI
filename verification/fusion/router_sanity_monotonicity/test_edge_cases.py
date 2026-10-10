"""
FusionMedAI - Phase C11.15: Test Suite for Degenerate & Edge Cases
Validates S15-11, S15-12.
"""

import unittest
from src.fusion.router_sanity.edge_case_evaluator import EdgeCaseEvaluator
from src.fusion.router.router_input import ModalityChannelInput, RouterContractValidationError, FROZEN_RELIABILITY_MAP
from src.fusion.router.coefficients import RouterCoefficients, RouterCoefficientError


class TestEdgeCases(unittest.TestCase):
    """Unit test suite for degenerate regimes and error boundaries."""

    @classmethod
    def setUpClass(cls):
        cls.evaluator = EdgeCaseEvaluator()

    def test_s15_11_empty_modalities_fail_closed(self):
        """S15-11: When all modalities are inactive, router must return NO_MODALITY_AVAILABLE."""
        res = self.evaluator.check_empty_modalities_fail_closed()
        self.assertTrue(res["passed"], f"Empty modalities handling failed: {res}")
        self.assertEqual(res["status"], "NO_MODALITY_AVAILABLE")
        self.assertEqual(res["num_active"], 0)

    def test_s15_12_masked_value_corruption_invariance(self):
        """S15-12: Perturbing inactive channel values must not alter active modality weights."""
        res = self.evaluator.check_masked_value_corruption_invariance()
        self.assertTrue(res["passed"], f"Masked corruption invariance failed: {res}")
        self.assertLess(res["max_active_weight_dev"], 1e-12)
        self.assertEqual(res["violation_count"], 0)

    def test_input_validation_aggregate(self):
        """Invalid inputs (NaN, Inf, out of range, invalid coefficients) must be rejected by contract validators."""
        res = self.evaluator.check_input_validation_rejection()
        self.assertTrue(res["all_invalid_inputs_rejected"], f"Validation rejection failed: {res}")
        self.assertGreaterEqual(res["total_cases"], 15)

    def test_individual_confidence_rejections(self):
        """Individual assertion: NaN, Inf, and out-of-bounds confidence values must raise errors."""
        with self.assertRaises((ValueError, RouterContractValidationError)):
            ModalityChannelInput(modality="retina", confidence=float("nan"), uncertainty=0.1, quality=0.8, availability=True, reliability=FROZEN_RELIABILITY_MAP["retina"])
        with self.assertRaises((ValueError, RouterContractValidationError)):
            ModalityChannelInput(modality="retina", confidence=1.5, uncertainty=0.1, quality=0.8, availability=True, reliability=FROZEN_RELIABILITY_MAP["retina"])
        with self.assertRaises((ValueError, RouterContractValidationError)):
            ModalityChannelInput(modality="retina", confidence=-0.1, uncertainty=0.1, quality=0.8, availability=True, reliability=FROZEN_RELIABILITY_MAP["retina"])

    def test_individual_quality_rejections(self):
        """Individual assertion: Quality on inactive channel or out-of-bounds quality must raise errors."""
        with self.assertRaises((ValueError, RouterContractValidationError)):
            ModalityChannelInput(modality="clinical", confidence=0.5, uncertainty=0.1, quality=0.8, availability=False, reliability=FROZEN_RELIABILITY_MAP["clinical"])
        with self.assertRaises((ValueError, RouterContractValidationError)):
            ModalityChannelInput(modality="clinical", confidence=0.5, uncertainty=0.1, quality=1.2, availability=True, reliability=FROZEN_RELIABILITY_MAP["clinical"])

    def test_individual_reliability_rejections(self):
        """Individual assertion: Mismatched or non-finite reliability must raise errors."""
        with self.assertRaises((ValueError, RouterContractValidationError)):
            ModalityChannelInput(modality="retina", confidence=0.5, uncertainty=0.1, quality=0.8, availability=True, reliability=0.50)
        with self.assertRaises((ValueError, RouterContractValidationError)):
            ModalityChannelInput(modality="retina", confidence=0.5, uncertainty=0.1, quality=0.8, availability=True, reliability=float("nan"))

    def test_individual_coefficient_rejections(self):
        """Individual assertion: Non-finite or out-of-bounds router coefficients must raise errors."""
        with self.assertRaises((ValueError, RouterCoefficientError)):
            RouterCoefficients(alpha=float("nan"), beta=1.5, gamma=1.0, eta=0.5)
        with self.assertRaises((ValueError, RouterCoefficientError)):
            RouterCoefficients(alpha=1.0, beta=1.5, gamma=-1.0, eta=0.5)
        with self.assertRaises((ValueError, RouterCoefficientError)):
            RouterCoefficients(alpha=1.0, beta=1.5, gamma=1.0, eta=10.0)


if __name__ == "__main__":
    unittest.main()
