"""
verification/fusion/dcri_selection/test_regime_invariance.py
Tests for Modality Availability Regimes (R, F, C, RF, RC, FC, RFC) and Fail-Closed EMPTY.
"""

import unittest
import sys
from pathlib import Path

# Ensure root is in path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.fusion.dcri_selection.selection_config import SEED, ACTIVE_REGIMES, ALL_REGIMES
from src.fusion.dcri_selection.candidate_grid import get_candidate_grid
from src.fusion.dcri_selection.regime_evaluator import RegimeEvaluator
from src.fusion.dcri_selection.selection_runner import DeltaSelectionExperimentRunner


class TestRegimeInvariance(unittest.TestCase):
    """Unit tests validating regime stratification and fail-closed safety."""

    @classmethod
    def setUpClass(cls):
        cls.runner = DeltaSelectionExperimentRunner(repo_root=root_dir, seed=SEED)
        cls.cohort = cls.runner.load_cohort()
        cls.grid = get_candidate_grid()
        cls.base_states = cls.runner.evaluator.precompute_packet_base_state(cls.cohort)
        cls.evals = [cls.runner.evaluator.evaluate_candidate_delta(g, cls.base_states) for g in cls.grid]
        cls.reg_eval = RegimeEvaluator()
        cls.reg_res = cls.reg_eval.evaluate_regimes(cls.evals, cls.cohort)

    def test_all_regimes_represented(self):
        """Validates that all active regimes are evaluated."""
        counts = self.reg_res["regime_counts"]
        for reg in ACTIVE_REGIMES:
            self.assertIn(reg, counts)
            self.assertGreater(counts[reg], 0, f"Regime {reg} must have non-zero packets")

    def test_empty_regime_fail_closed(self):
        """Validates that EMPTY regime is safely handled without raising or corrupting metrics."""
        empty_data = self.reg_res["by_candidate"]["D00"]["regimes"]["EMPTY"]
        self.assertTrue(empty_data["fail_closed_safe"])
        self.assertEqual(empty_data["cardinality"], 0)
        self.assertEqual(empty_data["router_status"], "NO_MODALITY_AVAILABLE")

    def test_router_empty_fail_closed_contract(self):
        """Directly exercises ACARAUv2Router on all-unavailable input and validates fail-closed contract."""
        from src.fusion.router.router_input import RouterInput, ModalityChannelInput
        from src.fusion.reliability.global_reliability import (
            FROZEN_RETINA_RELIABILITY,
            FROZEN_FOOT_RELIABILITY,
            FROZEN_CLINICAL_RELIABILITY,
        )
        empty_in = RouterInput(
            retina=ModalityChannelInput("retina", 0.0, FROZEN_RETINA_RELIABILITY, 1.0, 0.0, False),
            foot=ModalityChannelInput("foot", 0.0, FROZEN_FOOT_RELIABILITY, 1.0, 0.0, False),
            clinical=ModalityChannelInput("clinical", 0.0, FROZEN_CLINICAL_RELIABILITY, 1.0, 0.0, False),
        )
        empty_out = self.runner.evaluator.router.route(empty_in)
        self.assertEqual(empty_out.status, "NO_MODALITY_AVAILABLE")
        self.assertEqual(empty_out.num_active, 0)
        self.assertEqual(len(empty_out.active_modalities), 0)
        self.assertTrue(all(w == 0.0 for w in empty_out.weights.values()))
        self.assertEqual(empty_out.routing_entropy, 0.0)


if __name__ == "__main__":
    unittest.main()
