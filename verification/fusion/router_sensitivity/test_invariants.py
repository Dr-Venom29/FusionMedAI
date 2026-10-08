"""
verification/fusion/router_sensitivity/test_invariants.py
Validates active simplex conservation, non-negativity, entropy bounds, and empty modality safety.
"""

import unittest
from pathlib import Path
import math

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.router_input import RouterInput, ModalityChannelInput
from src.fusion.router_sensitivity.parameter_grid import get_sensitivity_parameter_grid
from src.fusion.router_sensitivity.sensitivity_config import REGIMES_TAXONOMY


class TestSensitivityInvariants(unittest.TestCase):
    """Unit tests validating routing invariants across the parameter grid."""

    @classmethod
    def setUpClass(cls):
        cls.cohort = load_frozen_cohort(Path(__file__).resolve().parents[3], n_packets=500, seed=115)
        cls.grid = get_sensitivity_parameter_grid()

    def test_simplex_sum_and_non_negativity_across_all_configs(self):
        """Validates sum(w_i) = 1.0 and w_i >= 0 across all 23 named evaluations / 19 unique coefficient vectors on tri-modal packets."""
        for cfg in self.grid:
            router = ACARAUv2Router(coefficients=cfg.to_coefficients())
            for pkt in self.cohort[:50]:
                r_in = pkt.to_router_input()
                r_out = router.route(r_in)
                w_sum = sum(r_out.weights.values())
                self.assertAlmostEqual(w_sum, 1.0, places=5, msg=f"Simplex violated for {cfg.config_id}")
                for m, w in r_out.weights.items():
                    self.assertGreaterEqual(w, 0.0, f"Negative weight for {m} in {cfg.config_id}")

    def test_all_regimes_simplex_conservation(self):
        """Validates active simplex conservation across all 7 availability regimes."""
        for cfg in self.grid[:5]:  # Sample configurations
            router = ACARAUv2Router(coefficients=cfg.to_coefficients())
            for regime_code, active_mods in REGIMES_TAXONOMY.items():
                pkt = self.cohort[0]
                ch_map = {}
                for m in ["retina", "foot", "clinical"]:
                    rec = pkt.records[m]
                    is_active = (m in active_mods)
                    ch_map[m] = ModalityChannelInput(
                        modality=m,
                        confidence=float(rec.confidence),
                        reliability=float(rec.reliability),
                        uncertainty=float(rec.uncertainty),
                        quality=float(rec.quality) if is_active else 0.0,
                        availability=is_active,
                    )
                r_in = RouterInput(retina=ch_map["retina"], foot=ch_map["foot"], clinical=ch_map["clinical"])
                r_out = router.route(r_in)

                active_sum = sum(r_out.weights[m] for m in active_mods)
                self.assertAlmostEqual(active_sum, 1.0, places=5, msg=f"Regime {regime_code} simplex failed")
                for m in ["retina", "foot", "clinical"]:
                    if m not in active_mods:
                        self.assertEqual(r_out.weights[m], 0.0, f"Inactive {m} non-zero in {regime_code}")

    def test_empty_modality_safe_failure(self):
        """Validates that empty modality set returns NO_MODALITY_AVAILABLE safely across all configs."""
        for cfg in self.grid:
            router = ACARAUv2Router(coefficients=cfg.to_coefficients())
            empty_in = RouterInput(
                retina=ModalityChannelInput("retina", 0.5, 0.929956, 0.5, 0.0, False),
                foot=ModalityChannelInput("foot", 0.5, 0.922266, 0.5, 0.0, False),
                clinical=ModalityChannelInput("clinical", 0.5, 0.825382, 0.5, 0.0, False),
            )
            empty_out = router.route(empty_in)
            self.assertEqual(empty_out.status, "NO_MODALITY_AVAILABLE")
            self.assertEqual(sum(empty_out.weights.values()), 0.0)
            self.assertEqual(empty_out.routing_entropy, 0.0)

    def test_numerical_softmax_stability(self):
        """Validates that extreme logits do not produce NaN or Inf."""
        for cfg in self.grid:
            router = ACARAUv2Router(coefficients=cfg.to_coefficients())
            # Test extreme channel inputs
            extreme_in = RouterInput(
                retina=ModalityChannelInput("retina", 1.0, 0.929956, 0.0, 1.0, True),
                foot=ModalityChannelInput("foot", 0.0, 0.922266, 1.0, 0.0, True),
                clinical=ModalityChannelInput("clinical", 0.5, 0.825382, 0.5, 0.5, True),
            )
            out = router.route(extreme_in)
            for m, w in out.weights.items():
                self.assertFalse(math.isnan(w), f"NaN weight for {m} in {cfg.config_id}")
                self.assertFalse(math.isinf(w), f"Inf weight for {m} in {cfg.config_id}")
            self.assertAlmostEqual(sum(out.weights.values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
