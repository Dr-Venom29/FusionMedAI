"""
verification/fusion/router_sensitivity/test_mask_invariance.py
Validates strong masked-value invariance across all 3 channels (Retina, Foot, Clinical)
for all parameter grid configurations.
"""

import unittest
from pathlib import Path

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.router_input import RouterInput, ModalityChannelInput
from src.fusion.router_sensitivity.parameter_grid import get_sensitivity_parameter_grid


class TestMaskInvariance(unittest.TestCase):
    """Unit tests validating zero cross-talk from unavailable channels across all configs and modalities."""

    @classmethod
    def setUpClass(cls):
        cls.cohort = load_frozen_cohort(Path(__file__).resolve().parents[3], n_packets=500, seed=115)
        cls.grid = get_sensitivity_parameter_grid()

    def test_masked_value_invariance_all_modalities_and_configs(self):
        """Validates that varying inactive channel values on R, F, or C produces exactly zero change in active weights."""
        test_packets = self.cohort[:15]

        for cfg in self.grid:
            router = ACARAUv2Router(coefficients=cfg.to_coefficients())

            for pkt in test_packets:
                ch_r = pkt.records["retina"].to_channel_input()
                ch_f = pkt.records["foot"].to_channel_input()
                ch_c = pkt.records["clinical"].to_channel_input()

                # Test 1: Clinical unavailable (A_C = 0)
                ch_c_base = ModalityChannelInput("clinical", 0.5, 0.825382, 0.5, 0.0, False)
                out_c_base = router.route(RouterInput(ch_r, ch_f, ch_c_base))

                for test_conf in [0.0, 0.25, 0.75, 1.0]:
                    for test_unc in [0.0, 0.5, 1.0]:
                        ch_c_pert = ModalityChannelInput("clinical", test_conf, 0.825382, test_unc, 0.0, False)
                        out_c_pert = router.route(RouterInput(ch_r, ch_f, ch_c_pert))
                        self.assertAlmostEqual(out_c_pert.weights["retina"], out_c_base.weights["retina"], places=10)
                        self.assertAlmostEqual(out_c_pert.weights["foot"], out_c_base.weights["foot"], places=10)
                        self.assertEqual(out_c_pert.weights["clinical"], 0.0)

                # Test 2: Foot unavailable (A_F = 0)
                ch_f_base = ModalityChannelInput("foot", 0.5, 0.922266, 0.5, 0.0, False)
                out_f_base = router.route(RouterInput(ch_r, ch_f_base, ch_c))

                for test_conf in [0.0, 0.5, 1.0]:
                    for test_unc in [0.0, 0.5, 1.0]:
                        ch_f_pert = ModalityChannelInput("foot", test_conf, 0.922266, test_unc, 0.0, False)
                        out_f_pert = router.route(RouterInput(ch_r, ch_f_pert, ch_c))
                        self.assertAlmostEqual(out_f_pert.weights["retina"], out_f_base.weights["retina"], places=10)
                        self.assertAlmostEqual(out_f_pert.weights["clinical"], out_f_base.weights["clinical"], places=10)
                        self.assertEqual(out_f_pert.weights["foot"], 0.0)

                # Test 3: Retina unavailable (A_R = 0)
                ch_r_base = ModalityChannelInput("retina", 0.5, 0.929956, 0.5, 0.0, False)
                out_r_base = router.route(RouterInput(ch_r_base, ch_f, ch_c))

                for test_conf in [0.0, 0.5, 1.0]:
                    for test_unc in [0.0, 0.5, 1.0]:
                        ch_r_pert = ModalityChannelInput("retina", test_conf, 0.929956, test_unc, 0.0, False)
                        out_r_pert = router.route(RouterInput(ch_r_pert, ch_f, ch_c))
                        self.assertAlmostEqual(out_r_pert.weights["foot"], out_r_base.weights["foot"], places=10)
                        self.assertAlmostEqual(out_r_pert.weights["clinical"], out_r_base.weights["clinical"], places=10)
                        self.assertEqual(out_r_pert.weights["retina"], 0.0)


if __name__ == "__main__":
    unittest.main()
