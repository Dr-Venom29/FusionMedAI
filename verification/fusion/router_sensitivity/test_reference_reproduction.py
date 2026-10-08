"""
verification/fusion/router_sensitivity/test_reference_reproduction.py
Validates that Theta_0 reproduces the sealed reference router metrics identically.
"""

import unittest
from pathlib import Path

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router_sensitivity.sensitivity_metrics import compute_packet_metrics, aggregate_cohort_metrics


class TestReferenceReproduction(unittest.TestCase):
    """Validates numerical reference reproduction within tolerance of the frozen reference configuration."""

    @classmethod
    def setUpClass(cls):
        cls.cohort = load_frozen_cohort(Path(__file__).resolve().parents[3], n_packets=500, seed=115)

    def test_frozen_reference_reproduction(self):
        """Verifies mean weights and entropy match sealed reference constants within numerical tolerance."""
        ref_coeff = RouterCoefficients(alpha=1.0, beta=1.5, gamma=1.0, eta=0.5)
        router = ACARAUv2Router(coefficients=ref_coeff)

        packet_metrics = []
        for pkt in self.cohort:
            r_in = pkt.to_router_input()
            r_out = router.route(r_in)
            m = compute_packet_metrics(pkt, r_out, delta=0.20)
            packet_metrics.append(m)

        summary = aggregate_cohort_metrics(packet_metrics)

        # Expected reference values
        self.assertAlmostEqual(summary["weights"]["retina"]["mean"], 0.501002, places=4)
        self.assertAlmostEqual(summary["weights"]["foot"]["mean"], 0.260925, places=4)
        self.assertAlmostEqual(summary["weights"]["clinical"]["mean"], 0.238073, places=4)
        self.assertAlmostEqual(summary["routing_entropy"]["mean"], 1.011041, places=4)
        self.assertAlmostEqual(summary["r_fusion"]["mean"], 0.289900, places=4)
        self.assertAlmostEqual(summary["dcri"]["mean"], 0.163113, places=4)


if __name__ == "__main__":
    unittest.main()
