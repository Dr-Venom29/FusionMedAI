"""
verification/fusion/router_sensitivity/test_coefficient_semantics.py
Validates exact mathematical logit derivative semantics for alpha, beta, gamma, and eta.
"""

import unittest
from pathlib import Path

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router_sensitivity.sensitivity_metrics import verify_logit_derivatives


class TestCoefficientSemantics(unittest.TestCase):
    """Validates that each coefficient uniquely and linearly modulates its respective term."""

    @classmethod
    def setUpClass(cls):
        cls.cohort = load_frozen_cohort(Path(__file__).resolve().parents[3], n_packets=500, seed=115)

    def test_alpha_confidence_logit_derivative(self):
        """Delta(z_i - z_j) = Delta_alpha * (C_i - C_j)."""
        c1 = RouterCoefficients(alpha=0.5, beta=1.5, gamma=1.0, eta=0.5)
        c2 = RouterCoefficients(alpha=1.5, beta=1.5, gamma=1.0, eta=0.5)

        for pkt in self.cohort[:100]:
            res = verify_logit_derivatives(pkt, c1, c2)
            self.assertLess(res["max_error"], 1e-11, "Alpha logit derivative violated")

    def test_beta_reliability_logit_derivative(self):
        """Delta(z_i - z_j) = Delta_beta * (R_i - R_j)."""
        c1 = RouterCoefficients(alpha=1.0, beta=1.0, gamma=1.0, eta=0.5)
        c2 = RouterCoefficients(alpha=1.0, beta=2.0, gamma=1.0, eta=0.5)

        for pkt in self.cohort[:100]:
            res = verify_logit_derivatives(pkt, c1, c2)
            self.assertLess(res["max_error"], 1e-11, "Beta logit derivative violated")

    def test_gamma_uncertainty_logit_derivative(self):
        """Delta(z_i - z_j) = -Delta_gamma * (U_i - U_j)."""
        c1 = RouterCoefficients(alpha=1.0, beta=1.5, gamma=0.5, eta=0.5)
        c2 = RouterCoefficients(alpha=1.0, beta=1.5, gamma=1.5, eta=0.5)

        for pkt in self.cohort[:100]:
            res = verify_logit_derivatives(pkt, c1, c2)
            self.assertLess(res["max_error"], 1e-11, "Gamma logit derivative violated")

    def test_eta_quality_logit_derivative(self):
        """Delta(z_i - z_j) = Delta_eta * (Q_i - Q_j)."""
        c1 = RouterCoefficients(alpha=1.0, beta=1.5, gamma=1.0, eta=0.25)
        c2 = RouterCoefficients(alpha=1.0, beta=1.5, gamma=1.0, eta=0.75)

        for pkt in self.cohort[:100]:
            res = verify_logit_derivatives(pkt, c1, c2)
            self.assertLess(res["max_error"], 1e-11, "Eta logit derivative violated")


if __name__ == "__main__":
    unittest.main()
