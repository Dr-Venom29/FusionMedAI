"""
verification/fusion/router_sensitivity/test_parameter_grid.py
Tests for Phase C11.12 Sensitivity Parameter Grid Structure, Bounds, and Pre-Registration.
"""

import unittest
from pathlib import Path

from src.fusion.router_sensitivity.parameter_grid import (
    get_sensitivity_parameter_grid,
    get_reference_config,
    SensitivityConfigItem,
    ParameterGridError,
)
from src.fusion.router_sensitivity.sensitivity_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    COEFFICIENT_BOUNDS,
)


class TestSensitivityParameterGrid(unittest.TestCase):
    """Unit tests validating parameter grid integrity and protocol boundaries."""

    def setUp(self):
        self.grid = get_sensitivity_parameter_grid()
        self.ref = get_reference_config()

    def test_grid_length_and_types(self):
        """Validates that grid contains all 23 named evaluations and 19 unique parameter vectors."""
        self.assertEqual(len(self.grid), 23, "Grid must contain exactly 23 named evaluations")
        unique_tuples = {(item.alpha, item.beta, item.gamma, item.eta) for item in self.grid}
        self.assertEqual(len(unique_tuples), 19, "Grid must span exactly 19 unique parameter tuples")
        config_ids = [item.config_id for item in self.grid]
        self.assertEqual(len(config_ids), len(set(config_ids)), "All config_ids in grid must be unique")

    def test_reference_configuration_lock(self):
        """Validates that reference configuration Theta_0 matches frozen constants."""
        self.assertEqual(self.ref.alpha, ALPHA_REF)
        self.assertEqual(self.ref.beta, BETA_REF)
        self.assertEqual(self.ref.gamma, GAMMA_REF)
        self.assertEqual(self.ref.eta, ETA_REF)
        self.assertTrue(self.ref.is_reference)

    def test_all_coefficients_within_bounds(self):
        """Validates that every configuration in the grid satisfies [0.0, 5.0] bounds."""
        for item in self.grid:
            for param in ["alpha", "beta", "gamma", "eta"]:
                val = getattr(item, param)
                min_b, max_b = COEFFICIENT_BOUNDS[param]
                self.assertGreaterEqual(val, min_b, f"Config {item.config_id} {param} below min bound")
                self.assertLessEqual(val, max_b, f"Config {item.config_id} {param} above max bound")

    def test_sweep_categorization(self):
        """Validates that sweeps are partitioned into alpha, beta, gamma, eta, and combined."""
        sweeps = [item.sweep_type for item in self.grid]
        self.assertIn("alpha_sweep", sweeps)
        self.assertIn("beta_sweep", sweeps)
        self.assertIn("gamma_sweep", sweeps)
        self.assertIn("eta_sweep", sweeps)
        self.assertIn("combined_sweep", sweeps)

    def test_invalid_parameter_rejection(self):
        """Validates that out-of-bounds or NaN coefficients raise ParameterGridError."""
        with self.assertRaises(ParameterGridError):
            SensitivityConfigItem("BAD", "Bad", -1.0, 1.5, 1.0, 0.5, "alpha_sweep", "Negative")
        with self.assertRaises(ParameterGridError):
            SensitivityConfigItem("BAD", "Bad", 1.0, 10.0, 1.0, 0.5, "beta_sweep", "Too high")
        with self.assertRaises(ParameterGridError):
            SensitivityConfigItem("BAD", "Bad", 1.0, 1.5, float("nan"), 0.5, "gamma_sweep", "NaN")


if __name__ == "__main__":
    unittest.main()
