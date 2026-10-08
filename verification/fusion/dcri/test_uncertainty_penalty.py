"""
Unit tests for uncertainty burden and penalty computation (Phase C11.6).
"""

import pytest
import math
from src.fusion.dcri.uncertainty_penalty import (
    compute_uncertainty_burden,
    compute_uncertainty_penalty_contributions,
    compute_uncertainty_penalty,
)


def test_uncertainty_burden_sum_and_mean():
    unc = {"retina": 0.10, "foot": 0.20, "clinical": 0.15}
    active = ["retina", "foot", "clinical"]

    u_sum, u_mean = compute_uncertainty_burden(unc, active)
    assert abs(u_sum - 0.45) < 1e-6
    assert abs(u_mean - 0.15) < 1e-6


def test_uncertainty_burden_growth_with_modality_count():
    """Exposes that U_sum scales with |A| whereas U_mean stays constant under equal uncertainties."""
    unc = {"retina": 0.40, "foot": 0.40, "clinical": 0.40}

    u_sum_1, u_mean_1 = compute_uncertainty_burden(unc, ["retina"])
    assert abs(u_sum_1 - 0.40) < 1e-6
    assert abs(u_mean_1 - 0.40) < 1e-6

    u_sum_3, u_mean_3 = compute_uncertainty_burden(unc, ["retina", "foot", "clinical"])
    assert abs(u_sum_3 - 1.20) < 1e-6
    assert abs(u_mean_3 - 0.40) < 1e-6
    assert abs(u_sum_3 - 3.0 * u_sum_1) < 1e-6


def test_uncertainty_penalty_contributions_conservation():
    unc = {"retina": 0.10, "foot": 0.20, "clinical": 0.15}
    active = ["retina", "foot", "clinical"]
    delta = 0.20

    contributions = compute_uncertainty_penalty_contributions(unc, delta, active)
    assert abs(contributions["retina"] - 0.02) < 1e-6
    assert abs(contributions["foot"] - 0.04) < 1e-6
    assert abs(contributions["clinical"] - 0.03) < 1e-6

    u_sum, _ = compute_uncertainty_burden(unc, active)
    total_penalty = compute_uncertainty_penalty(u_sum, delta)
    assert abs(total_penalty - 0.09) < 1e-6
    assert abs(sum(contributions.values()) - total_penalty) < 1e-6


def test_zero_uncertainty_gives_zero_penalty():
    unc = {"retina": 0.0, "foot": 0.0, "clinical": 0.0}
    active = ["retina", "foot", "clinical"]
    u_sum, _ = compute_uncertainty_burden(unc, active)
    for d in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        assert compute_uncertainty_penalty(u_sum, d) == 0.0


def test_penalty_scaling_with_delta():
    u_sum = 0.80
    assert abs(compute_uncertainty_penalty(u_sum, 0.0) - 0.0) < 1e-6
    assert abs(compute_uncertainty_penalty(u_sum, 0.05) - 0.04) < 1e-6
    assert abs(compute_uncertainty_penalty(u_sum, 0.10) - 0.08) < 1e-6
    assert abs(compute_uncertainty_penalty(u_sum, 0.20) - 0.16) < 1e-6
    assert abs(compute_uncertainty_penalty(u_sum, 0.50) - 0.40) < 1e-6
    assert abs(compute_uncertainty_penalty(u_sum, 1.00) - 0.80) < 1e-6
