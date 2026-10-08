"""
Unit tests for weighted risk aggregation and conservation (Phase C11.6).
"""

import pytest
import math
from src.fusion.dcri.aggregation import (
    compute_weighted_risk_contributions,
    compute_r_fusion,
)


def test_weighted_risk_contributions_trimodal():
    weights = {"retina": 0.60, "foot": 0.25, "clinical": 0.15}
    risks = {"retina": 0.70, "foot": 0.40, "clinical": 0.80}
    active = ["retina", "foot", "clinical"]

    k = compute_weighted_risk_contributions(weights, risks, active)
    assert abs(k["retina"] - 0.42) < 1e-6
    assert abs(k["foot"] - 0.10) < 1e-6
    assert abs(k["clinical"] - 0.12) < 1e-6


def test_weighted_risk_contributions_single_modality():
    weights = {"retina": 1.0, "foot": 0.0, "clinical": 0.0}
    risks = {"retina": 0.65, "foot": 0.30, "clinical": 0.20}
    active = ["retina"]

    k = compute_weighted_risk_contributions(weights, risks, active)
    assert abs(k["retina"] - 0.65) < 1e-6
    assert k["foot"] == 0.0
    assert k["clinical"] == 0.0


def test_contribution_conservation_sum_k_equals_r_fusion():
    weights = {"retina": 0.50, "foot": 0.30, "clinical": 0.20}
    risks = {"retina": 0.80, "foot": 0.50, "clinical": 0.10}
    active = ["retina", "foot", "clinical"]

    k = compute_weighted_risk_contributions(weights, risks, active)
    r_fusion = compute_r_fusion(k, active)

    expected_r = 0.50 * 0.80 + 0.30 * 0.50 + 0.20 * 0.10  # 0.40 + 0.15 + 0.02 = 0.57
    assert abs(r_fusion - expected_r) < 1e-6
    assert abs(sum(k.values()) - r_fusion) < 1e-6
    assert 0.0 <= r_fusion <= 1.0


def test_zero_modality_r_fusion_returns_zero():
    k = {"retina": 0.0, "foot": 0.0, "clinical": 0.0}
    active = []
    r_fusion = compute_r_fusion(k, active)
    assert r_fusion == 0.0


def test_r_fusion_rejects_nan_and_inf():
    weights = {"retina": float("nan"), "foot": 0.5, "clinical": 0.5}
    risks = {"retina": 0.5, "foot": 0.5, "clinical": 0.5}
    with pytest.raises(ValueError):
        compute_weighted_risk_contributions(weights, risks, ["retina", "foot", "clinical"])
