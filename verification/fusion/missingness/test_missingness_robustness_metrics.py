"""
verification/fusion/missingness/test_robustness_metrics.py
Unit tests for mathematical robustness metrics, redistribution, and DCRI decomposition.
"""

import pytest
import math

from src.fusion.missingness.robustness_metrics import (
    calc_stats,
    compute_risk_delta,
    compute_dcri_delta,
    compute_authority_redistribution,
    compute_dcri_decomposition,
    compute_routing_entropy,
    compute_max_authority,
    compute_bootstrap_ci,
)


def test_compute_risk_and_dcri_delta():
    assert compute_risk_delta(0.60, 0.45) == pytest.approx(0.15, abs=1e-6)
    assert compute_dcri_delta(0.40, 0.20) == pytest.approx(0.20, abs=1e-6)


def test_compute_authority_redistribution():
    w_full = {"retina": 0.50, "foot": 0.30, "clinical": 0.20}
    w_subset = {"retina": 0.65, "foot": 0.35, "clinical": 0.0}
    delta_w = compute_authority_redistribution(w_full, w_subset)
    assert delta_w["retina"] == pytest.approx(0.15, abs=1e-6)
    assert delta_w["foot"] == pytest.approx(0.05, abs=1e-6)
    assert delta_w["clinical"] == pytest.approx(-0.20, abs=1e-6)


def test_compute_dcri_decomposition_exactness():
    r_full = 0.55
    r_sub = 0.40
    u_full = 0.60
    u_sub = 0.30
    delta = 0.20

    decomp = compute_dcri_decomposition(r_full, r_sub, u_full, u_sub, delta=delta)
    # Delta R_signed = 0.55 - 0.40 = 0.15
    # Delta U_sum = 0.60 - 0.30 = 0.30
    # Delta P_U = 0.20 * 0.30 = 0.06
    # Delta DCRI_signed = 0.15 - 0.06 = 0.09
    assert decomp["delta_r_signed"] == pytest.approx(0.15, abs=1e-6)
    assert decomp["delta_u_sum"] == pytest.approx(0.30, abs=1e-6)
    assert decomp["delta_penalty"] == pytest.approx(0.06, abs=1e-6)
    assert decomp["delta_dcri_signed"] == pytest.approx(0.09, abs=1e-6)


def test_compute_routing_entropy():
    # Uniform on 3 channels: ln(3) approx 1.098612
    uniform_3 = {"retina": 1/3, "foot": 1/3, "clinical": 1/3}
    assert compute_routing_entropy(uniform_3) == pytest.approx(math.log(3), abs=1e-5)

    # Degenerate single channel: H = 0.0
    unimodal = {"retina": 1.0, "foot": 0.0, "clinical": 0.0}
    assert compute_routing_entropy(unimodal) == pytest.approx(0.0, abs=1e-6)


def test_compute_bootstrap_ci():
    vals = [0.10, 0.12, 0.11, 0.15, 0.14, 0.13, 0.09, 0.16, 0.12, 0.13]
    ci = compute_bootstrap_ci(vals, n_resamples=1000, seed=115)
    assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]
