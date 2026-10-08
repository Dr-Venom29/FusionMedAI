"""
verification/fusion/conflict/test_conflict_metrics.py
Unit tests for continuous conflict metrics (max/mean disagreement, weighted variance/std, entropy).
"""

import pytest
import math
from src.fusion.conflict.conflict_metrics import (
    compute_max_disagreement,
    compute_mean_disagreement,
    compute_weighted_dispersion,
    compute_weight_entropy,
)
from src.fusion.conflict.pairwise_disagreement import compute_all_pairwise_records


def test_max_and_mean_disagreement():
    active_mods = ("retina", "foot", "clinical")
    risks = {"retina": 0.80, "foot": 0.20, "clinical": 0.30}
    weights = {"retina": 0.50, "foot": 0.30, "clinical": 0.20}
    uncertainties = {"retina": 0.05, "foot": 0.40, "clinical": 0.10}
    reliabilities = {"retina": 0.929956, "foot": 0.922266, "clinical": 0.825382}

    records = compute_all_pairwise_records(
        active_modalities=active_mods,
        risks=risks,
        weights=weights,
        uncertainties=uncertainties,
        reliabilities=reliabilities,
    )

    max_disag, dominant_pair = compute_max_disagreement(records)
    mean_disag = compute_mean_disagreement(records)

    # Pairs: RF = |0.80 - 0.20| = 0.60, RC = |0.80 - 0.30| = 0.50, FC = |0.20 - 0.30| = 0.10
    # max = 0.60 (retina, foot), mean = (0.60 + 0.50 + 0.10) / 3 = 0.40
    assert pytest.approx(max_disag, abs=1e-6) == 0.60
    assert dominant_pair == ("retina", "foot")
    assert pytest.approx(mean_disag, abs=1e-6) == 0.40


def test_weighted_dispersion():
    active_mods = ("retina", "foot", "clinical")
    risks = {"retina": 0.80, "foot": 0.20, "clinical": 0.30}
    weights = {"retina": 0.50, "foot": 0.30, "clinical": 0.20}
    # R_fusion = 0.50*0.80 + 0.30*0.20 + 0.20*0.30 = 0.40 + 0.06 + 0.06 = 0.52
    r_fusion = 0.52

    # V_w = 0.50*(0.80 - 0.52)^2 + 0.30*(0.20 - 0.52)^2 + 0.20*(0.30 - 0.52)^2
    #     = 0.50*(0.28)^2 + 0.30*(-0.32)^2 + 0.20*(-0.22)^2
    #     = 0.50*0.0784 + 0.30*0.1024 + 0.20*0.0484
    #     = 0.0392 + 0.03072 + 0.00968 = 0.0796
    # sigma_w = sqrt(0.0796) approx 0.2821347
    var_w, std_w = compute_weighted_dispersion(
        active_modalities=active_mods,
        risks=risks,
        weights=weights,
        r_fusion=r_fusion,
    )

    assert pytest.approx(var_w, abs=1e-6) == 0.0796
    assert pytest.approx(std_w, abs=1e-6) == math.sqrt(0.0796)


def test_zero_dispersion_consensus():
    active_mods = ("retina", "foot", "clinical")
    risks = {"retina": 0.50, "foot": 0.50, "clinical": 0.50}
    weights = {"retina": 0.40, "foot": 0.35, "clinical": 0.25}
    r_fusion = 0.50

    var_w, std_w = compute_weighted_dispersion(
        active_modalities=active_mods,
        risks=risks,
        weights=weights,
        r_fusion=r_fusion,
    )

    assert var_w == 0.0
    assert std_w == 0.0


def test_weight_entropy():
    # Uniform 3-way: H = -3 * (1/3 * ln(1/3)) = ln(3) approx 1.098612
    active_mods = ("retina", "foot", "clinical")
    weights_uniform = {"retina": 1/3, "foot": 1/3, "clinical": 1/3}
    h_uniform = compute_weight_entropy(active_mods, weights_uniform)
    assert pytest.approx(h_uniform, abs=1e-5) == math.log(3)

    # Completely concentrated: H = 0.0
    weights_conc = {"retina": 1.0, "foot": 0.0, "clinical": 0.0}
    h_conc = compute_weight_entropy(active_mods, weights_conc)
    assert h_conc == 0.0
