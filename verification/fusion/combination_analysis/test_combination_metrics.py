"""
verification/fusion/combination_analysis/test_combination_metrics.py
Phase C11.9: Unit tests for combination statistical aggregations, entropy, and conflict metrics.
"""

import pytest
import math
import numpy as np

from src.fusion.combination_analysis.combination_metrics import (
    calc_distribution_stats,
    compute_routing_entropy,
    compute_conflict_metrics,
)


def test_calc_distribution_stats():
    vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    stats = calc_distribution_stats(vals)
    assert np.isclose(stats["mean"], 0.3)
    assert np.isclose(stats["median"], 0.3)
    assert stats["min"] == 0.1
    assert stats["max"] == 0.5
    assert stats["std"] > 0.0


def test_empty_distribution_stats():
    stats = calc_distribution_stats([])
    assert stats["mean"] == 0.0
    assert stats["std"] == 0.0


def test_routing_entropy():
    # Uniform 3-way
    w_uniform = {"retina": 1/3, "foot": 1/3, "clinical": 1/3}
    h_unif = compute_routing_entropy(w_uniform)
    assert np.isclose(h_unif, math.log(3), atol=1e-5)

    # 1-hot
    w_onehot = {"retina": 1.0, "foot": 0.0, "clinical": 0.0}
    h_one = compute_routing_entropy(w_onehot)
    assert h_one == 0.0


def test_conflict_metrics():
    risks = {"retina": 0.8, "foot": 0.2, "clinical": 0.5}
    weights = {"retina": 0.5, "foot": 0.3, "clinical": 0.2}
    active = ("retina", "foot", "clinical")

    conf = compute_conflict_metrics(risks, weights, active)
    assert np.isclose(conf["delta_max"], 0.6)  # |0.8 - 0.2|
    assert conf["delta_mean"] > 0.0
    assert conf["sigma_w"] > 0.0


def test_unimodal_conflict_metrics_zero():
    risks = {"retina": 0.8, "foot": 0.0, "clinical": 0.0}
    weights = {"retina": 1.0, "foot": 0.0, "clinical": 0.0}
    active = ("retina",)

    conf = compute_conflict_metrics(risks, weights, active)
    assert conf["delta_max"] == 0.0
    assert conf["delta_mean"] == 0.0
    assert conf["sigma_w"] == 0.0
