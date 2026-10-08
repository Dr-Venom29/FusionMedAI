"""
verification/fusion/combination_analysis/test_bootstrap_analysis.py
Phase C11.9: Unit tests for bootstrap confidence intervals and seed determinism.
"""

import pytest
import numpy as np

from src.fusion.combination_analysis.bootstrap_analysis import (
    compute_bootstrap_ci_1d,
    compute_bootstrap_ci_std,
    compute_paired_bootstrap_ci_diff,
)


def test_bootstrap_ci_contains_sample_mean():
    data = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    low, high = compute_bootstrap_ci_1d(data, n_resamples=500, seed=115)

    mean_val = np.mean(data)
    assert low <= mean_val <= high


def test_bootstrap_ci_determinism():
    data = [0.1, 0.2, 0.3, 0.4, 0.5]
    ci1 = compute_bootstrap_ci_1d(data, n_resamples=200, seed=115)
    ci2 = compute_bootstrap_ci_1d(data, n_resamples=200, seed=115)

    assert ci1 == ci2


def test_bootstrap_std_ci():
    data = [0.1, 0.2, 0.3, 0.4, 0.5]
    low, high = compute_bootstrap_ci_std(data, n_resamples=200, seed=115)
    std_val = np.std(data, ddof=1)
    assert low <= std_val <= high


def test_paired_bootstrap_ci_diff():
    a = [0.2, 0.3, 0.4, 0.5]
    b = [0.1, 0.2, 0.3, 0.4]
    mean_diff, low, high = compute_paired_bootstrap_ci_diff(a, b, n_resamples=200, seed=115)
    assert np.isclose(mean_diff, 0.1, atol=1e-6)
    assert low <= 0.1 <= high
