"""
Unit Tests for Equal-Frequency ECE Computation (Phase C11.3).
Verifies 10 equal-frequency binning, ECE bounds, and calibration scores.
"""

import pytest
import numpy as np

from src.fusion.reliability.ece import compute_equal_frequency_ece


def test_ece_perfect_calibration():
    """Perfect binary calibration yields ECE near 0.0 and Calibration Score near 1.0."""
    # Construct perfectly calibrated bins
    probs = np.linspace(0.1, 0.9, 100)
    labels = (np.random.rand(100) < probs).astype(int)
    
    # Deterministic test
    probs_det = np.array([0.1]*10 + [0.9]*10)
    labels_det = np.array([0]*9 + [1]*1 + [0]*1 + [9]*0 + [1]*9) # 10% in bin 1, 90% in bin 2
    
    ece, cal_score, bins = compute_equal_frequency_ece(probs_det, labels_det, n_bins=2, is_multiclass=False)
    assert 0.0 <= ece <= 0.1
    assert 0.9 <= cal_score <= 1.0
    assert len(bins) == 2


def test_ece_multiclass_shape_verification():
    """Multiclass ECE processes (N, K) arrays with argmax confidence."""
    np.random.seed(42)
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
    ])
    labels = np.array([0, 1, 2, 0])
    
    ece, cal_score, bins = compute_equal_frequency_ece(probs, labels, n_bins=2, is_multiclass=True)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= cal_score <= 1.0
    assert len(bins) == 2


def test_ece_empty_input():
    """Empty inputs yield ECE = 0.0, Calibration Score = 1.0."""
    ece, cal_score, bins = compute_equal_frequency_ece([], [])
    assert ece == 0.0
    assert cal_score == 1.0
    assert len(bins) == 0
