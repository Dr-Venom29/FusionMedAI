"""
Unit Tests for AUC Computation (Phase C11.3).
Verifies binary AUC, multiclass Macro OvR AUC, and bound enforcement.
"""

import pytest
import numpy as np

from src.fusion.reliability.auc import compute_binary_auc, compute_macro_ovr_auc


def test_binary_auc_perfect_discrimination():
    """Perfect binary predictions yield AUC = 1.0."""
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    auc = compute_binary_auc(y_true, y_prob)
    assert auc == 1.0


def test_binary_auc_chance_discrimination():
    """Uniform chance predictions yield AUC = 0.5."""
    y_true = [0, 0, 1, 1]
    y_prob = [0.5, 0.5, 0.5, 0.5]
    auc = compute_binary_auc(y_true, y_prob)
    assert auc == 0.5


def test_binary_auc_single_class_error():
    """Single class in ground truth must raise ValueError."""
    with pytest.raises(ValueError):
        compute_binary_auc([1, 1, 1], [0.5, 0.6, 0.7])


def test_macro_ovr_auc_multiclass_perfect():
    """Perfect 3-class predictions yield Macro OvR AUC = 1.0."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_prob = [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.85, 0.1, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.1, 0.85],
    ]
    auc = compute_macro_ovr_auc(y_true, y_prob, num_classes=3)
    assert auc == 1.0


def test_macro_ovr_auc_dimension_mismatch():
    """Dimension mismatch between num_classes and probability columns raises ValueError."""
    with pytest.raises(ValueError):
        compute_macro_ovr_auc([0, 1], [[0.5, 0.5]], num_classes=3)
