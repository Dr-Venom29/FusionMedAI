"""
verification/fusion/conflict/test_conflict_classifier.py
Unit tests for operational conflict severity classification.
"""

import pytest
from src.fusion.conflict.conflict_classifier import (
    classify_conflict_severity,
    OPERATIONAL_LOW_THRESHOLD,
    OPERATIONAL_HIGH_THRESHOLD,
)


def test_classify_availability_states():
    assert classify_conflict_severity(num_active=0, max_disagreement=None) == "NO_MODALITY_AVAILABLE"
    assert classify_conflict_severity(num_active=1, max_disagreement=None) == "NOT_APPLICABLE"


def test_classify_severity_bands():
    # Low: < 0.20
    assert classify_conflict_severity(num_active=2, max_disagreement=0.00) == "LOW"
    assert classify_conflict_severity(num_active=3, max_disagreement=0.199) == "LOW"

    # Moderate: [0.20, 0.35)
    assert classify_conflict_severity(num_active=2, max_disagreement=0.20) == "MODERATE"
    assert classify_conflict_severity(num_active=3, max_disagreement=0.349) == "MODERATE"

    # High: >= 0.35
    assert classify_conflict_severity(num_active=2, max_disagreement=0.35) == "HIGH"
    assert classify_conflict_severity(num_active=3, max_disagreement=0.85) == "HIGH"


def test_invalid_multi_modality_without_disagreement():
    with pytest.raises(ValueError, match="max_disagreement cannot be None"):
        classify_conflict_severity(num_active=2, max_disagreement=None)
