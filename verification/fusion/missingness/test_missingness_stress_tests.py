"""
verification/fusion/missingness/test_stress_tests.py
Unit tests for targeted stress dropouts and availability vs low-quality distinctions.
"""

from pathlib import Path
import pytest

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.dcri.dcri_engine import DCRIEngine
from src.fusion.missingness.stress_tests import (
    evaluate_stress_scenarios,
    evaluate_unavailable_vs_low_quality,
)


@pytest.fixture(scope="module")
def dcri_engine():
    return DCRIEngine()


@pytest.fixture(scope="module")
def sample_packet():
    cohort = load_frozen_cohort(repo_root=Path("."), n_packets=500, seed=115)
    return cohort[0]


def test_evaluate_stress_scenarios(sample_packet, dcri_engine):
    stress_dict = evaluate_stress_scenarios(sample_packet, dcri_engine, delta=0.20)
    assert len(stress_dict) == 7

    assert "missing_highest_confidence" in stress_dict
    assert "missing_highest_reliability" in stress_dict
    assert "missing_lowest_uncertainty" in stress_dict

    # Check that in missing_highest_reliability, retina is dropped and receives weight 0.0
    rec = stress_dict["missing_highest_reliability"]
    assert rec.dropped_modality == "retina"
    assert rec.weights_stress["retina"] == 0.0
    assert sum(rec.weights_stress.values()) == pytest.approx(1.0, abs=1e-5)


def test_unavailable_vs_low_quality(sample_packet, dcri_engine):
    res = evaluate_unavailable_vs_low_quality(sample_packet, dcri_engine, delta=0.20)
    assert res["distinction_verified"] is True
    assert res["unavailable_retina_weight"] == 0.0
    assert res["degraded_retina_weight"] > 0.0
