"""
verification/fusion/missingness/test_invariance.py
Unit tests for masked-value invariance (unavailable modality signals cannot alter active outputs).
"""

from pathlib import Path
import pytest

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.dcri.dcri_engine import DCRIEngine
from src.fusion.missingness.stress_tests import evaluate_masked_value_invariance


@pytest.fixture(scope="module")
def dcri_engine():
    return DCRIEngine()


@pytest.fixture(scope="module")
def sample_packet():
    cohort = load_frozen_cohort(repo_root=Path("."), n_packets=500, seed=115)
    return cohort[0]


def test_masked_value_invariance_sample_packet(sample_packet, dcri_engine):
    inv_records = evaluate_masked_value_invariance(sample_packet, dcri_engine, delta=0.20)
    # 3 bimodal regimes * 5 perturbation sets = 15 checks
    assert len(inv_records) == 15
    for rec in inv_records:
        assert rec.passed is True
        assert rec.active_weights_match is True
        assert rec.r_fusion_match is True
        assert rec.dcri_match is True
        assert rec.max_weight_diff < 1e-9
        assert rec.r_fusion_diff < 1e-9
        assert rec.dcri_diff < 1e-9
