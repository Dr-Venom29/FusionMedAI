"""
verification/fusion/degradation/test_quality_response.py
Phase C11.10: Comprehensive Verification of Experiment A Quality Response Across All 12 Operators
"""

import pytest
from pathlib import Path
from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.degradation.degradation_engine import DegradationEngine
from src.fusion.degradation.degradation_spec import (
    SEVERITY_D0_CLEAN,
    SEVERITY_D1_MILD,
    SEVERITY_D2_MODERATE,
    SEVERITY_D3_SEVERE,
    MODALITY_RETINA,
    MODALITY_FOOT,
    MODALITY_CLINICAL,
    RETINA_OPERATORS,
    FOOT_OPERATORS,
    CLINICAL_OPERATORS,
)

repo_root = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def engine():
    cohort = load_frozen_cohort(repo_root=repo_root, n_packets=500, seed=115)
    return DegradationEngine(cohort=cohort, n_bootstrap=50, seed=115)


@pytest.mark.parametrize("operator", RETINA_OPERATORS)
def test_all_retina_operators_quality_decay(engine, operator):
    recs = engine.evaluate_quality_response(MODALITY_RETINA, operator)
    assert len(recs) == 4
    # Q(D0) >= Q(D1) >= Q(D2) >= Q(D3)
    assert recs[0].mean_q_degraded >= recs[1].mean_q_degraded - 1e-5
    assert recs[1].mean_q_degraded >= recs[2].mean_q_degraded - 1e-5
    assert recs[2].mean_q_degraded >= recs[3].mean_q_degraded - 1e-5
    assert recs[0].mean_delta_q == pytest.approx(0.0, abs=1e-6)
    assert recs[3].mean_delta_q < 0.0
    for r in recs:
        assert 0.0 <= r.mean_q_degraded <= 1.0


@pytest.mark.parametrize("operator", FOOT_OPERATORS)
def test_all_foot_operators_quality_decay(engine, operator):
    recs = engine.evaluate_quality_response(MODALITY_FOOT, operator)
    assert len(recs) == 4
    assert recs[0].mean_q_degraded >= recs[1].mean_q_degraded - 1e-5
    assert recs[1].mean_q_degraded >= recs[2].mean_q_degraded - 1e-5
    assert recs[2].mean_q_degraded >= recs[3].mean_q_degraded - 1e-5
    assert recs[0].mean_delta_q == pytest.approx(0.0, abs=1e-6)
    assert recs[3].mean_delta_q < 0.0
    for r in recs:
        assert 0.0 <= r.mean_q_degraded <= 1.0


@pytest.mark.parametrize("operator", CLINICAL_OPERATORS)
def test_all_clinical_operators_quality_decay(engine, operator):
    recs = engine.evaluate_quality_response(MODALITY_CLINICAL, operator)
    assert len(recs) == 4
    assert recs[0].mean_q_degraded >= recs[1].mean_q_degraded - 1e-5
    assert recs[1].mean_q_degraded >= recs[2].mean_q_degraded - 1e-5
    assert recs[2].mean_q_degraded >= recs[3].mean_q_degraded - 1e-5
    assert recs[0].mean_delta_q == pytest.approx(0.0, abs=1e-6)
    assert recs[3].mean_delta_q < 0.0
    for r in recs:
        assert 0.0 <= r.mean_q_degraded <= 1.0
