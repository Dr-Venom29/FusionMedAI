"""
verification/fusion/degradation/test_routing_response.py
Phase C11.10: Verification of Experiment B Routing Response & Authority Redistribution Across All 12 Operators
"""

import pytest
from pathlib import Path
from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.degradation.degradation_engine import DegradationEngine, apply_degradation_to_packet
from src.fusion.degradation.degradation_spec import (
    SEVERITY_D0_CLEAN,
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
def test_retina_operators_routing_response(engine, operator):
    r_recs, _, _ = engine.evaluate_routing_response(MODALITY_RETINA, operator)
    assert len(r_recs) == 4
    assert r_recs[3].mean_delta_w < 0.0
    assert r_recs[3].mean_rar > 0.0
    for r in r_recs:
        assert r.redistributed_authority_mean == pytest.approx(-r.mean_delta_w, abs=1e-5)


@pytest.mark.parametrize("operator", FOOT_OPERATORS)
def test_foot_operators_routing_response(engine, operator):
    r_recs, _, _ = engine.evaluate_routing_response(MODALITY_FOOT, operator)
    assert len(r_recs) == 4
    assert r_recs[3].mean_delta_w < 0.0
    assert r_recs[3].mean_rar > 0.0
    for r in r_recs:
        assert r.redistributed_authority_mean == pytest.approx(-r.mean_delta_w, abs=1e-5)


@pytest.mark.parametrize("operator", CLINICAL_OPERATORS)
def test_clinical_operators_routing_response(engine, operator):
    r_recs, _, _ = engine.evaluate_routing_response(MODALITY_CLINICAL, operator)
    assert len(r_recs) == 4
    assert r_recs[3].mean_delta_w < 0.0
    assert r_recs[3].mean_rar > 0.0
    for r in r_recs:
        assert r.redistributed_authority_mean == pytest.approx(-r.mean_delta_w, abs=1e-5)


def test_simplex_invariant_under_severe_degradation(engine):
    sample_packet = engine.cohort[0]
    deg_packet = apply_degradation_to_packet(
        sample_packet, MODALITY_RETINA, RETINA_OPERATORS[0], SEVERITY_D3_SEVERE
    )
    res = engine.dcri_engine.evaluate_packet(deg_packet, delta=0.20)
    total_w = sum(res.modality_weights.values())
    assert total_w == pytest.approx(1.0, abs=1e-6)
    for w in res.modality_weights.values():
        assert 0.0 <= w <= 1.0
