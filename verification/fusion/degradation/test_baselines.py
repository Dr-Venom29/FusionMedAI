"""
verification/fusion/degradation/test_baselines.py
Phase C11.10: Verification of Baseline Comparisons (B1–B6, B5 vs B6 Quality Isolation) Across All Operators
"""

import pytest
from pathlib import Path
from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.degradation.degradation_engine import DegradationEngine
from src.fusion.degradation.degradation_spec import (
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
def test_retina_b6_vs_b5_quality_isolation(engine, operator):
    b_recs = engine.evaluate_baseline_comparisons(MODALITY_RETINA, operator)
    assert len(b_recs) == 24
    severe_b6 = [r for r in b_recs if r.severity == SEVERITY_D3_SEVERE and r.baseline_id == "B6"][0]
    severe_b5 = [r for r in b_recs if r.severity == SEVERITY_D3_SEVERE and r.baseline_id == "B5"][0]
    assert severe_b6.mean_delta_w < severe_b5.mean_delta_w
    assert severe_b6.b6_vs_b5_delta_w < 0.0


@pytest.mark.parametrize("operator", FOOT_OPERATORS)
def test_foot_b6_vs_b5_quality_isolation(engine, operator):
    b_recs = engine.evaluate_baseline_comparisons(MODALITY_FOOT, operator)
    assert len(b_recs) == 24
    severe_b6 = [r for r in b_recs if r.severity == SEVERITY_D3_SEVERE and r.baseline_id == "B6"][0]
    severe_b5 = [r for r in b_recs if r.severity == SEVERITY_D3_SEVERE and r.baseline_id == "B5"][0]
    assert severe_b6.mean_delta_w < severe_b5.mean_delta_w
    assert severe_b6.b6_vs_b5_delta_w < 0.0


@pytest.mark.parametrize("operator", CLINICAL_OPERATORS)
def test_clinical_b6_vs_b5_quality_isolation(engine, operator):
    b_recs = engine.evaluate_baseline_comparisons(MODALITY_CLINICAL, operator)
    assert len(b_recs) == 24
    severe_b6 = [r for r in b_recs if r.severity == SEVERITY_D3_SEVERE and r.baseline_id == "B6"][0]
    severe_b5 = [r for r in b_recs if r.severity == SEVERITY_D3_SEVERE and r.baseline_id == "B5"][0]
    assert severe_b6.mean_delta_w < severe_b5.mean_delta_w
    assert severe_b6.b6_vs_b5_delta_w < 0.0
