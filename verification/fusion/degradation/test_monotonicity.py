"""
verification/fusion/degradation/test_monotonicity.py
Phase C11.10: Verification of Monotonicity Rates & Quality-Authority Slopes Across All 12 Operators
"""

import pytest
from pathlib import Path
from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.degradation.degradation_engine import DegradationEngine
from src.fusion.degradation.degradation_spec import (
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
def test_retina_operators_monotonicity_and_slopes(engine, operator):
    _, slopes, mono = engine.evaluate_routing_response(MODALITY_RETINA, operator)
    assert mono.quality_monotonic_rate >= 0.90
    assert mono.routing_monotonic_rate >= 0.90
    assert slopes[-1].mean_slope > 0.0


@pytest.mark.parametrize("operator", FOOT_OPERATORS)
def test_foot_operators_monotonicity_and_slopes(engine, operator):
    _, slopes, mono = engine.evaluate_routing_response(MODALITY_FOOT, operator)
    assert mono.quality_monotonic_rate >= 0.90
    assert mono.routing_monotonic_rate >= 0.90
    assert slopes[-1].mean_slope > 0.0


@pytest.mark.parametrize("operator", CLINICAL_OPERATORS)
def test_clinical_operators_monotonicity_and_slopes(engine, operator):
    _, slopes, mono = engine.evaluate_routing_response(MODALITY_CLINICAL, operator)
    assert mono.quality_monotonic_rate >= 0.90
    assert mono.routing_monotonic_rate >= 0.90
    assert slopes[-1].mean_slope > 0.0

