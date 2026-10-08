"""
verification/fusion/missingness/test_baseline_comparison.py
Unit tests for comparative evaluation of baselines B1–B6 across missingness regimes.
"""

from pathlib import Path
import pytest

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.missingness.missingness_engine import MissingnessEngine


@pytest.fixture(scope="module")
def engine():
    return MissingnessEngine()


@pytest.fixture(scope="module")
def sample_cohort():
    cohort = load_frozen_cohort(repo_root=Path("."), n_packets=500, seed=115)
    return cohort[:20]  # Fast subset for unit test


def test_baseline_comparative_sweep(engine, sample_cohort):
    comp = engine.evaluate_baselines_comparative(sample_cohort)
    for b_id in ["B1", "B2", "B3", "B4", "B5", "B6"]:
        assert b_id in comp
        assert "missing_retina_FC" in comp[b_id]
        assert "missing_foot_RC" in comp[b_id]
        assert "missing_clinical_RF" in comp[b_id]
        assert comp[b_id]["missing_retina_FC"]["stats"]["mean"] >= 0.0
