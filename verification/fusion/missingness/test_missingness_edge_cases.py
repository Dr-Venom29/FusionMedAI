"""
verification/fusion/missingness/test_edge_cases.py
Unit tests for boundary conditions, safe rejection, and unimodal simplex exactness.
"""

from pathlib import Path
import pytest

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.missingness.missingness_engine import MissingnessEngine


@pytest.fixture(scope="module")
def engine():
    return MissingnessEngine()


@pytest.fixture(scope="module")
def sample_packet():
    cohort = load_frozen_cohort(repo_root=Path("."), n_packets=500, seed=115)
    return cohort[0]


def test_zero_modality_safe_rejection(engine, sample_packet):
    zero_eval = engine.evaluate_packet_regime(sample_packet, "zero_modality", delta=0.20)
    assert zero_eval.num_active == 0
    assert zero_eval.status == "NO_MODALITY_AVAILABLE"
    assert zero_eval.r_fusion == 0.0
    assert zero_eval.dcri == 0.0
    assert zero_eval.weights == {"retina": 0.0, "foot": 0.0, "clinical": 0.0}


def test_unimodal_simplex_exactness(engine, sample_packet):
    for m in ["retina", "foot", "clinical"]:
        reg_name = f"{m}_only"
        u_eval = engine.evaluate_packet_regime(sample_packet, reg_name, delta=0.20)
        assert u_eval.num_active == 1
        assert u_eval.status == "SUCCESS"
        assert u_eval.weights[m] == pytest.approx(1.0, abs=1e-6)
        assert u_eval.entropy == pytest.approx(0.0, abs=1e-6)
        assert u_eval.w_max == pytest.approx(1.0, abs=1e-6)
        assert u_eval.dominant_modality == m
