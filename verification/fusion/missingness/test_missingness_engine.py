"""
verification/fusion/missingness/test_missingness_engine.py
Unit tests for MissingnessEngine packet and cohort evaluations.
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


def test_evaluate_packet_all_8_regimes(engine, sample_packet):
    regimes = engine.evaluate_packet_all_regimes(sample_packet, delta=0.20)
    assert len(regimes) == 8

    # Tri-modal check
    assert regimes["tri_modal"].num_active == 3
    assert regimes["tri_modal"].status == "SUCCESS"

    # Zero modality check
    assert regimes["zero_modality"].num_active == 0
    assert regimes["zero_modality"].status == "NO_MODALITY_AVAILABLE"
    assert regimes["zero_modality"].r_fusion == 0.0
    assert regimes["zero_modality"].dcri == 0.0


def test_evaluate_redistribution_calculation(engine, sample_packet):
    full_eval = engine.evaluate_packet_regime(sample_packet, "tri_modal", delta=0.20)
    fc_eval = engine.evaluate_packet_regime(sample_packet, "foot_clinical", delta=0.20)

    rec = engine.compute_redistribution_record(full_eval, fc_eval, delta=0.20)
    assert rec.removed_modalities == ("retina",)
    assert set(rec.remaining_modalities) == {"foot", "clinical"}
    assert rec.delta_w["retina"] < 0.0  # Dropped from positive weight to 0.0
    assert rec.delta_w["foot"] >= 0.0   # Absorbed redistributed authority
    assert rec.delta_w["clinical"] >= 0.0
