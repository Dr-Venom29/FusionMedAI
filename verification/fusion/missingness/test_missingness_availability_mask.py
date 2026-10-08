"""
verification/fusion/missingness/test_availability_mask.py
Unit tests for availability mask transformations and invariant enforcement.
"""

from pathlib import Path
import pytest
import numpy as np

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.missingness.availability_mask import (
    ALL_REGIMES,
    NON_EMPTY_REGIMES,
    BIMODAL_REGIMES,
    UNIMODAL_REGIMES,
    MODALITY_NAMES,
    apply_availability_mask,
    verify_availability_invariants,
)


@pytest.fixture(scope="module")
def sample_packet():
    cohort = load_frozen_cohort(repo_root=Path("."), n_packets=500, seed=115)
    return cohort[0]


def test_regime_definitions():
    assert len(ALL_REGIMES) == 8
    assert len(NON_EMPTY_REGIMES) == 7
    assert len(BIMODAL_REGIMES) == 3
    assert len(UNIMODAL_REGIMES) == 3
    assert ALL_REGIMES["tri_modal"] == ("retina", "foot", "clinical")
    assert ALL_REGIMES["zero_modality"] == ()


def test_apply_availability_mask_bimodal(sample_packet):
    masked = apply_availability_mask(sample_packet, ("retina", "clinical"), suffix="test_rc")
    assert masked.packet_id == f"{sample_packet.packet_id}_test_rc"
    assert masked.retina.availability is True
    assert masked.clinical.availability is True
    assert masked.foot.availability is False
    assert masked.foot.confidence == 0.0
    assert masked.foot.uncertainty == 0.0
    assert masked.foot.quality == 0.0
    # Underlying risk and reliability preserved for test invariants
    assert masked.foot.risk == sample_packet.foot.risk
    assert masked.foot.reliability == sample_packet.foot.reliability
    assert set(masked.available_modalities) == {"retina", "clinical"}
    assert masked.num_available == 2


def test_apply_availability_mask_zero_modality(sample_packet):
    masked = apply_availability_mask(sample_packet, (), suffix="test_zero")
    assert masked.available_modalities == []
    assert masked.num_available == 0
    assert not masked.retina.availability
    assert not masked.foot.availability
    assert not masked.clinical.availability


def test_verify_availability_invariants_valid(sample_packet):
    masked = apply_availability_mask(sample_packet, ("retina", "foot"))
    weights = {"retina": 0.65, "foot": 0.35, "clinical": 0.0}
    is_valid, err = verify_availability_invariants(masked, weights)
    assert is_valid is True
    assert err is None


def test_verify_availability_invariants_leakage_rejection(sample_packet):
    masked = apply_availability_mask(sample_packet, ("retina", "foot"))
    # Inactive clinical modality receives weight > 0 (leakage)
    weights = {"retina": 0.60, "foot": 0.35, "clinical": 0.05}
    is_valid, err = verify_availability_invariants(masked, weights)
    assert is_valid is False
    assert "Unavailable modality 'clinical' received non-zero weight" in err


def test_verify_availability_invariants_simplex_violation(sample_packet):
    masked = apply_availability_mask(sample_packet, ("retina", "foot"))
    # Active weights do not sum to 1.0
    weights = {"retina": 0.50, "foot": 0.40, "clinical": 0.0}
    is_valid, err = verify_availability_invariants(masked, weights)
    assert is_valid is False
    assert "Active modality weights sum to" in err
