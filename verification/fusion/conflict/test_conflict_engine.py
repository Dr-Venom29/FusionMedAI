"""
verification/fusion/conflict/test_conflict_engine.py
Unit tests for ConflictEngine orchestration over single packets, availability regimes, and cohorts.
"""

from pathlib import Path
import pytest
from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord
from src.fusion.conflict.conflict_engine import ConflictEngine
from src.fusion.dcri.cohort_loader import load_frozen_cohort


@pytest.fixture
def sample_packet() -> ControlledDecisionPacket:
    return ControlledDecisionPacket(
        packet_id="test_packet_001",
        retina=ModalityRecord(
            sample_id="retina_001",
            modality="retina",
            risk=0.80,
            calibrated_probability=0.80,
            confidence=0.85,
            uncertainty=0.08,
            quality=0.95,
            availability=1,
            reliability=0.929956,
            model_version="retina_v1",
        ),
        foot=ModalityRecord(
            sample_id="foot_001",
            modality="foot",
            risk=0.25,
            calibrated_probability=0.25,
            confidence=0.70,
            uncertainty=0.35,
            quality=0.90,
            availability=1,
            reliability=0.922266,
            model_version="foot_v1",
        ),
        clinical=ModalityRecord(
            sample_id="clinical_001",
            modality="clinical",
            risk=0.30,
            calibrated_probability=0.30,
            confidence=0.65,
            uncertainty=0.10,
            quality=0.88,
            availability=1,
            reliability=0.825382,
            model_version="clinical_v1",
        ),
        seed=115,
    )


def test_evaluate_packet(sample_packet):
    engine = ConflictEngine()
    res = engine.evaluate_packet(packet=sample_packet, delta=0.20)

    assert res.packet_id == "test_packet_001"
    assert res.num_active == 3
    assert res.conflict_available is True
    assert res.max_disagreement == pytest.approx(0.55, abs=1e-5) # |0.80 - 0.25|
    assert res.dominant_conflict_pair == ("retina", "foot")
    assert res.conflict_severity == "HIGH"
    assert res.weighted_variance > 0.0
    assert res.weighted_std > 0.0
    assert len(res.pairwise_records) == 3


def test_evaluate_modality_configurations(sample_packet):
    engine = ConflictEngine()
    regimes = engine.evaluate_modality_configurations(sample_packet, delta=0.20)

    assert set(regimes.keys()) == {
        "retina_only",
        "foot_only",
        "clinical_only",
        "retina_foot",
        "retina_clinical",
        "foot_clinical",
        "tri_modal",
        "zero_modality",
    }

    # Unimodal
    assert regimes["retina_only"].conflict_available is False
    assert regimes["retina_only"].conflict_severity == "NOT_APPLICABLE"
    assert regimes["retina_only"].max_disagreement is None

    # Zero
    assert regimes["zero_modality"].conflict_available is False
    assert regimes["zero_modality"].conflict_severity == "NO_MODALITY_AVAILABLE"

    # Bimodal RF
    assert regimes["retina_foot"].conflict_available is True
    assert regimes["retina_foot"].max_disagreement == pytest.approx(0.55, abs=1e-5)
    assert len(regimes["retina_foot"].pairwise_records) == 1

    # Tri-Modal
    assert regimes["tri_modal"].conflict_available is True
    assert len(regimes["tri_modal"].pairwise_records) == 3


def test_frozen_cohort_execution():
    cohort = load_frozen_cohort(repo_root=Path("."))
    assert len(cohort) == 500

    engine = ConflictEngine()
    results = engine.evaluate_cohort(cohort, delta=0.20)

    assert len(results) == 500
    for r in results:
        assert r.conflict_available is True
        assert r.max_disagreement >= 0.0
        assert r.mean_disagreement >= 0.0
        assert r.weighted_variance >= 0.0
        assert r.conflict_severity in {"LOW", "MODERATE", "HIGH"}
