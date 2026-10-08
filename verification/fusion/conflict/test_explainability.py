"""
verification/fusion/conflict/test_explainability.py
Unit tests for deterministic structured conflict explainability summaries.
"""

import pytest
from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord
from src.fusion.conflict.conflict_engine import ConflictEngine
from src.fusion.conflict.conflict_explainability import generate_conflict_summary


@pytest.fixture
def high_conflict_packet() -> ControlledDecisionPacket:
    return ControlledDecisionPacket(
        packet_id="hc_packet_001",
        retina=ModalityRecord(
            sample_id="retina_hc001",
            modality="retina",
            risk=0.85,
            calibrated_probability=0.85,
            confidence=0.85,
            uncertainty=0.05,
            quality=0.95,
            availability=1,
            reliability=0.929956,
            model_version="retina_v1",
        ),
        foot=ModalityRecord(
            sample_id="foot_hc001",
            modality="foot",
            risk=0.15,
            calibrated_probability=0.15,
            confidence=0.70,
            uncertainty=0.40,
            quality=0.90,
            availability=1,
            reliability=0.922266,
            model_version="foot_v1",
        ),
        clinical=ModalityRecord(
            sample_id="clinical_hc001",
            modality="clinical",
            risk=0.20,
            calibrated_probability=0.20,
            confidence=0.65,
            uncertainty=0.10,
            quality=0.88,
            availability=1,
            reliability=0.825382,
            model_version="clinical_v1",
        ),
        seed=115,
    )


def test_high_conflict_explanation(high_conflict_packet):
    engine = ConflictEngine()
    res = engine.evaluate_packet(high_conflict_packet, delta=0.20)
    explanation = generate_conflict_summary(res)

    assert explanation["status"] == "HIGH"
    assert "High Decision Conflict Alert" in explanation["summary"]
    assert explanation["dominant_pair"] == ("retina", "foot")
    assert len(explanation["pairwise_breakdown"]) == 3
    assert explanation["dominant_modality"] == "retina"


def test_single_modality_explanation(high_conflict_packet):
    engine = ConflictEngine()
    regimes = engine.evaluate_modality_configurations(high_conflict_packet, delta=0.20)
    explanation = generate_conflict_summary(regimes["retina_only"])

    assert explanation["status"] == "SINGLE_MODALITY_EVALUATION"
    assert "Single modality" in explanation["summary"]


def test_zero_modality_explanation(high_conflict_packet):
    engine = ConflictEngine()
    regimes = engine.evaluate_modality_configurations(high_conflict_packet, delta=0.20)
    explanation = generate_conflict_summary(regimes["zero_modality"])

    assert explanation["status"] == "NO_INPUT_AVAILABLE"
    assert "No modalities" in explanation["summary"]
