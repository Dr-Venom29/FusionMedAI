"""
Unit tests for Baseline B1 (Reliability-Selected Unimodal Baseline) - Phase C11.5.
"""

import pytest
from src.fusion.baselines.decision_packet import (
    ModalityRecord,
    ControlledDecisionPacket,
    FusionResult,
)
from src.fusion.baselines.b1_reliability_selected import ReliabilitySelectedBaseline
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)


@pytest.fixture
def standard_packet():
    return ControlledDecisionPacket(
        packet_id="PACKET_TEST_001",
        retina=ModalityRecord("r1", "retina", 0.35, (0.5, 0.5), 0.85, 0.10, 0.90, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.60, (0.5, 0.5), 0.75, 0.15, 0.85, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.20, (0.5, 0.5), 0.70, 0.20, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )


def test_b1_tri_modal_selection(standard_packet):
    """Tests B1 selects Retina when all three are available (R_R=0.929956 > R_F=0.922266 > R_C=0.825382)."""
    b1 = ReliabilitySelectedBaseline()
    res = b1.evaluate(standard_packet)

    assert res.status == "SUCCESS"
    assert res.dominant_modality == "retina"
    assert res.weights == {"retina": 1.0, "foot": 0.0, "clinical": 0.0}
    assert res.r_fusion == standard_packet.retina.risk
    assert res.routing_entropy == 0.0


def test_b1_foot_selection_when_retina_missing(standard_packet):
    """Tests B1 selects Foot when Retina is unavailable (R_F > R_C)."""
    b1 = ReliabilitySelectedBaseline()
    packet = ControlledDecisionPacket(
        packet_id="PACKET_TEST_002",
        retina=ModalityRecord("r1", "retina", 0.35, (0.5, 0.5), 0.85, 0.10, 0.0, False, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=standard_packet.foot,
        clinical=standard_packet.clinical,
        seed=115,
    )
    res = b1.evaluate(packet)

    assert res.status == "SUCCESS"
    assert res.dominant_modality == "foot"
    assert res.weights == {"retina": 0.0, "foot": 1.0, "clinical": 0.0}
    assert res.r_fusion == standard_packet.foot.risk


def test_b1_zero_modality_safe_rejection(standard_packet):
    """Tests B1 safely rejects when all modalities are missing."""
    b1 = ReliabilitySelectedBaseline()
    packet = ControlledDecisionPacket(
        packet_id="PACKET_TEST_003",
        retina=ModalityRecord("r1", "retina", 0.35, (0.5, 0.5), 0.85, 0.10, 0.0, False, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.60, (0.5, 0.5), 0.75, 0.15, 0.0, False, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.20, (0.5, 0.5), 0.70, 0.20, 0.0, False, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    res = b1.evaluate(packet)

    assert res.status == "NO_MODALITY_AVAILABLE"
    assert res.num_active == 0
    assert res.r_fusion == 0.0
    assert res.dominant_modality is None
