"""
Unit tests for Baseline B4 (Confidence + Reliability Fusion) - Phase C11.5.
"""

import pytest
from src.fusion.baselines.decision_packet import (
    ModalityRecord,
    ControlledDecisionPacket,
)
from src.fusion.baselines.b4_confidence_reliability import ConfidenceReliabilityBaseline
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)


def test_b4_confidence_and_reliability_integration():
    """Tests B4 integrates both confidence and historical reliability prior."""
    b4 = ConfidenceReliabilityBaseline(alpha=1.0, beta=1.0)

    # Identical confidence: reliability prior dictates ordering
    packet = ControlledDecisionPacket(
        packet_id="PACKET_TEST_001",
        retina=ModalityRecord("r1", "retina", 0.5, (0.5, 0.5), 0.70, 0.20, 0.80, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.5, (0.5, 0.5), 0.70, 0.20, 0.80, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.5, (0.5, 0.5), 0.70, 0.20, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    res = b4.evaluate(packet)

    assert res.status == "SUCCESS"
    assert res.weights["retina"] > res.weights["foot"] > res.weights["clinical"]
