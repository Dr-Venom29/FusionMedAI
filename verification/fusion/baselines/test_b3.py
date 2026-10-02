"""
Unit tests for Baseline B3 (Confidence-Only Fusion) - Phase C11.5.
"""

import math
import pytest
from src.fusion.baselines.decision_packet import (
    ModalityRecord,
    ControlledDecisionPacket,
)
from src.fusion.baselines.b3_confidence import ConfidenceFusionBaseline
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)


def test_b3_confidence_responsiveness():
    """Tests B3 allocates weight dynamically based on model confidence."""
    b3 = ConfidenceFusionBaseline(alpha=1.0)

    packet = ControlledDecisionPacket(
        packet_id="PACKET_TEST_001",
        retina=ModalityRecord("r1", "retina", 0.8, (0.5, 0.5), 0.90, 0.10, 0.90, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.5, (0.5, 0.5), 0.50, 0.10, 0.90, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.2, (0.5, 0.5), 0.20, 0.10, 0.90, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    res = b3.evaluate(packet)

    assert res.status == "SUCCESS"
    assert res.weights["retina"] > res.weights["foot"] > res.weights["clinical"]
    assert res.dominant_modality == "retina"
    assert abs(sum(res.weights.values()) - 1.0) < 1e-7
