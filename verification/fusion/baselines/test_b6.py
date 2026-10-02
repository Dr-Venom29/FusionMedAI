"""
Unit tests for Baseline B6 (Full ACARA-U Dynamic Fusion) - Phase C11.5.
"""

import pytest
from src.fusion.baselines.decision_packet import (
    ModalityRecord,
    ControlledDecisionPacket,
)
from src.fusion.baselines.b6_acarau import FullACARAUBaseline
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)


def test_b6_full_dynamic_routing():
    """Tests B6 incorporates confidence, reliability, uncertainty, and quality bonus."""
    b6 = FullACARAUBaseline()

    packet = ControlledDecisionPacket(
        packet_id="PACKET_TEST_001",
        retina=ModalityRecord("r1", "retina", 0.70, (0.5, 0.5), 0.95, 0.05, 0.95, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.40, (0.5, 0.5), 0.60, 0.40, 0.60, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.20, (0.5, 0.5), 0.50, 0.50, 0.50, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    res = b6.evaluate(packet)

    assert res.status == "SUCCESS"
    assert res.dominant_modality == "retina"
    assert res.weights["retina"] > 0.60
    assert 0.0 <= res.r_fusion <= 1.0
