"""
Unit tests for Baseline B2 (Uniform Average Fusion) - Phase C11.5.
"""

import math
import pytest
from src.fusion.baselines.decision_packet import (
    ModalityRecord,
    ControlledDecisionPacket,
)
from src.fusion.baselines.b2_uniform import UniformAverageBaseline
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)


@pytest.fixture
def standard_packet():
    return ControlledDecisionPacket(
        packet_id="PACKET_TEST_001",
        retina=ModalityRecord("r1", "retina", 0.40, (0.5, 0.5), 0.85, 0.10, 0.90, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.70, (0.5, 0.5), 0.75, 0.15, 0.85, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.20, (0.5, 0.5), 0.70, 0.20, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )


def test_b2_tri_modal_equal_weighting(standard_packet):
    """Tests B2 assigns exactly 1/3 weight and entropy ln(3) in tri-modal state."""
    b2 = UniformAverageBaseline()
    res = b2.evaluate(standard_packet)

    assert res.status == "SUCCESS"
    for mod in ["retina", "foot", "clinical"]:
        assert abs(res.weights[mod] - (1.0 / 3.0)) < 1e-7

    expected_r = (0.40 + 0.70 + 0.20) / 3.0
    assert abs(res.r_fusion - expected_r) < 1e-6
    assert abs(res.routing_entropy - math.log(3.0)) < 1e-6


def test_b2_bi_modal_equal_weighting(standard_packet):
    """Tests B2 assigns exactly 0.5 weight in bi-modal state."""
    packet = ControlledDecisionPacket(
        packet_id="PACKET_TEST_002",
        retina=standard_packet.retina,
        foot=standard_packet.foot,
        clinical=ModalityRecord("c1", "clinical", 0.20, (0.5, 0.5), 0.70, 0.20, 0.0, False, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    b2 = UniformAverageBaseline()
    res = b2.evaluate(packet)

    assert res.status == "SUCCESS"
    assert res.weights["retina"] == 0.5
    assert res.weights["foot"] == 0.5
    assert res.weights["clinical"] == 0.0
    assert abs(res.r_fusion - 0.55) < 1e-6
    assert abs(res.routing_entropy - math.log(2.0)) < 1e-6
