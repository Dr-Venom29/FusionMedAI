"""
Unit tests for Baseline B5 (Confidence + Reliability - Uncertainty Fusion) - Phase C11.5.
"""

import pytest
from src.fusion.baselines.decision_packet import (
    ModalityRecord,
    ControlledDecisionPacket,
)
from src.fusion.baselines.b5_confidence_reliability_uncertainty import ConfidenceReliabilityUncertaintyBaseline
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)


def test_b5_uncertainty_penalty_responsiveness():
    """Tests B5 downweights a modality when predictive uncertainty increases."""
    b5 = ConfidenceReliabilityUncertaintyBaseline(alpha=1.0, beta=1.0, gamma=1.0)

    # Retina has higher uncertainty -> penalizes weight
    p_low_u = ControlledDecisionPacket(
        packet_id="PACKET_TEST_001",
        retina=ModalityRecord("r1", "retina", 0.5, (0.5, 0.5), 0.80, 0.05, 0.80, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.5, (0.5, 0.5), 0.80, 0.20, 0.80, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.5, (0.5, 0.5), 0.80, 0.20, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    p_high_u = ControlledDecisionPacket(
        packet_id="PACKET_TEST_002",
        retina=ModalityRecord("r1", "retina", 0.5, (0.5, 0.5), 0.80, 0.80, 0.80, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.5, (0.5, 0.5), 0.80, 0.20, 0.80, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.5, (0.5, 0.5), 0.80, 0.20, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )

    res_low = b5.evaluate(p_low_u)
    res_high = b5.evaluate(p_high_u)

    assert res_low.weights["retina"] > res_high.weights["retina"]
