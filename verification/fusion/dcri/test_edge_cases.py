"""
Unit tests for numerical edge cases, boundaries, negative DCRI, and precision (Phase C11.6).
"""

import pytest
import math
from src.fusion.baselines.decision_packet import ModalityRecord, ControlledDecisionPacket, DecisionPacketError
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)
from src.fusion.dcri.dcri_engine import DCRIEngine
from src.fusion.dcri.dcri_result import DCRIContractError


def test_negative_dcri_allowed_and_bounded():
    """Verifies that DCRI is not improperly clamped and bounded in [-delta * |A|, 1.0]."""
    engine = DCRIEngine()
    # High uncertainty, low risk
    p = ControlledDecisionPacket(
        packet_id="P_NEG",
        retina=ModalityRecord("r1", "retina", 0.05, (0.5, 0.5), 0.85, 0.90, 0.90, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.05, (0.5, 0.5), 0.85, 0.90, 0.90, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.05, (0.5, 0.5), 0.85, 0.90, 0.90, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    res = engine.evaluate_packet(p, delta=1.0)
    assert res.dcri < 0.0
    # Expected: R_fusion = 0.05, penalty = 1.0 * 2.70 = 2.70 => DCRI = -2.65
    assert abs(res.dcri - (0.05 - 2.70)) < 1e-6
    assert res.dcri >= -3.0


def test_extreme_boundaries_zero_and_one():
    engine = DCRIEngine()
    # Boundary 0
    p_zero = ControlledDecisionPacket(
        packet_id="P_ZERO",
        retina=ModalityRecord("r1", "retina", 0.0, (1.0, 0.0), 1.0, 0.0, 1.0, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.0, (1.0, 0.0), 1.0, 0.0, 1.0, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.0, (1.0, 0.0), 1.0, 0.0, 1.0, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    res_zero = engine.evaluate_packet(p_zero, delta=0.2)
    assert res_zero.r_fusion == 0.0
    assert res_zero.uncertainty_penalty == 0.0
    assert res_zero.dcri == 0.0

    # Boundary 1
    p_one = ControlledDecisionPacket(
        packet_id="P_ONE",
        retina=ModalityRecord("r1", "retina", 1.0, (0.0, 1.0), 1.0, 0.0, 1.0, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 1.0, (0.0, 1.0), 1.0, 0.0, 1.0, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 1.0, (0.0, 1.0), 1.0, 0.0, 1.0, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    res_one = engine.evaluate_packet(p_one, delta=0.0)
    assert res_one.r_fusion == 1.0
    assert res_one.dcri == 1.0


def test_deterministic_reproducibility():
    engine = DCRIEngine()
    p = ControlledDecisionPacket(
        packet_id="P_DET",
        retina=ModalityRecord("r1", "retina", 0.55, (0.5, 0.5), 0.85, 0.15, 0.90, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.45, (0.5, 0.5), 0.75, 0.25, 0.85, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.35, (0.5, 0.5), 0.70, 0.20, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    res1 = engine.evaluate_packet(p, delta=0.20)
    res2 = engine.evaluate_packet(p, delta=0.20)
    assert res1.dcri == res2.dcri
    assert res1.r_fusion == res2.r_fusion
    assert res1.weighted_risk_contributions == res2.weighted_risk_contributions
