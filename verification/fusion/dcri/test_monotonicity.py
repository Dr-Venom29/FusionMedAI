"""
Unit tests for monotonicity mathematical invariants in DCRI (Phase C11.6).
Includes both end-to-end monotonicity and isolated fixed-router Tier-2 discounting tests.
"""

import pytest
from src.fusion.baselines.decision_packet import ModalityRecord, ControlledDecisionPacket
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)
from src.fusion.router.router_result import RouterResult
from src.fusion.dcri.dcri_engine import DCRIEngine


def test_uncertainty_monotonicity_fixed_router_weights():
    """
    Scientifically isolates Tier-2 uncertainty discounting:
    Holds ACARA-U router weights w_i strictly fixed and varies only U_i.
    Verifies that DCRI strictly decreases with rate Delta DCRI == -delta * Delta U_sum.
    """
    engine = DCRIEngine()
    delta = 0.20

    # Fixed mock router result (holding weights constant)
    fixed_router_res = RouterResult(
        weights={"retina": 0.50, "foot": 0.30, "clinical": 0.20},
        logits={"retina": 1.0, "foot": 0.5, "clinical": 0.2},
        masked_logits={"retina": 1.0, "foot": 0.5, "clinical": 0.2},
        availability={"retina": True, "foot": True, "clinical": True},
        coefficients={},
        active_modalities=["retina", "foot", "clinical"],
        num_active=3,
        normalization_sum=1.0,
        routing_entropy=1.0,
        dominant_modality="retina",
        status="SUCCESS",
        router_version="test_v1.0",
    )

    p1 = ControlledDecisionPacket(
        packet_id="P_FIXED_U1",
        retina=ModalityRecord("r1", "retina", 0.60, (0.5, 0.5), 0.80, 0.10, 0.80, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.50, (0.5, 0.5), 0.80, 0.10, 0.80, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.40, (0.5, 0.5), 0.80, 0.10, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    p2 = ControlledDecisionPacket(
        packet_id="P_FIXED_U2",
        retina=ModalityRecord("r1", "retina", 0.60, (0.5, 0.5), 0.80, 0.30, 0.80, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.50, (0.5, 0.5), 0.80, 0.30, 0.80, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.40, (0.5, 0.5), 0.80, 0.30, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )

    res1 = engine.evaluate_packet(p1, delta=delta, router_result=fixed_router_res)
    res2 = engine.evaluate_packet(p2, delta=delta, router_result=fixed_router_res)

    # R_fusion must be identical because weights and risks are identical
    assert abs(res1.r_fusion - res2.r_fusion) < 1e-7
    # DCRI must strictly decrease
    assert res1.dcri > res2.dcri
    # Delta DCRI must equal -delta * (U_sum_2 - U_sum_1)
    delta_u_sum = res2.u_sum - res1.u_sum  # 0.90 - 0.30 = 0.60
    expected_shift = -delta * delta_u_sum  # -0.20 * 0.60 = -0.12
    actual_shift = res2.dcri - res1.dcri
    assert abs(actual_shift - expected_shift) < 1e-6


def test_uncertainty_monotonicity_end_to_end():
    """End-to-end test where uncertainty modulates both router logits (Tier 1) and DCRI discount (Tier 2)."""
    engine = DCRIEngine()
    delta = 0.20

    p_low = ControlledDecisionPacket(
        packet_id="P_LOW_U",
        retina=ModalityRecord("r1", "retina", 0.60, (0.5, 0.5), 0.80, 0.05, 0.80, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.50, (0.5, 0.5), 0.80, 0.05, 0.80, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.40, (0.5, 0.5), 0.80, 0.05, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    p_med = ControlledDecisionPacket(
        packet_id="P_MED_U",
        retina=ModalityRecord("r1", "retina", 0.60, (0.5, 0.5), 0.80, 0.25, 0.80, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.50, (0.5, 0.5), 0.80, 0.25, 0.80, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.40, (0.5, 0.5), 0.80, 0.25, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    p_high = ControlledDecisionPacket(
        packet_id="P_HIGH_U",
        retina=ModalityRecord("r1", "retina", 0.60, (0.5, 0.5), 0.80, 0.70, 0.90, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.50, (0.5, 0.5), 0.80, 0.70, 0.90, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.40, (0.5, 0.5), 0.80, 0.70, 0.90, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )

    res_low = engine.evaluate_packet(p_low, delta=delta)
    res_med = engine.evaluate_packet(p_med, delta=delta)
    res_high = engine.evaluate_packet(p_high, delta=delta)

    assert res_low.dcri > res_med.dcri > res_high.dcri


def test_delta_monotonicity_fixed_packet():
    """Tests delta in [0, 0.05, 0.1, 0.2, 0.5, 1.0] strictly decreases DCRI."""
    engine = DCRIEngine()
    packet = ControlledDecisionPacket(
        packet_id="P_MONO_DELTA",
        retina=ModalityRecord("r1", "retina", 0.70, (0.5, 0.5), 0.80, 0.15, 0.90, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.50, (0.5, 0.5), 0.75, 0.30, 0.85, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.30, (0.5, 0.5), 0.70, 0.10, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )

    deltas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    dcris = [engine.evaluate_packet(packet, delta=d).dcri for d in deltas]

    for i in range(len(dcris) - 1):
        assert dcris[i] > dcris[i+1], f"Delta monotonicity failure: delta={deltas[i]} ({dcris[i]}) <= delta={deltas[i+1]} ({dcris[i+1]})"
