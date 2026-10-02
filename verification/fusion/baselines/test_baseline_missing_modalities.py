"""
Unit tests for missing-modality handling across all baselines B1–B6 - Phase C11.5.
"""

import pytest
from src.fusion.baselines.decision_packet import (
    ModalityRecord,
    ControlledDecisionPacket,
)
from src.fusion.baselines.fusion_runner import FusionRunner
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)


@pytest.mark.parametrize(
    "a_r,a_f,a_c",
    [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_all_baselines_7_configs_zero_weight_for_missing(a_r, a_f, a_c):
    """Tests that every baseline assigns exactly 0.0 weight to unavailable modalities across all 7 configs."""
    packet = ControlledDecisionPacket(
        packet_id="PACKET_CFG",
        retina=ModalityRecord("r1", "retina", 0.65, (0.5, 0.5), 0.85, 0.15, 0.90 if a_r else 0.0, a_r, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.45, (0.5, 0.5), 0.80, 0.20, 0.85 if a_f else 0.0, a_f, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.30, (0.5, 0.5), 0.75, 0.25, 0.80 if a_c else 0.0, a_c, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )

    runner = FusionRunner()
    results = runner.evaluate_all_baselines(packet)

    for b_id, res in results.items():
        assert res.status == "SUCCESS"
        if not a_r:
            assert res.weights["retina"] == 0.0
        if not a_f:
            assert res.weights["foot"] == 0.0
        if not a_c:
            assert res.weights["clinical"] == 0.0
        assert abs(sum(res.weights.values()) - 1.0) < 1e-7


def test_zero_modality_rejection_across_baselines():
    """Tests that every baseline safely rejects when no modalities are available."""
    packet = ControlledDecisionPacket(
        packet_id="PACKET_EMPTY",
        retina=ModalityRecord("r1", "retina", 0.65, (0.5, 0.5), 0.85, 0.15, 0.0, False, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.45, (0.5, 0.5), 0.80, 0.20, 0.0, False, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.30, (0.5, 0.5), 0.75, 0.25, 0.0, False, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )

    runner = FusionRunner()
    results = runner.evaluate_all_baselines(packet)

    for b_id, res in results.items():
        assert res.status == "NO_MODALITY_AVAILABLE"
        assert res.num_active == 0
        assert res.r_fusion == 0.0
        assert res.weights == {"retina": 0.0, "foot": 0.0, "clinical": 0.0}
