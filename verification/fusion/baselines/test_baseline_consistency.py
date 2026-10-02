"""
Unit tests for baseline consistency, determinism, and cross-method comparisons - Phase C11.5.
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


@pytest.fixture
def standard_packet():
    return ControlledDecisionPacket(
        packet_id="PACKET_TEST_001",
        retina=ModalityRecord("r1", "retina", 0.65, (0.5, 0.5), 0.85, 0.15, 0.90, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.45, (0.5, 0.5), 0.80, 0.20, 0.85, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.30, (0.5, 0.5), 0.75, 0.25, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )


def test_all_baselines_sum_to_one_and_bounded_risk(standard_packet):
    """Tests that every baseline B1–B6 produces weights summing to 1.0 and r_fusion in [0, 1]."""
    runner = FusionRunner()
    results = runner.evaluate_all_baselines(standard_packet)

    for b_id, res in results.items():
        assert res.status == "SUCCESS"
        assert abs(sum(res.weights.values()) - 1.0) < 1e-7
        assert 0.0 <= res.r_fusion <= 1.0
        for w in res.weights.values():
            assert 0.0 <= w <= 1.0


def test_all_baselines_determinism(standard_packet):
    """Tests that evaluating twice produces identical outputs (f(X) == f(X))."""
    runner = FusionRunner()
    res1 = runner.evaluate_all_baselines(standard_packet)
    res2 = runner.evaluate_all_baselines(standard_packet)

    for b_id in res1.keys():
        assert res1[b_id].weights == res2[b_id].weights
        assert res1[b_id].r_fusion == res2[b_id].r_fusion
        assert res1[b_id].routing_entropy == res2[b_id].routing_entropy
