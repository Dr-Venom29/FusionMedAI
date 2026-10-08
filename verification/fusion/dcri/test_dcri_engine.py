"""
Unit tests for DCRIEngine orchestration and packet evaluation (Phase C11.6).
"""

import pytest
import math
from src.fusion.baselines.decision_packet import ModalityRecord, ControlledDecisionPacket
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)
from src.fusion.dcri.dcri_engine import DCRIEngine
from src.fusion.dcri.dcri_result import DCRIResult


@pytest.fixture
def sample_packet():
    return ControlledDecisionPacket(
        packet_id="PACKET_TEST_001",
        retina=ModalityRecord("r1", "retina", 0.72, (0.5, 0.5), 0.85, 0.10, 0.95, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.41, (0.5, 0.5), 0.70, 0.20, 0.90, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.62, (0.5, 0.5), 0.65, 0.15, 0.85, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )


def test_dcri_engine_initialization():
    engine = DCRIEngine()
    assert len(engine.delta_grid) == 6
    assert engine.DEFAULT_DELTA == 0.2


def test_evaluate_packet_delta_zero_invariant(sample_packet):
    engine = DCRIEngine()
    res = engine.evaluate_packet(sample_packet, delta=0.0)
    assert res.status == "SUCCESS"
    assert res.uncertainty_penalty == 0.0
    assert abs(res.dcri - res.r_fusion) < 1e-7


def test_evaluate_packet_dcri_equation_exact(sample_packet):
    engine = DCRIEngine()
    res = engine.evaluate_packet(sample_packet, delta=0.20)
    assert res.status == "SUCCESS"
    expected_penalty = 0.20 * (0.10 + 0.20 + 0.15)  # 0.20 * 0.45 = 0.09
    assert abs(res.uncertainty_penalty - expected_penalty) < 1e-6
    assert abs(res.dcri - (res.r_fusion - expected_penalty)) < 1e-6


def test_evaluate_packet_across_grid(sample_packet):
    engine = DCRIEngine()
    grid_results = engine.evaluate_packet_across_grid(sample_packet)
    assert len(grid_results) == 6
    for d, res in grid_results.items():
        assert isinstance(res, DCRIResult)
        assert res.delta == d
        assert res.status == "SUCCESS"


def test_evaluate_cohort_statistics_shape(sample_packet):
    engine = DCRIEngine()
    cohort_res = engine.evaluate_cohort([sample_packet, sample_packet])
    assert cohort_res["num_packets"] == 2
    assert cohort_res["num_success"] == 2
    assert "r_fusion_statistics" in cohort_res
    assert "delta_sensitivity" in cohort_res
