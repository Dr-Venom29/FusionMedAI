"""
Unit tests for delta sensitivity and delta grid invariants (Phase C11.6).
"""

import pytest
import numpy as np
from src.fusion.baselines.decision_packet import ModalityRecord, ControlledDecisionPacket
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)
from src.fusion.dcri.dcri_engine import DCRIEngine


@pytest.fixture
def test_packet():
    return ControlledDecisionPacket(
        packet_id="PACKET_DELTA_TEST",
        retina=ModalityRecord("r1", "retina", 0.60, (0.5, 0.5), 0.85, 0.15, 0.90, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.45, (0.5, 0.5), 0.75, 0.25, 0.85, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.30, (0.5, 0.5), 0.70, 0.20, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )


def test_delta_grid_monotonicity(test_packet):
    """Verifies that DCRI strictly decreases as delta increases for non-zero uncertainty."""
    engine = DCRIEngine()
    grid = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    results = engine.evaluate_packet_across_grid(test_packet, grid=grid)

    prev_dcri = float("inf")
    for d in grid:
        res = results[d]
        assert res.dcri < prev_dcri, f"Monotonicity violation at delta={d}: {res.dcri} >= {prev_dcri}"
        prev_dcri = res.dcri


def test_delta_dcri_equals_minus_delta_u_sum(test_packet):
    """Verifies Delta DCRI = DCRI_delta - R_fusion == -delta * U_sum exactly."""
    engine = DCRIEngine()
    grid = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    results = engine.evaluate_packet_across_grid(test_packet, grid=grid)

    r_fusion = results[0.0].r_fusion
    u_sum = results[0.0].u_sum

    for d in grid:
        res = results[d]
        shift = res.dcri - r_fusion
        expected_shift = -d * u_sum
        assert abs(shift - expected_shift) < 1e-6


def test_delta_zero_no_penalty(test_packet):
    engine = DCRIEngine()
    res = engine.evaluate_packet(test_packet, delta=0.0)
    assert res.uncertainty_penalty == 0.0
    assert res.dcri == res.r_fusion
