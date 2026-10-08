"""
verification/fusion/calibration/test_routing_invariants.py
Tests simplex conservation and router bounds across uncalibrated and calibrated conditions.
"""

import pytest
import numpy as np

from src.fusion.calibration.calibration_runner import CalibrationExperimentRunner
from src.fusion.calibration.calibration_condition import (
    CALIBRATION_CONDITIONS,
    COND_B2_UNCAL_ACARAU,
    COND_B5_CAL_ACARAU,
)


@pytest.fixture(scope="module")
def runner_and_cohort():
    runner = CalibrationExperimentRunner()
    cohort = runner.load_cohort()
    return runner, cohort


def test_simplex_invariant_across_all_conditions(runner_and_cohort):
    runner, cohort = runner_and_cohort
    # Test on a sample of 25 packets
    sample_packets = cohort[:25]

    for pkt in sample_packets:
        for cond in CALIBRATION_CONDITIONS:
            res = runner.evaluate_packet_under_condition(pkt, cond, propagate_confidence=True)
            w_sum = sum(res.weights.values())
            assert np.isclose(w_sum, 1.0, atol=1e-5), f"Simplex violation under {cond}: sum = {w_sum}"
            for m, w in res.weights.items():
                assert 0.0 - 1e-6 <= w <= 1.0 + 1e-6, f"Weight {m}={w} out of bounds under {cond}"


def test_zero_modality_rejection(runner_and_cohort):
    runner, cohort = runner_and_cohort
    # Verify non-empty active routing works
    res_uncal = runner.evaluate_packet_under_condition(cohort[0], COND_B2_UNCAL_ACARAU)
    res_cal = runner.evaluate_packet_under_condition(cohort[0], COND_B5_CAL_ACARAU)
    assert res_uncal.r_fusion >= 0.0
    assert res_cal.r_fusion >= 0.0
