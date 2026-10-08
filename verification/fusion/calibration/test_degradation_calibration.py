"""
verification/fusion/calibration/test_degradation_calibration.py
Tests calibration behavior under progressive input degradation ladders D0–D3.
"""

import pytest
import numpy as np

from src.fusion.calibration.calibration_runner import CalibrationExperimentRunner
from src.fusion.degradation.degradation_spec import (
    SEVERITY_D0_CLEAN,
    SEVERITY_D3_SEVERE,
    OP_RETINA_BLUR,
    OP_FOOT_BLUR,
    OP_CLINICAL_RANDOM_MASK,
)


@pytest.fixture(scope="module")
def degradation_results():
    runner = CalibrationExperimentRunner()
    cohort = runner.load_cohort()[:20]  # Fast subset for unit test
    deg_res = runner.run_degradation_comparison(cohort, propagate_confidence=True)
    return deg_res


def test_degradation_ladder_coverage(degradation_results):
    assert OP_RETINA_BLUR in degradation_results
    assert OP_FOOT_BLUR in degradation_results
    assert OP_CLINICAL_RANDOM_MASK in degradation_results

    for op, sev_dict in degradation_results.items():
        assert len(sev_dict) == 4
        assert SEVERITY_D0_CLEAN in sev_dict
        assert SEVERITY_D3_SEVERE in sev_dict


def test_authority_attenuation_direction_under_degradation(degradation_results):
    r_res = degradation_results[OP_RETINA_BLUR]
    w_clean = r_res[SEVERITY_D0_CLEAN]["mean_weight_calibrated"]
    w_severe = r_res[SEVERITY_D3_SEVERE]["mean_weight_calibrated"]
    # Authority should attenuate under severe blur
    assert w_severe < w_clean
