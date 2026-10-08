"""
verification/fusion/calibration/test_calibration_conditions.py
Tests calibration condition definitions, forward/inverse mappings, and probability bounds.
"""

import pytest
import numpy as np

from src.fusion.calibration.calibration_condition import (
    CALIBRATION_CONDITIONS,
    FROZEN_RETINA_TEMPERATURE,
    FROZEN_FOOT_VECTOR_WEIGHTS,
    FROZEN_FOOT_VECTOR_BIAS,
    uncalibrate_retina_probability,
    calibrate_retina_probability,
    uncalibrate_foot_probability,
    calibrate_foot_probability,
    uncalibrate_clinical_probability,
    calibrate_clinical_probability,
)


def test_calibration_conditions_count():
    assert len(CALIBRATION_CONDITIONS) == 6
    assert "B0_uncalibrated_uniform" in CALIBRATION_CONDITIONS
    assert "B2_uncalibrated_acarau" in CALIBRATION_CONDITIONS
    assert "B5_calibrated_acarau" in CALIBRATION_CONDITIONS


def test_retina_probability_roundtrip():
    original_p = (0.10, 0.20, 0.40, 0.20, 0.10)
    uncal = uncalibrate_retina_probability(original_p, temperature=FROZEN_RETINA_TEMPERATURE)
    recal = calibrate_retina_probability(uncal, temperature=FROZEN_RETINA_TEMPERATURE)

    assert len(uncal) == 5
    assert len(recal) == 5
    assert np.isclose(sum(uncal), 1.0, atol=1e-5)
    assert np.isclose(sum(recal), 1.0, atol=1e-5)
    assert np.allclose(original_p, recal, atol=1e-4)


def test_foot_probability_roundtrip():
    original_p = (0.15, 0.25, 0.35, 0.25)
    uncal = uncalibrate_foot_probability(original_p)
    recal = calibrate_foot_probability(uncal)

    assert len(uncal) == 4
    assert len(recal) == 4
    assert np.isclose(sum(uncal), 1.0, atol=1e-5)
    assert np.isclose(sum(recal), 1.0, atol=1e-5)
    assert np.allclose(original_p, recal, atol=1e-2)



def test_clinical_probability_roundtrip():
    original_p = (0.70, 0.30)
    uncal = uncalibrate_clinical_probability(original_p)
    recal = calibrate_clinical_probability(uncal)

    assert len(uncal) == 2
    assert len(recal) == 2
    assert np.isclose(sum(uncal), 1.0, atol=1e-5)
    assert np.isclose(sum(recal), 1.0, atol=1e-5)
    assert np.allclose(original_p, recal, atol=1e-4)
