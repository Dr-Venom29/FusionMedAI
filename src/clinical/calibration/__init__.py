"""
Clinical Probability Calibration & Risk Reliability Package (Phase C7).
"""

from src.clinical.calibration.metrics import (
    compute_calibration_curve,
    compute_ece,
    compute_mce,
    compute_calibration_slope_intercept,
    evaluate_calibration_metrics,
)
from src.clinical.calibration.calibrator import (
    BaseClinicalCalibrator,
    PlattCalibrator,
    IsotonicCalibrator,
    BetaCalibrator,
)
from src.clinical.calibration.dca import (
    compute_net_benefit,
    compute_operating_threshold_metrics,
)
from src.clinical.calibration.subgroup_calibration import (
    evaluate_subgroup_calibration,
)

__all__ = [
    "compute_calibration_curve",
    "compute_ece",
    "compute_mce",
    "compute_calibration_slope_intercept",
    "evaluate_calibration_metrics",
    "BaseClinicalCalibrator",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "BetaCalibrator",
    "compute_net_benefit",
    "compute_operating_threshold_metrics",
    "evaluate_subgroup_calibration",
]
