"""
Clinical Phase C9: Robustness, Fairness & Distribution Shift Analysis module.
"""

from .metrics import (
    evaluate_full_robustness_profile,
    compute_calibration_slope_intercept,
    compute_ece,
    compute_selective_prediction_metrics,
)

__all__ = [
    "evaluate_full_robustness_profile",
    "compute_calibration_slope_intercept",
    "compute_ece",
    "compute_selective_prediction_metrics",
]
