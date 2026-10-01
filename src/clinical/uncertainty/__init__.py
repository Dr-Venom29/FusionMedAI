"""
Clinical Prediction Uncertainty Package for FusionMedAI (Phase C8).
"""

from .bootstrap_ensemble import BootstrapCatBoostEnsemble
from .evaluation import (
    evaluate_error_detection,
    evaluate_risk_coverage,
    evaluate_threshold_uncertainty_tiers,
    evaluate_subgroup_uncertainty,
    evaluate_convergence,
)

__all__ = [
    "BootstrapCatBoostEnsemble",
    "evaluate_error_detection",
    "evaluate_risk_coverage",
    "evaluate_threshold_uncertainty_tiers",
    "evaluate_subgroup_uncertainty",
    "evaluate_convergence",
]
