from src.foot.uncertainty.mc_dropout import enable_foot_mc_dropout
from src.foot.uncertainty.metrics import (
    compute_entropy,
    compute_mc_uncertainty_metrics,
    compute_error_detection_metrics
)
from src.foot.uncertainty.risk_coverage import compute_risk_coverage_curve, evaluate_risk_coverage
from src.foot.uncertainty.convergence import run_mc_convergence_study
from src.foot.uncertainty.evaluation import (
    evaluate_classwise_uncertainty,
    evaluate_boundary_uncertainty,
    extract_high_uncertainty_samples,
    generate_uncertainty_explainability_overlays
)
from src.foot.uncertainty.run_uncertainty_experiment import run_uncertainty_experiment

__all__ = [
    "enable_foot_mc_dropout",
    "compute_entropy",
    "compute_mc_uncertainty_metrics",
    "compute_error_detection_metrics",
    "compute_risk_coverage_curve",
    "evaluate_risk_coverage",
    "run_mc_convergence_study",
    "evaluate_classwise_uncertainty",
    "evaluate_boundary_uncertainty",
    "extract_high_uncertainty_samples",
    "generate_uncertainty_explainability_overlays",
    "run_uncertainty_experiment"
]
