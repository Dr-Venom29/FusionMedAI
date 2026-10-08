"""
FusionMedAI - Phase C11.12: ACARA-U Parameter & Weighting Sensitivity Analysis Package
"""

from .sensitivity_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    DELTA_PROVISIONAL,
    SEED,
    N_PACKETS,
    N_BOOTSTRAPS,
)
from .parameter_grid import (
    SensitivityConfigItem,
    get_sensitivity_parameter_grid,
    get_reference_config,
)
from .sensitivity_metrics import (
    compute_packet_metrics,
    compute_sensitivity_slopes,
    verify_logit_derivatives,
)
from .paired_bootstrap import compute_paired_bootstrap
from .sensitivity_runner import SensitivityExperimentRunner

__all__ = [
    "ALPHA_REF",
    "BETA_REF",
    "GAMMA_REF",
    "ETA_REF",
    "DELTA_PROVISIONAL",
    "SEED",
    "N_PACKETS",
    "N_BOOTSTRAPS",
    "SensitivityConfigItem",
    "get_sensitivity_parameter_grid",
    "get_reference_config",
    "compute_packet_metrics",
    "compute_sensitivity_slopes",
    "verify_logit_derivatives",
    "compute_paired_bootstrap",
    "SensitivityExperimentRunner",
]
