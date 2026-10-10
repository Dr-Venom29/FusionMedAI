"""
FusionMedAI - Phase C11.14: DCRI Decision Policy Analysis Configuration
Defines frozen upstream parameters, pre-specified policy actions, threshold grids,
and statistical evaluation constants.
"""

from typing import Tuple, Dict, Any, List

# Frozen Upstream Constants (Phase C11.12 & C11.13 Locked)
ALPHA_REF: float = 1.0
BETA_REF: float = 1.5
GAMMA_REF: float = 1.0
ETA_REF: float = 0.5
DELTA_FROZEN: float = 0.10  # Formally selected and frozen in C11.13

# Historical Provisional (for reference / comparison only)
DELTA_PROVISIONAL_HISTORICAL: float = 0.20

# Experimental Cohort and Statistical Parameters
N_PACKETS: int = 500
SEED: int = 115
N_BOOTSTRAPS: int = 1000
BOOTSTRAP_CI_ALPHA: float = 0.05  # 95% Confidence Interval

# Operational Decision Action Taxonomy (3 Categories)
# Note: These are operational workload/routing placeholders within the controlled benchmark.
ACTION_ROUTINE_REVIEW: int = 0
ACTION_ADDITIONAL_ASSESSMENT: int = 1
ACTION_ESCALATION: int = 2

ACTION_NAMES: Dict[int, str] = {
    ACTION_ROUTINE_REVIEW: "Routine Review",
    ACTION_ADDITIONAL_ASSESSMENT: "Additional Assessment",
    ACTION_ESCALATION: "Escalation for Human Review",
}

ACTION_KEYS: Dict[int, str] = {
    ACTION_ROUTINE_REVIEW: "routine_review",
    ACTION_ADDITIONAL_ASSESSMENT: "additional_assessment",
    ACTION_ESCALATION: "escalation",
}

# Standard Reference Operating Thresholds (Pre-specified primary baseline)
TAU_1_DEFAULT: float = 0.20  # Low-risk boundary
TAU_2_DEFAULT: float = 0.40  # High-risk / Escalation boundary

# Pre-specified Systematic Threshold Grid for Sensitivity Sweeps
# (Tau_1 in [0.10, 0.30], Tau_2 in [0.35, 0.60], strictly Tau_1 < Tau_2)
TAU_1_GRID: Tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30)
TAU_2_GRID: Tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.60)

# Modality Regimes Taxonomy (7 Active Regimes + EMPTY Fail-Closed)
ACTIVE_REGIMES: Tuple[str, ...] = (
    "R",
    "F",
    "C",
    "RF",
    "RC",
    "FC",
    "RFC",
)
ALL_REGIMES: Tuple[str, ...] = ACTIVE_REGIMES + ("EMPTY",)

# Modalities List
MODALITIES: Tuple[str, ...] = ("retina", "foot", "clinical")

# Hard Invariant Tolerances
FLOAT_TOLERANCE: float = 1e-12
RESIDUAL_TOLERANCE: float = 1e-10
MAX_THEORETICAL_M: int = 3
