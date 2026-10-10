"""
FusionMedAI - Phase C11.15: Router Sanity & Monotonicity Configuration
Centralizes frozen parameters, tolerances, acceptance criteria identifiers,
and test matrix configurations for ACARA-U router sanity analysis.
"""

from typing import Dict, Tuple, List, Any
from pathlib import Path

# =============================================================================
# Frozen Parameters & Reference Constants
# =============================================================================

# Frozen Reference Router Coefficients Theta_0 (Sealed in C11.12)
ALPHA_REF: float = 1.0   # Confidence weighting
BETA_REF: float = 1.5    # Validation reliability weighting
GAMMA_REF: float = 1.0   # Predictive uncertainty penalty
ETA_REF: float = 0.5     # Input signal quality bonus

ROUTER_COEFFS_REF: Dict[str, float] = {
    "alpha": ALPHA_REF,
    "beta": BETA_REF,
    "gamma": GAMMA_REF,
    "eta": ETA_REF,
}

# Frozen DCRI Multiplier (Sealed in C11.13)
DELTA_FROZEN: float = 0.10

# Reference Controlled Cohort Metadata (Sealed in Phases C11.1-C11.14)
# Note: Phase C11.15 records this metadata to identify the upstream context
# on which Theta_0 and delta* were selected and frozen. Specific monotonicity,
# invariant, and contract evaluations in C11.15 are performed across deterministic
# parameter grids, boundary configurations, and mathematical kernel fixtures.
BENCHMARK_COHORT_N: int = 500
BENCHMARK_COHORT_SEED: int = 115
N_PACKETS: int = BENCHMARK_COHORT_N
SEED: int = BENCHMARK_COHORT_SEED

# Numerical Tolerances
FLOAT_TOLERANCE: float = 1e-12
SIMPLEX_TOLERANCE: float = 1e-10
STRICT_MONOTONIC_EPS: float = 1e-7
SHIFT_INVARIANCE_TOLERANCE: float = 1e-12

# Modality Channel Identifiers
MODALITIES: Tuple[str, ...] = ("retina", "foot", "clinical")
MODALITY_SYMBOLS: Dict[str, str] = {
    "retina": "R",
    "foot": "F",
    "clinical": "C",
}

# Modality Availability Regimes
ACTIVE_REGIMES: Tuple[str, ...] = ("R", "F", "C", "RF", "RC", "FC", "RFC")
ALL_REGIMES: Tuple[str, ...] = ("R", "F", "C", "RF", "RC", "FC", "RFC", "EMPTY")

# Test Matrix Input Grid for Monotonicity
INTERIOR_GRID: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
BOUNDARY_GRID: Tuple[float, ...] = (0.0, 1.0)
PERTURBATION_STEPS: Tuple[float, ...] = (0.05, 0.10, 0.20)

# Acceptance Criteria S15 Identifiers
ACCEPTANCE_CRITERIA: Dict[str, str] = {
    "S15-01": "Frozen Reference Configuration Integrity (Theta_0, delta=0.10)",
    "S15-02": "Confidence Monotonicity (d w_i / d C_i > 0 for active multi-channel)",
    "S15-03": "Reliability Monotonicity (d w_i / d R_i > 0 in isolated kernel fixture)",
    "S15-04": "Uncertainty Monotonicity (d w_i / d U_i < 0 for active multi-channel)",
    "S15-05": "Quality Monotonicity (d w_i / d Q_i > 0 for active multi-channel)",
    "S15-06": "Simplex Conservation (sum w_i = 1.0 within 1e-10, w_i >= 0)",
    "S15-07": "Availability Masking (A_i = 0 => w_i = 0.0 exact)",
    "S15-08": "Softmax Shift Invariance (z_i + c => w_i invariant within 1e-12)",
    "S15-09": "Reference Normalization Kernel Permutation Invariance (Order Independence)",
    "S15-10": "Numerical Stability & Input Validation Rejection (extreme logits & invalid inputs)",
    "S15-11": "Empty Modality Fail-Closed Handling (NO_MODALITY_AVAILABLE)",
    "S15-12": "Masked-Value Corruption Invariance (inactive channel perturbations do not affect active weights)",
    "S15-13": "Pipeline Stage Decoupling & Downstream Production Contract (Router -> Fused Risk -> DCRI)",
    "S15-14": "Regression Invariance (Executed C11.14 Suite: 22/22 tests passed)",
    "S15-15": "Verifier Mutation Fault Detection (5/5 injected fault cases caught)",
}
