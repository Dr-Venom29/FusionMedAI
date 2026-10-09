"""
FusionMedAI - Phase C11.13: DCRI Global Uncertainty Penalty Selection Configuration
Defines the pre-specified candidate delta grid, frozen upstream constants,
hard validity constraints, and statistical evaluation parameters.
"""

from typing import Tuple, Dict, Any, List

# Frozen Upstream Constants (Phase C11.12 Reference Configuration Theta_0)
ALPHA_REF: float = 1.0
BETA_REF: float = 1.5
GAMMA_REF: float = 1.0
ETA_REF: float = 0.5

# Historical / Provisional Reference (for comparison purposes only)
DELTA_PROVISIONAL_HISTORICAL: float = 0.20

# Pre-specified Candidate Delta Grid
# 11 points spanning no penalty (0.00) to full penalty (1.00),
# with dense resolution in the candidate region [0.00, 0.30].
CANDIDATE_DELTA_GRID: Tuple[float, ...] = (
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.75,
    1.00,
)

# Experimental Cohort and Statistical Parameters
N_PACKETS: int = 500
SEED: int = 115
N_BOOTSTRAPS: int = 1000
BOOTSTRAP_CI_ALPHA: float = 0.05  # 95% Confidence Interval

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

# Hard Validity Constraints
MAX_ACCEPTABLE_FLOAT_TOLERANCE: float = 1e-12
RESIDUAL_ERROR_TOLERANCE: float = 1e-10
MAX_THEORETICAL_M: int = 3  # Max modalities available

# Hard Behavioral Feasibility Constraints (Tier 2 Filter)
MAX_TOLERABLE_NEGATIVE_RATE: float = 0.35  # Maximum allowable negative DCRI rate in full cohort (35%)
MAX_TOLERABLE_MEAN_PENALTY_RATIO: float = 0.60  # Maximum allowable penalty-to-base ratio (60%)
MIN_RANK_STABILITY_SPEARMAN: float = 0.90  # Hard floor for Spearman rank stability (0.90)

# Preferred Parsimonious Selection Criteria (Tier 4 Rule)
# A candidate is considered to provide a substantial, non-trivial uncertainty discount without excess distortion if:
MIN_MEANINGFUL_PENALTY_RATIO: float = 0.20  # Mean penalty must be at least 20% of mean base fused risk
MAX_PREFERRED_NEGATIVE_RATE: float = 0.10   # Negative DCRI rate must remain <= 10%
MIN_PREFERRED_RANK_STABILITY: float = 0.98  # Spearman rank correlation must remain >= 0.98

