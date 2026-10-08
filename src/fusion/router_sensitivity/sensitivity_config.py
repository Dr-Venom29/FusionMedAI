"""
FusionMedAI - Phase C11.12: Sensitivity Analysis Configuration & Frozen Locks
Defines frozen experimental constants, regime taxonomy, and boundary constraints.
"""

from typing import Dict, Any, Tuple

# Frozen Reference Router Coefficients (Phase C11.4 / C11.8 reference)
ALPHA_REF: float = 1.0       # Confidence term
BETA_REF: float = 1.5        # Empirical reliability term
GAMMA_REF: float = 1.0       # Uncertainty penalty term
ETA_REF: float = 0.5         # Quality bonus term

# Provisional DCRI uncertainty weighting (Phase C11.6 provisional value)
DELTA_PROVISIONAL: float = 0.20

# Cohort & Statistical Sampling Constants
SEED: int = 115
N_PACKETS: int = 500
N_BOOTSTRAPS: int = 1000
CONFIDENCE_LEVEL: float = 0.95

# Modality & Regime Definitions
VALID_MODALITIES: Tuple[str, ...] = ("retina", "foot", "clinical")

REGIMES_TAXONOMY: Dict[str, Tuple[str, ...]] = {
    "R": ("retina",),
    "F": ("foot",),
    "C": ("clinical",),
    "RF": ("retina", "foot"),
    "RC": ("retina", "clinical"),
    "FC": ("foot", "clinical"),
    "RFC": ("retina", "foot", "clinical"),
}

# Frozen Upstream Modality Reliability Priors (Phase C11.3)
FROZEN_RELIABILITY_MAP: Dict[str, float] = {
    "retina": 0.929956,
    "foot": 0.922266,
    "clinical": 0.825382,
}

# Domain & Protocol Boundary Constraints
COEFFICIENT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "alpha": (0.0, 5.0),
    "beta": (0.0, 5.0),
    "gamma": (0.0, 5.0),
    "eta": (0.0, 5.0),
}

# Stability Categories
CATEGORY_STABLE: str = "STABLE"
CATEGORY_SENSITIVE: str = "SENSITIVE"
CATEGORY_UNSTABLE: str = "UNSTABLE"
