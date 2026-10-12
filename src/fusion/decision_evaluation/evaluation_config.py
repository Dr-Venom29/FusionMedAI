"""
FusionMedAI - Phase C11.17: Final Decision-Level Evaluation Configuration
Frozen parameters, paths, thresholds, and schemas for the comprehensive
evaluation of ACARA-U and baseline policies across 5 evaluation dimensions.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path

# Repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Frozen Hyperparameters for ACARA-U Reference B6
ALPHA_REF: float = 1.0  # Dynamic confidence weight
BETA_REF: float = 1.5   # Historical reliability prior weight
GAMMA_REF: float = 1.0  # Predictive uncertainty penalty weight
ETA_REF: float = 0.5    # Input signal quality bonus weight
DELTA_FROZEN: float = 0.10  # DCRI uncertainty discount parameter (frozen in C11.13)

# Experimental Cohort and Reproducibility Parameters
N_PACKETS: int = 500
DATA_SEED: int = 115
BOOTSTRAP_ROUNDS: int = 1000
BOOTSTRAP_SEED: int = 7117
BOOTSTRAP_ALPHA: float = 0.05  # 95% Confidence Interval

# Decision Action Tiers (Hypothetical evaluation boundaries)
# Tau_1 = 0.20, Tau_2 = 0.40
TAU_1_DEFAULT: float = 0.20
TAU_2_DEFAULT: float = 0.40

TIER_0_ROUTINE: int = 0
TIER_1_ASSESSMENT: int = 1
TIER_2_ESCALATION: int = 2

TIER_NAMES: Dict[int, str] = {
    TIER_0_ROUTINE: "Routine Review (Tier 0)",
    TIER_1_ASSESSMENT: "Additional Assessment (Tier 1)",
    TIER_2_ESCALATION: "Escalation for Human Review (Tier 2)",
}

TIER_KEYS: Dict[int, str] = {
    TIER_0_ROUTINE: "routine_review",
    TIER_1_ASSESSMENT: "additional_assessment",
    TIER_2_ESCALATION: "escalation",
}

# Modality definitions
MODALITY_RETINA = "retina"
MODALITY_FOOT = "foot"
MODALITY_CLINICAL = "clinical"
MODALITIES = [MODALITY_RETINA, MODALITY_FOOT, MODALITY_CLINICAL]

# Baseline Policy IDs
POLICY_B1 = "B1"  # Reliability-Selected Unimodal
POLICY_B2 = "B2"  # Uniform Average
POLICY_B3 = "B3"  # Confidence Fusion
POLICY_B4 = "B4"  # Confidence + Reliability
POLICY_B5 = "B5"  # Confidence + Reliability - Uncertainty
POLICY_B6 = "B6"  # Full ACARA-U

ALL_POLICIES = [POLICY_B1, POLICY_B2, POLICY_B3, POLICY_B4, POLICY_B5, POLICY_B6]
BASELINE_COMPARATORS = [POLICY_B1, POLICY_B2, POLICY_B3, POLICY_B4, POLICY_B5]

POLICY_DESCRIPTIONS: Dict[str, str] = {
    POLICY_B1: "Reliability-Selected Unimodal (Winner-take-all by highest historical reliability prior)",
    POLICY_B2: "Uniform Average (Equal 1/|A| allocation across active channels)",
    POLICY_B3: "Confidence Fusion (Weighted by sample predictive confidence alpha=1.0)",
    POLICY_B4: "Confidence + Reliability (alpha=1.0, beta=1.5, gamma=0.0, eta=0.0)",
    POLICY_B5: "Confidence + Reliability - Uncertainty (alpha=1.0, beta=1.5, gamma=1.0, eta=0.0)",
    POLICY_B6: "Full ACARA-U Dynamic Router (alpha=1.0, beta=1.5, gamma=1.0, eta=0.5)",
}

# Modality Availability Regimes
REGIME_RFC = "RFC"
REGIME_RF = "RF"
REGIME_RC = "RC"
REGIME_FC = "FC"
REGIME_R = "R"
REGIME_F = "F"
REGIME_C = "C"
REGIME_EMPTY = "EMPTY"

VALID_REGIMES = [REGIME_RFC, REGIME_RF, REGIME_RC, REGIME_FC, REGIME_R, REGIME_F, REGIME_C]
ALL_REGIMES = VALID_REGIMES + [REGIME_EMPTY]

REGIME_MODALITY_MAP: Dict[str, List[str]] = {
    REGIME_RFC: [MODALITY_RETINA, MODALITY_FOOT, MODALITY_CLINICAL],
    REGIME_RF: [MODALITY_RETINA, MODALITY_FOOT],
    REGIME_RC: [MODALITY_RETINA, MODALITY_CLINICAL],
    REGIME_FC: [MODALITY_FOOT, MODALITY_CLINICAL],
    REGIME_R: [MODALITY_RETINA],
    REGIME_F: [MODALITY_FOOT],
    REGIME_C: [MODALITY_CLINICAL],
    REGIME_EMPTY: [],
}

# Input Degradation Severities
DEGRADATION_D0 = "D0"  # Clean reference
DEGRADATION_D1 = "D1"  # Mild degradation
DEGRADATION_D2 = "D2"  # Moderate degradation
DEGRADATION_D3 = "D3"  # Severe degradation

ALL_DEGRADATIONS = [DEGRADATION_D0, DEGRADATION_D1, DEGRADATION_D2, DEGRADATION_D3]

# Upstream Artifact Dependencies
UPSTREAM_PACKET_MANIFEST = REPO_ROOT / "experiments" / "fusion" / "dcri" / "packet_manifest.json"
UPSTREAM_PACKET_MANIFEST_SIZE_BYTES: int = 821319
UPSTREAM_PACKET_MANIFEST_SHA256: str = "1ff37fbeeba33feae7bf14d57d0df4a10df48ac789edeedf9dbd2fa8121382bb"

# Output Directories
EXP_DIR = REPO_ROOT / "experiments" / "fusion" / "decision_evaluation"
RESULTS_DIR = EXP_DIR / "results"
REPORTS_DIR = EXP_DIR / "reports"
RESEARCH_DIR = REPO_ROOT / "research" / "fusion" / "Volume_17_Decision_Level_Evaluation"


def validate_evaluation_config() -> None:
    """
    Validates evaluation configuration, thresholds, and upstream dependencies.
    Raises ValueError or FileNotFoundError if any precondition is violated.
    """
    import hashlib
    import math

    if not (0.0 < TAU_1_DEFAULT < TAU_2_DEFAULT < 1.0):
        raise ValueError(
            f"Invalid decision thresholds: TAU_1={TAU_1_DEFAULT}, TAU_2={TAU_2_DEFAULT}. "
            "Must satisfy 0 < TAU_1 < TAU_2 < 1."
        )

    for name, val in [
        ("ALPHA_REF", ALPHA_REF),
        ("BETA_REF", BETA_REF),
        ("GAMMA_REF", GAMMA_REF),
        ("ETA_REF", ETA_REF),
    ]:
        if not math.isfinite(val) or val < 0.0:
            raise ValueError(f"Invalid coefficient {name}={val}. Must be finite and non-negative.")

    if not math.isfinite(DELTA_FROZEN) or not (0.0 <= DELTA_FROZEN <= 1.0):
        raise ValueError(f"DELTA_FROZEN must be finite and in [0, 1], got {DELTA_FROZEN}")

    if N_PACKETS <= 0:
        raise ValueError(f"N_PACKETS must be positive, got {N_PACKETS}")

    if BOOTSTRAP_ROUNDS <= 1:
        raise ValueError(f"BOOTSTRAP_ROUNDS must be > 1, got {BOOTSTRAP_ROUNDS}")

    if not (0.0 < BOOTSTRAP_ALPHA < 0.5):
        raise ValueError(f"BOOTSTRAP_ALPHA must be in (0, 0.5), got {BOOTSTRAP_ALPHA}")

    if not isinstance(BOOTSTRAP_SEED, int) or BOOTSTRAP_SEED < 0:
        raise ValueError(f"BOOTSTRAP_SEED must be a non-negative integer, got {BOOTSTRAP_SEED}")

    if not UPSTREAM_PACKET_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Upstream packet manifest not found at {UPSTREAM_PACKET_MANIFEST}"
        )

    actual_size = UPSTREAM_PACKET_MANIFEST.stat().st_size
    if actual_size != UPSTREAM_PACKET_MANIFEST_SIZE_BYTES:
        raise ValueError(
            f"Upstream packet manifest size mismatch: expected {UPSTREAM_PACKET_MANIFEST_SIZE_BYTES} bytes, "
            f"found {actual_size} bytes."
        )

    h = hashlib.sha256()
    with open(UPSTREAM_PACKET_MANIFEST, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual_sha = h.hexdigest()

    if actual_sha != UPSTREAM_PACKET_MANIFEST_SHA256:
        raise ValueError(
            f"Upstream packet manifest SHA-256 mismatch: expected {UPSTREAM_PACKET_MANIFEST_SHA256}, "
            f"found {actual_sha}."
        )

