"""
FusionMedAI - Phase C11.16: ACARA-U Component Ablation Configuration
Defines frozen reference parameters, ablation variants (A0-A6),
decision policy thresholds, and statistical constants.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path


# =============================================================================
# Upstream Frozen Reference Hyperparameters (Phase C11.4, C11.13, C11.15 Locked)
# =============================================================================
ALPHA_REF: float = 1.0  # Confidence term coefficient
BETA_REF: float = 1.5   # Reliability term coefficient
GAMMA_REF: float = 1.0  # Predictive uncertainty penalty coefficient
ETA_REF: float = 0.5    # Input signal quality bonus coefficient
DELTA_FROZEN: float = 0.10  # DCRI uncertainty discount coefficient (locked in C11.13)

# Experimental Cohort and Statistical Parameters
N_PACKETS: int = 500
SEED: int = 115
BOOTSTRAP_ROUNDS: int = 1000
BOOTSTRAP_SEED: int = 7116
BOOTSTRAP_ALPHA: float = 0.05  # 95% Confidence Interval

# Hypothetical Decision Action Taxonomy (3 Tiers under Tau_1=0.20, Tau_2=0.40)
# NOTE: These tiers define an evaluation-only controlled policy for analyzing score sensitivity.
# They do not constitute clinical recommendations, validated triage protocols, or medical advice.
TAU_1_DEFAULT: float = 0.20
TAU_2_DEFAULT: float = 0.40

ACTION_ROUTINE_REVIEW: int = 0
ACTION_ADDITIONAL_ASSESSMENT: int = 1
ACTION_ESCALATION: int = 2

ACTION_NAMES: Dict[int, str] = {
    ACTION_ROUTINE_REVIEW: "Routine Review (Hypothetical Tier 0)",
    ACTION_ADDITIONAL_ASSESSMENT: "Additional Assessment (Hypothetical Tier 1)",
    ACTION_ESCALATION: "Escalation for Review (Hypothetical Tier 2)",
}

ACTION_KEYS: Dict[int, str] = {
    ACTION_ROUTINE_REVIEW: "routine_review",
    ACTION_ADDITIONAL_ASSESSMENT: "additional_assessment",
    ACTION_ESCALATION: "escalation",
}

# Modality Taxonomy
MODALITIES: Tuple[str, ...] = ("retina", "foot", "clinical")

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

REGIME_MODALITY_MAP: Dict[str, Tuple[str, ...]] = {
    "R": ("retina",),
    "F": ("foot",),
    "C": ("clinical",),
    "RF": ("retina", "foot"),
    "RC": ("retina", "clinical"),
    "FC": ("foot", "clinical"),
    "RFC": ("retina", "foot", "clinical"),
    "EMPTY": (),
}


# =============================================================================
# Ablation Variant Specifications
# =============================================================================
@dataclass(frozen=True)
class AblationVariantSpec:
    """
    Specification of an experimental ablation variant configuration.
    
    NOTE: Ablation specifications describe controlled diagnostic configurations
    for component-level attribution analysis and do not imply production readiness,
    hyperparameter suitability, or clinical deployment candidate status.
    """
    variant_id: str
    name: str
    description: str
    alpha: float
    beta: float
    gamma: float
    eta: float
    delta: float
    ablated_component: str
    is_reference: bool = False
    is_negative_control: bool = False
    is_diagnostic_baseline: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "description": self.description,
            "coefficients": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "eta": self.eta,
            },
            "delta": self.delta,
            "ablated_component": self.ablated_component,
            "is_reference": self.is_reference,
            "is_negative_control": self.is_negative_control,
            "is_diagnostic_baseline": self.is_diagnostic_baseline,
        }


# Canonical Ablation Matrix (A0 to A6 + A_UNIFORM)
VARIANTS: Dict[str, AblationVariantSpec] = {
    "A0": AblationVariantSpec(
        variant_id="A0",
        name="Full B6 Reference",
        description="Full frozen ACARA-U router (alpha=1.0, beta=1.5, gamma=1.0, eta=0.5) with delta*=0.10.",
        alpha=ALPHA_REF,
        beta=BETA_REF,
        gamma=GAMMA_REF,
        eta=ETA_REF,
        delta=DELTA_FROZEN,
        ablated_component="none",
        is_reference=True,
    ),
    "A1": AblationVariantSpec(
        variant_id="A1",
        name="No Confidence Contribution",
        description="Ablates confidence weighting: alpha=0.0, retaining beta=1.5, gamma=1.0, eta=0.5, delta=0.10.",
        alpha=0.0,
        beta=BETA_REF,
        gamma=GAMMA_REF,
        eta=ETA_REF,
        delta=DELTA_FROZEN,
        ablated_component="confidence",
    ),
    "A2": AblationVariantSpec(
        variant_id="A2",
        name="No Reliability Contribution",
        description="Ablates global reliability prior: beta=0.0, retaining alpha=1.0, gamma=1.0, eta=0.5, delta=0.10.",
        alpha=ALPHA_REF,
        beta=0.0,
        gamma=GAMMA_REF,
        eta=ETA_REF,
        delta=DELTA_FROZEN,
        ablated_component="reliability",
    ),
    "A3": AblationVariantSpec(
        variant_id="A3",
        name="No Uncertainty Contribution",
        description="Ablates predictive uncertainty penalty in router: gamma=0.0, retaining alpha=1.0, beta=1.5, eta=0.5, delta=0.10.",
        alpha=ALPHA_REF,
        beta=BETA_REF,
        gamma=0.0,
        eta=ETA_REF,
        delta=DELTA_FROZEN,
        ablated_component="uncertainty_router",
    ),
    "A4": AblationVariantSpec(
        variant_id="A4",
        name="No Quality Contribution",
        description="Ablates input quality bonus in router: eta=0.0, retaining alpha=1.0, beta=1.5, gamma=1.0, delta=0.10.",
        alpha=ALPHA_REF,
        beta=BETA_REF,
        gamma=GAMMA_REF,
        eta=0.0,
        delta=DELTA_FROZEN,
        ablated_component="quality",
    ),
    "A5": AblationVariantSpec(
        variant_id="A5",
        name="No DCRI Uncertainty Discount",
        description="Ablates downstream uncertainty discounting: delta=0.0, retaining frozen B6 routing coefficients.",
        alpha=ALPHA_REF,
        beta=BETA_REF,
        gamma=GAMMA_REF,
        eta=ETA_REF,
        delta=0.0,
        ablated_component="dcri_discount",
    ),
    "A_UNIFORM": AblationVariantSpec(
        variant_id="A_UNIFORM",
        name="Uniform Active Baseline",
        description="Diagnostic normalization baseline with all four routing contributions set to zero (alpha=beta=gamma=eta=0.0, delta=0.10).",
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        eta=0.0,
        delta=DELTA_FROZEN,
        ablated_component="all_router_contributions",
        is_diagnostic_baseline=True,
    ),
    "A6": AblationVariantSpec(
        variant_id="A6",
        name="Availability-Mask Negative Control",
        description="Diagnostic fault injection: deliberately violates availability masking by assigning non-zero weight to an inactive channel.",
        alpha=ALPHA_REF,
        beta=BETA_REF,
        gamma=GAMMA_REF,
        eta=ETA_REF,
        delta=DELTA_FROZEN,
        ablated_component="availability_mask_violation",
        is_negative_control=True,
    ),
}

# Primary Candidate Variants Evaluated for Scientific Attribution (A0 - A5)
PRIMARY_VARIANTS: Tuple[str, ...] = ("A0", "A1", "A2", "A3", "A4", "A5")
ALL_EVALUATED_VARIANTS: Tuple[str, ...] = ("A0", "A1", "A2", "A3", "A4", "A5", "A_UNIFORM")
