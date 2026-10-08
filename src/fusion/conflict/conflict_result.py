"""
src/fusion/conflict/conflict_result.py
Phase C11.7: Cross-Modality Conflict & Discordance Analysis

Defines immutable, self-validating data contracts for pairwise modality disagreement,
weighted consensus dispersion, and operational conflict classification.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass(frozen=True)
class PairwiseRecord:
    """
    Immutable record of directional and absolute risk disagreement between two modalities.
    """
    modality_a: str
    modality_b: str
    risk_a: float
    risk_b: float
    signed_difference: float      # r_a - r_b
    absolute_difference: float    # |r_a - r_b|
    higher_risk_modality: str     # modality name or "tied"
    lower_risk_modality: str      # modality name or "tied"
    weight_a: float
    weight_b: float
    dominant_authority_modality: str  # modality name with higher weight or "tied"
    uncertainty_a: float
    uncertainty_b: float
    reliability_a: float
    reliability_b: float
    reliability_authority_aligned: bool  # True if higher reliability modality gets >= weight

    def __post_init__(self) -> None:
        # Bounds validation
        for name, val in [
            ("risk_a", self.risk_a),
            ("risk_b", self.risk_b),
            ("weight_a", self.weight_a),
            ("weight_b", self.weight_b),
            ("uncertainty_a", self.uncertainty_a),
            ("uncertainty_b", self.uncertainty_b),
            ("reliability_a", self.reliability_a),
            ("reliability_b", self.reliability_b),
        ]:
            if math.isnan(val) or math.isinf(val) or not (0.0 <= val <= 1.0):
                raise ValueError(f"PairwiseRecord field {name}={val} must be finite in [0.0, 1.0]")

        if not (0.0 <= self.absolute_difference <= 1.0):
            raise ValueError(f"absolute_difference={self.absolute_difference} must be in [0.0, 1.0]")
        if not (-1.0 <= self.signed_difference <= 1.0):
            raise ValueError(f"signed_difference={self.signed_difference} must be in [-1.0, 1.0]")

        # Mathematical consistency
        expected_signed = self.risk_a - self.risk_b
        if abs(self.signed_difference - expected_signed) > 1e-6:
            raise ValueError(
                f"Signed difference {self.signed_difference} does not match risk_a - risk_b ({expected_signed})"
            )

        expected_abs = abs(expected_signed)
        if abs(self.absolute_difference - expected_abs) > 1e-6:
            raise ValueError(
                f"Absolute difference {self.absolute_difference} does not match |risk_a - risk_b| ({expected_abs})"
            )


@dataclass(frozen=True)
class ConflictResult:
    """
    Immutable, self-validating result of cross-modality conflict analysis.
    Evaluates disagreement magnitude, weighted dispersion, and pairwise attribution
    without modifying upstream router weights or DCRI values.
    """
    packet_id: str
    active_modalities: Tuple[str, ...]
    num_active: int
    conflict_available: bool
    max_disagreement: Optional[float]      # Delta_max = max |r_j - r_k| (None if < 2 modalities)
    mean_disagreement: Optional[float]     # Delta_mean = (1/P) sum |r_j - r_k| (None if < 2 modalities)
    weighted_variance: Optional[float]     # V_w = sum w_i (r_i - R_fusion)^2
    weighted_std: Optional[float]          # sigma_w = sqrt(V_w)
    pairwise_records: Tuple[PairwiseRecord, ...]
    dominant_conflict_pair: Optional[Tuple[str, str]]
    conflict_severity: str                 # "LOW", "MODERATE", "HIGH", "NOT_APPLICABLE", "NO_MODALITY_AVAILABLE"
    weight_entropy: float                  # H = -sum w_i log(w_i)
    max_weight: float                      # max(w_i)
    dominant_modality: str                 # modality with max(w_i) or "tied" / "none"
    r_fusion: float
    dcri: float
    uncertainty_sum: float
    uncertainty_mean: float

    def __post_init__(self) -> None:
        if not self.packet_id:
            raise ValueError("packet_id cannot be empty")

        if self.num_active != len(self.active_modalities):
            raise ValueError(
                f"num_active={self.num_active} does not match len(active_modalities)={len(self.active_modalities)}"
            )

        # Zero Modality Case
        if self.num_active == 0:
            if self.conflict_available:
                raise ValueError("conflict_available must be False for zero modalities")
            if self.conflict_severity != "NO_MODALITY_AVAILABLE":
                raise ValueError("conflict_severity must be 'NO_MODALITY_AVAILABLE' when active count is 0")
            if self.max_disagreement is not None or self.mean_disagreement is not None:
                raise ValueError("Conflict disagreement metrics must be None when no modalities are available")
            return

        # Single Modality Case
        if self.num_active == 1:
            if self.conflict_available:
                raise ValueError("conflict_available must be False for a single modality")
            if self.conflict_severity != "NOT_APPLICABLE":
                raise ValueError("conflict_severity must be 'NOT_APPLICABLE' for a single modality")
            if self.max_disagreement is not None or self.mean_disagreement is not None:
                raise ValueError("Pairwise disagreement metrics must be None for single modality")
            if self.weighted_variance is not None and abs(self.weighted_variance) > 1e-6:
                raise ValueError(f"Weighted variance for single modality must be 0.0, got {self.weighted_variance}")
            if self.weighted_std is not None and abs(self.weighted_std) > 1e-6:
                raise ValueError(f"Weighted std for single modality must be 0.0, got {self.weighted_std}")
            return

        # Multi-Modality Case (num_active >= 2)
        if not self.conflict_available:
            raise ValueError("conflict_available must be True when num_active >= 2")

        if self.conflict_severity not in {"LOW", "MODERATE", "HIGH"}:
            raise ValueError(
                f"conflict_severity must be 'LOW', 'MODERATE', or 'HIGH' for multi-modality packets, got {self.conflict_severity}"
            )

        if self.max_disagreement is None or not (0.0 <= self.max_disagreement <= 1.0):
            raise ValueError(f"max_disagreement={self.max_disagreement} must be in [0.0, 1.0]")

        if self.mean_disagreement is None or not (0.0 <= self.mean_disagreement <= 1.0):
            raise ValueError(f"mean_disagreement={self.mean_disagreement} must be in [0.0, 1.0]")

        if self.max_disagreement < self.mean_disagreement - 1e-6:
            raise ValueError(
                f"max_disagreement ({self.max_disagreement}) cannot be less than mean_disagreement ({self.mean_disagreement})"
            )

        if self.weighted_variance is None or not (0.0 <= self.weighted_variance <= 0.25 + 1e-6):
            raise ValueError(f"weighted_variance={self.weighted_variance} must be in [0.0, 0.25]")

        if self.weighted_std is None or not (0.0 <= self.weighted_std <= 0.5 + 1e-6):
            raise ValueError(f"weighted_std={self.weighted_std} must be in [0.0, 0.5]")

        expected_pairs = (self.num_active * (self.num_active - 1)) // 2
        if len(self.pairwise_records) != expected_pairs:
            raise ValueError(
                f"Expected {expected_pairs} pairwise records for {self.num_active} modalities, got {len(self.pairwise_records)}"
            )

        if self.dominant_conflict_pair is None:
            raise ValueError("dominant_conflict_pair cannot be None when conflict is available")
