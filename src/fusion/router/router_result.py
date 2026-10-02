"""
FusionMedAI - Phase C11.4: Router Diagnostics & Output Contract
Defines research-grade immutable diagnostics container RouterResult for ACARA-U dynamic routing.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import math


@dataclass(frozen=True)
class RouterResult:
    """
    Immutable research-grade result container emitted by ACARA-U v2 dynamic router.
    
    Attributes:
        weights: Normalized modality weights w_i (sum to 1.0 across active modalities; 0.0 for unavailable).
        logits: Raw pre-masking logit scores z_i = alpha*C_i + beta*R_i - gamma*U_i + eta*Q_i.
        masked_logits: Post-masking logits z_tilde_i (equal to z_i if active, -inf if unavailable).
        availability: Map of modality availability flags A_i.
        coefficients: Router hyperparameters Theta = (alpha, beta, gamma, eta) applied.
        active_modalities: List of available modality identifiers (|A| in 0..3).
        num_active: Number of active modalities.
        normalization_sum: Sum of weights (1.0 for non-empty, 0.0 for zero-modality).
        routing_entropy: Modality weight dispersion entropy H(w) = -sum w_i * ln(w_i).
        dominant_modality: Modality with highest assigned routing weight (None if zero-modality).
        status: Routing status ('SUCCESS' or 'NO_MODALITY_AVAILABLE').
        router_version: Frozen router version string.
    """
    weights: Dict[str, float]
    logits: Dict[str, float]
    masked_logits: Dict[str, float]
    availability: Dict[str, bool]
    coefficients: Dict[str, float]
    active_modalities: List[str]
    num_active: int
    normalization_sum: float
    routing_entropy: float
    dominant_modality: Optional[str]
    status: str = "SUCCESS"
    router_version: str = "acarau_v2.0"

    def __post_init__(self) -> None:
        # Check invariants
        if self.status == "NO_MODALITY_AVAILABLE":
            if self.num_active != 0 or len(self.active_modalities) != 0:
                raise ValueError("NO_MODALITY_AVAILABLE status must have 0 active modalities.")
            for mod, w in self.weights.items():
                if w != 0.0:
                    raise ValueError(f"Weight for '{mod}' must be 0.0 in NO_MODALITY_AVAILABLE state.")
            if self.dominant_modality is not None:
                raise ValueError("Dominant modality must be None when no modalities are available.")
        else:
            if self.num_active == 0:
                raise ValueError("SUCCESS status requires at least 1 active modality.")
            # Normalization invariant
            if abs(self.normalization_sum - 1.0) > 1e-6:
                raise ValueError(f"Weights must sum to 1.0, got {self.normalization_sum:.8f}.")
            # Masking invariant: unavailable modality must have weight 0.0
            for mod, is_avail in self.availability.items():
                w = self.weights.get(mod, 0.0)
                if not is_avail and w > 1e-7:
                    raise ValueError(f"Unavailable modality '{mod}' assigned non-zero weight {w:.8f}.")
                if w < 0.0 - 1e-7 or w > 1.0 + 1e-7:
                    raise ValueError(f"Modality '{mod}' weight {w:.8f} out of bounds [0.0, 1.0].")

    @classmethod
    def no_modality_available(
        cls,
        coefficients: Dict[str, float],
        router_version: str = "acarau_v2.0",
    ) -> "RouterResult":
        """Constructs a graceful failure result when all modalities are missing."""
        return cls(
            weights={"retina": 0.0, "foot": 0.0, "clinical": 0.0},
            logits={"retina": float("-inf"), "foot": float("-inf"), "clinical": float("-inf")},
            masked_logits={"retina": float("-inf"), "foot": float("-inf"), "clinical": float("-inf")},
            availability={"retina": False, "foot": False, "clinical": False},
            coefficients=coefficients,
            active_modalities=[],
            num_active=0,
            normalization_sum=0.0,
            routing_entropy=0.0,
            dominant_modality=None,
            status="NO_MODALITY_AVAILABLE",
            router_version=router_version,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result into a JSON-compatible dictionary."""
        return {
            "weights": {m: round(float(w), 6) for m, w in self.weights.items()},
            "logits": {m: round(float(z), 6) if math.isfinite(z) else str(z) for m, z in self.logits.items()},
            "masked_logits": {m: round(float(z), 6) if math.isfinite(z) else str(z) for m, z in self.masked_logits.items()},
            "availability": self.availability,
            "coefficients": self.coefficients,
            "active_modalities": self.active_modalities,
            "num_active": self.num_active,
            "normalization_sum": round(float(self.normalization_sum), 6),
            "routing_entropy": round(float(self.routing_entropy), 6),
            "dominant_modality": self.dominant_modality,
            "status": self.status,
            "router_version": self.router_version,
        }
