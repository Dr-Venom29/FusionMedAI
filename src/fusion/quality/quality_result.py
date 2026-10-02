"""
Unified Quality Result Contract (Phase C11.2).
Strictly encapsulates input availability A_i and input quality Q_i before entering the fusion layer.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


class QualityContractError(ValueError):
    """Raised when a QualityResult violates contract invariants."""
    pass


@dataclass(frozen=True)
class QualityResult:
    """
    Immutable representation of modality input availability and signal quality.
    
    Invariants:
    1. availability must be a boolean.
    2. quality must be a float in [0.0, 1.0].
    3. If availability is False, quality must be exactly 0.0.
    """
    availability: bool
    quality: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.availability, bool):
            raise QualityContractError(
                f"availability must be a bool, got {type(self.availability).__name__}"
            )
        
        if not isinstance(self.quality, (int, float)):
            raise QualityContractError(
                f"quality must be a float, got {type(self.quality).__name__}"
            )
        
        quality_val = float(self.quality)
        
        # Check for NaN / Inf
        if quality_val != quality_val or quality_val == float("inf") or quality_val == float("-inf"):
            raise QualityContractError("quality must be a finite real number, not NaN or Inf")
        
        if quality_val < 0.0 or quality_val > 1.0:
            raise QualityContractError(
                f"quality must be in [0.0, 1.0], got {quality_val}"
            )
        
        if not self.availability and quality_val != 0.0:
            raise QualityContractError(
                f"Invariant violation: availability=False requires quality=0.0, got {quality_val}"
            )


def make_unavailable_quality(reason: str = "INPUT_UNAVAILABLE") -> QualityResult:
    """Helper to construct a standard unavailable QualityResult."""
    return QualityResult(
        availability=False,
        quality=0.0,
        diagnostics={"status": "UNAVAILABLE", "reason": reason}
    )
