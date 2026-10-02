"""
Data models and contract classes for Modality Reliability (Phase C11.3).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


class ReliabilityContractError(ValueError):
    """Raised when reliability attributes violate protocol invariants."""
    pass


@dataclass(frozen=True)
class ModalityReliability:
    """
    Immutable representation of a single modality's frozen global reliability.
    
    Invariants:
    1. 0.0 <= auc <= 1.0
    2. 0.0 <= ece <= 1.0
    3. 0.0 <= reliability <= 1.0
    4. reliability == 0.5 * (auc + (1.0 - ece))
    """
    modality: str
    auc: float
    ece: float
    reliability: float
    n_val_samples: int
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.auc <= 1.0):
            raise ReliabilityContractError(f"AUC must be in [0.0, 1.0], got {self.auc}")
        if not (0.0 <= self.ece <= 1.0):
            raise ReliabilityContractError(f"ECE must be in [0.0, 1.0], got {self.ece}")
        if not (0.0 <= self.reliability <= 1.0):
            raise ReliabilityContractError(f"Reliability must be in [0.0, 1.0], got {self.reliability}")
        
        expected_rel = 0.5 * (self.auc + (1.0 - self.ece))
        if abs(self.reliability - expected_rel) > 1e-5:
            raise ReliabilityContractError(
                f"Reliability mathematical mismatch: expected {expected_rel:.6f}, got {self.reliability:.6f}"
            )


@dataclass(frozen=True)
class GlobalReliabilitySnapshot:
    """
    Immutable container of global reliability coefficients across all modalities.
    """
    retina: ModalityReliability
    foot: ModalityReliability
    clinical: ModalityReliability
    method: str = "uniform_auc_calibration_reliability"
    data_source: str = "locked_validation_only"
    test_used_for_selection: bool = False
