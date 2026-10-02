"""
FusionMedAI - Phase C11.4: Router Input Contracts & Channel Invariants
Defines immutable per-modality and multi-channel input containers for ACARA-U v2 dynamic router.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np

from src.fusion.contracts.modality_output import ModalityOutput
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)

# Immutable set of recognized modality identifiers
VALID_MODALITIES: Tuple[str, ...] = ("retina", "foot", "clinical")

# Mapping of frozen validation global reliability priors from C11.3
FROZEN_RELIABILITY_MAP: Dict[str, float] = {
    "retina": FROZEN_RETINA_RELIABILITY,      # 0.929956
    "foot": FROZEN_FOOT_RELIABILITY,          # 0.922266
    "clinical": FROZEN_CLINICAL_RELIABILITY,  # 0.825382
}


class RouterContractValidationError(ValueError):
    """Raised when router input invariants or boundaries are violated."""
    pass


@dataclass(frozen=True)
class ModalityChannelInput:
    """
    Immutable container representing the normalized routing attributes of a single modality channel.
    
    Attributes:
        modality: Modality identifier ('retina', 'foot', or 'clinical').
        confidence: Normalized model certainty C_i in [0.0, 1.0].
        reliability: Historical validation reliability prior R_i in [0.0, 1.0].
        uncertainty: Normalized predictive uncertainty U_i in [0.0, 1.0].
        quality: Input signal quality Q_i in [0.0, 1.0].
        availability: Binary availability flag A_i (True = 1, False = 0).
    """
    modality: str
    confidence: float
    reliability: float
    uncertainty: float
    quality: float
    availability: bool

    def __post_init__(self) -> None:
        # 1. Modality Name Validation
        if self.modality not in VALID_MODALITIES:
            raise RouterContractValidationError(
                f"Invalid modality '{self.modality}'. Must be one of {VALID_MODALITIES}."
            )

        # 2. Availability Type Validation
        if not isinstance(self.availability, bool):
            raise RouterContractValidationError(
                f"Modality '{self.modality}' availability must be a boolean, got {type(self.availability).__name__}."
            )

        # 3. Scalar Range Invariants [0.0, 1.0] and Non-NaN / Non-Inf
        for attr_name, val in [
            ("confidence", self.confidence),
            ("reliability", self.reliability),
            ("uncertainty", self.uncertainty),
            ("quality", self.quality),
        ]:
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise RouterContractValidationError(
                    f"Modality '{self.modality}' {attr_name} must be a finite float, got {val}."
                )
            if val < 0.0 - 1e-7 or val > 1.0 + 1e-7:
                raise RouterContractValidationError(
                    f"Modality '{self.modality}' {attr_name} = {val:.6f} exceeds bounded domain [0.0, 1.0]."
                )

        # 4. Hard Availability Invariant: A_i = 0 => Q_i = 0.0 (C11.2 invariant)
        if not self.availability and self.quality > 1e-7:
            raise RouterContractValidationError(
                f"Modality '{self.modality}' is unavailable (A_i=0) but has non-zero quality Q_i = {self.quality:.6f}. "
                f"Hard invariant A_i=0 => Q_i=0.0 violated."
            )

        # 5. Frozen Reliability Invariant: R_i must match Phase C11.3 frozen constant
        expected_rel = FROZEN_RELIABILITY_MAP[self.modality]
        if abs(self.reliability - expected_rel) > 1e-5:
            raise RouterContractValidationError(
                f"Modality '{self.modality}' reliability R_i = {self.reliability:.6f} does not match "
                f"Phase C11.3 frozen prior {expected_rel:.6f}."
            )

    @classmethod
    def from_modality_output(cls, modality: str, output: ModalityOutput) -> "ModalityChannelInput":
        """Constructs a ModalityChannelInput directly from a Phase C11.1 ModalityOutput contract."""
        return cls(
            modality=modality,
            confidence=float(output.confidence),
            reliability=float(output.reliability),
            uncertainty=float(output.uncertainty),
            quality=float(output.quality),
            availability=bool(output.availability),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts channel attributes to a JSON-serializable dictionary."""
        return {
            "modality": self.modality,
            "confidence": round(float(self.confidence), 6),
            "reliability": round(float(self.reliability), 6),
            "uncertainty": round(float(self.uncertainty), 6),
            "quality": round(float(self.quality), 6),
            "availability": bool(self.availability),
        }


@dataclass(frozen=True)
class RouterInput:
    """
    Canonical tri-modal container passed to ACARA-U v2 Router.
    Contains immutable channel representations for Retina, Foot, and Clinical modalities.
    """
    retina: ModalityChannelInput
    foot: ModalityChannelInput
    clinical: ModalityChannelInput

    def __post_init__(self) -> None:
        # Validate channel modality identities
        if self.retina.modality != "retina":
            raise RouterContractValidationError(f"Expected 'retina' channel, got '{self.retina.modality}'.")
        if self.foot.modality != "foot":
            raise RouterContractValidationError(f"Expected 'foot' channel, got '{self.foot.modality}'.")
        if self.clinical.modality != "clinical":
            raise RouterContractValidationError(f"Expected 'clinical' channel, got '{self.clinical.modality}'.")

    @property
    def channels(self) -> Dict[str, ModalityChannelInput]:
        """Returns map of modality name to channel input."""
        return {
            "retina": self.retina,
            "foot": self.foot,
            "clinical": self.clinical,
        }

    @property
    def available_modalities(self) -> List[str]:
        """Returns list of active/available modality names."""
        return [m for m, ch in self.channels.items() if ch.availability]

    @property
    def num_available(self) -> int:
        """Returns count of active modalities (0 to 3)."""
        return len(self.available_modalities)

    @classmethod
    def from_outputs(
        cls,
        retina: ModalityOutput,
        foot: ModalityOutput,
        clinical: ModalityOutput,
    ) -> "RouterInput":
        """Constructs a RouterInput from three ModalityOutput contracts."""
        return cls(
            retina=ModalityChannelInput.from_modality_output("retina", retina),
            foot=ModalityChannelInput.from_modality_output("foot", foot),
            clinical=ModalityChannelInput.from_modality_output("clinical", clinical),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes router input to dictionary."""
        return {
            "retina": self.retina.to_dict(),
            "foot": self.foot.to_dict(),
            "clinical": self.clinical.to_dict(),
            "available_modalities": self.available_modalities,
            "num_available": self.num_available,
        }
