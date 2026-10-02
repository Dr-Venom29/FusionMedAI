"""
FusionMedAI - Phase C11.5: Controlled Decision Packet & Baseline Results Contracts
Defines immutable data containers for synthetic decision-level multimodal packets and fusion results.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import math
import numpy as np

from src.fusion.contracts.modality_output import ModalityOutput
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)
from src.fusion.router.router_input import ModalityChannelInput, RouterInput


class DecisionPacketError(ValueError):
    """Raised when decision packet invariants are violated."""
    pass


@dataclass(frozen=True)
class ModalityRecord:
    """
    Immutable representation of a single modality prediction inside a decision packet.
    
    Attributes:
        sample_id: Identifier of the source sample / encounter.
        modality: Modality identifier ('retina', 'foot', or 'clinical').
        risk: Continuous scalar risk projection r_i in [0.0, 1.0].
        calibrated_probability: Full class posterior vector summing to 1.0.
        confidence: Normalized model confidence C_i in [0.0, 1.0].
        uncertainty: Normalized predictive uncertainty U_i in [0.0, 1.0].
        quality: Input quality score Q_i in [0.0, 1.0].
        availability: Binary availability flag A_i (True = 1, False = 0).
        reliability: Historical empirical validation reliability prior R_i.
        model_version: Modality architecture version string.
    """
    sample_id: str
    modality: str
    risk: float
    calibrated_probability: Tuple[float, ...]
    confidence: float
    uncertainty: float
    quality: float
    availability: bool
    reliability: float
    model_version: str

    def __post_init__(self) -> None:
        if self.modality not in ("retina", "foot", "clinical"):
            raise DecisionPacketError(f"Invalid modality identifier '{self.modality}'.")
        for attr, val in [
            ("risk", self.risk),
            ("confidence", self.confidence),
            ("uncertainty", self.uncertainty),
            ("quality", self.quality),
            ("reliability", self.reliability),
        ]:
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise DecisionPacketError(f"Modality '{self.modality}' {attr} must be finite float, got {val}.")
            if val < 0.0 - 1e-7 or val > 1.0 + 1e-7:
                raise DecisionPacketError(f"Modality '{self.modality}' {attr} = {val:.6f} exceeds [0, 1].")

        # Invariant: A_i = 0 => Q_i = 0.0
        if not self.availability and self.quality > 1e-7:
            raise DecisionPacketError(f"Unavailable modality '{self.modality}' has non-zero quality {self.quality}.")

    def to_channel_input(self) -> ModalityChannelInput:
        """Converts record to ModalityChannelInput for ACARA-U routing."""
        return ModalityChannelInput(
            modality=self.modality,
            confidence=float(self.confidence),
            reliability=float(self.reliability),
            uncertainty=float(self.uncertainty),
            quality=float(self.quality),
            availability=bool(self.availability),
        )

    @classmethod
    def from_modality_output(cls, sample_id: str, modality: str, output: ModalityOutput) -> "ModalityRecord":
        """Constructs ModalityRecord directly from a C11.1 ModalityOutput contract."""
        return cls(
            sample_id=sample_id,
            modality=modality,
            risk=float(output.risk),
            calibrated_probability=tuple(float(p) for p in output.calibrated_probability),
            confidence=float(output.confidence),
            uncertainty=float(output.uncertainty),
            quality=float(output.quality),
            availability=bool(output.availability),
            reliability=float(output.reliability),
            model_version=output.model_version,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "modality": self.modality,
            "risk": round(float(self.risk), 6),
            "calibrated_probability": [round(float(p), 6) for p in self.calibrated_probability],
            "confidence": round(float(self.confidence), 6),
            "uncertainty": round(float(self.uncertainty), 6),
            "quality": round(float(self.quality), 6),
            "availability": bool(self.availability),
            "reliability": round(float(self.reliability), 6),
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class ControlledDecisionPacket:
    """
    Controlled decision-level packet bundling held-out predictions from three independent cohorts.
    Explicitly labeled as 'CONTROLLED_DECISION_PACKET' to prevent invalid patient-level claims.
    """
    packet_id: str
    retina: ModalityRecord
    foot: ModalityRecord
    clinical: ModalityRecord
    seed: int
    packet_type: str = "CONTROLLED_DECISION_PACKET"

    def __post_init__(self) -> None:
        if self.packet_type != "CONTROLLED_DECISION_PACKET":
            raise DecisionPacketError(
                f"Packet type must be 'CONTROLLED_DECISION_PACKET', got '{self.packet_type}'."
            )
        if self.retina.modality != "retina":
            raise DecisionPacketError("Expected 'retina' record.")
        if self.foot.modality != "foot":
            raise DecisionPacketError("Expected 'foot' record.")
        if self.clinical.modality != "clinical":
            raise DecisionPacketError("Expected 'clinical' record.")

    @property
    def records(self) -> Dict[str, ModalityRecord]:
        return {
            "retina": self.retina,
            "foot": self.foot,
            "clinical": self.clinical,
        }

    @property
    def available_modalities(self) -> List[str]:
        return [m for m, rec in self.records.items() if rec.availability]

    @property
    def num_available(self) -> int:
        return len(self.available_modalities)

    def to_router_input(self) -> RouterInput:
        """Constructs canonical RouterInput container."""
        return RouterInput(
            retina=self.retina.to_channel_input(),
            foot=self.foot.to_channel_input(),
            clinical=self.clinical.to_channel_input(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "packet_type": self.packet_type,
            "seed": self.seed,
            "retina": self.retina.to_dict(),
            "foot": self.foot.to_dict(),
            "clinical": self.clinical.to_dict(),
            "available_modalities": self.available_modalities,
            "num_available": self.num_available,
        }


@dataclass(frozen=True)
class FusionResult:
    """
    Immutable result object produced by any baseline (B1–B6) on a decision packet.
    """
    baseline_id: str                    # 'B1', 'B2', 'B3', 'B4', 'B5', or 'B6'
    baseline_name: str                  # Descriptive name
    packet_id: str                      # Identifier of evaluated packet
    weights: Dict[str, float]           # Routing weights w_i
    r_fusion: float                     # Decision-level aggregated risk index in [0.0, 1.0]
    active_modalities: List[str]        # Available modalities
    num_active: int                     # Count of available modalities
    routing_entropy: float              # Modality weight entropy H(w)
    dominant_modality: Optional[str]    # Modality receiving max weight (None if zero-modality)
    disagreement: Dict[str, float]      # Cross-modality risk absolute differences
    status: str = "SUCCESS"             # 'SUCCESS' or 'NO_MODALITY_AVAILABLE'

    def __post_init__(self) -> None:
        if self.status == "NO_MODALITY_AVAILABLE":
            if self.num_active != 0:
                raise DecisionPacketError("NO_MODALITY_AVAILABLE must have num_active == 0.")
            if self.r_fusion != 0.0:
                raise DecisionPacketError("r_fusion must be 0.0 in NO_MODALITY_AVAILABLE state.")
        else:
            if self.num_active == 0:
                raise DecisionPacketError("SUCCESS status requires at least 1 active modality.")
            if self.r_fusion < 0.0 - 1e-7 or self.r_fusion > 1.0 + 1e-7:
                raise DecisionPacketError(f"r_fusion = {self.r_fusion:.6f} out of bounds [0.0, 1.0].")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "baseline_name": self.baseline_name,
            "packet_id": self.packet_id,
            "weights": {m: round(float(w), 6) for m, w in self.weights.items()},
            "r_fusion": round(float(self.r_fusion), 6),
            "active_modalities": self.active_modalities,
            "num_active": self.num_active,
            "routing_entropy": round(float(self.routing_entropy), 6),
            "dominant_modality": self.dominant_modality,
            "disagreement": {k: round(float(v), 6) for k, v in self.disagreement.items()},
            "status": self.status,
        }
