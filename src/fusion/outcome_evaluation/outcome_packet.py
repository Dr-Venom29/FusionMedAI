"""
FusionMedAI - Outcome-Grounded Multimodal Decision Packet Schema
Defines strictly validated data containers for evaluating predictive error (MAE, RMSE)
and decision loss against an independent latent oracle target Y* in [0.0, 1.0].
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import math

from src.fusion.baselines.decision_packet import ModalityRecord, ControlledDecisionPacket, DecisionPacketError


class OutcomePacketError(ValueError):
    """Raised when outcome-grounded packet invariants are violated."""
    pass


VALID_SCENARIOS = (
    "CLEAN",
    "MILD_DEGRADATION",
    "MODERATE_DEGRADATION",
    "SEVERE_DEGRADATION",
    "MISLEADING_QUALITY",
    "UNCERTAINTY_MISCALIBRATION",
    "SPARSE_AVAILABILITY",
)

VALID_FIDELITIES = (
    "ACCURATE",
    "MISLEADING_UNNOTICED",
    "MISLEADING_FALSE_ALARM",
    "UNCERTAINTY_MISCALIBRATED",
)


@dataclass(frozen=True)
class OutcomeGroundedPacket:
    """
    Multimodal decision packet paired with an independent latent oracle risk target.
    
    Attributes:
        packet_id: Unique identifier for the packet.
        seed: Random seed used to generate the packet.
        oracle_risk: True latent risk Y* in [0.0, 1.0] generated prior to modality observations.
        oracle_tier: Decision tier corresponding to oracle_risk ('TIER_0_ROUTINE', 'TIER_1_ASSESSMENT', 'TIER_2_ESCALATION').
        retina: Modality record for retinal image channel.
        foot: Modality record for foot ulcer image channel.
        clinical: Modality record for structured clinical EHR channel.
        scenario: Generation scenario string.
        degraded_modality: Modality that underwent degradation (if applicable, else None).
        quality_score_fidelity: Fidelity of quality scoring ('ACCURATE', 'MISLEADING_UNNOTICED', 'MISLEADING_FALSE_ALARM', 'UNCERTAINTY_MISCALIBRATED').
    """
    packet_id: str
    seed: int
    oracle_risk: float
    oracle_tier: str
    retina: ModalityRecord
    foot: ModalityRecord
    clinical: ModalityRecord
    scenario: str
    degraded_modality: Optional[str] = None
    quality_score_fidelity: str = "ACCURATE"
    packet_type: str = "OUTCOME_GROUNDED_DECISION_PACKET"

    def __post_init__(self) -> None:
        if self.packet_type != "OUTCOME_GROUNDED_DECISION_PACKET":
            raise OutcomePacketError(f"Invalid packet_type: {self.packet_type}")
        if not isinstance(self.oracle_risk, (int, float)) or math.isnan(self.oracle_risk) or math.isinf(self.oracle_risk):
            raise OutcomePacketError(f"Oracle risk must be finite float, got {self.oracle_risk}")
        if self.oracle_risk < 0.0 or self.oracle_risk > 1.0:
            raise OutcomePacketError(f"Oracle risk = {self.oracle_risk:.6f} out of bounds [0.0, 1.0]")
        if self.oracle_tier not in ("TIER_0_ROUTINE", "TIER_1_ASSESSMENT", "TIER_2_ESCALATION"):
            raise OutcomePacketError(f"Invalid oracle_tier: {self.oracle_tier}")

        # Strict reconciliation between oracle_risk and oracle_tier without ambiguous boundary tolerances
        if self.oracle_risk < 0.20:
            if self.oracle_tier != "TIER_0_ROUTINE":
                raise OutcomePacketError(f"Oracle risk {self.oracle_risk:.6f} < 0.20 requires TIER_0_ROUTINE, got {self.oracle_tier}")
        elif self.oracle_risk < 0.40:
            if self.oracle_tier != "TIER_1_ASSESSMENT":
                raise OutcomePacketError(f"Oracle risk {self.oracle_risk:.6f} in [0.20, 0.40) requires TIER_1_ASSESSMENT, got {self.oracle_tier}")
        else:
            if self.oracle_tier != "TIER_2_ESCALATION":
                raise OutcomePacketError(f"Oracle risk {self.oracle_risk:.6f} >= 0.40 requires TIER_2_ESCALATION, got {self.oracle_tier}")

        if self.scenario not in VALID_SCENARIOS:
            raise OutcomePacketError(f"Invalid scenario '{self.scenario}'. Expected one of {VALID_SCENARIOS}")
        if self.quality_score_fidelity not in VALID_FIDELITIES:
            raise OutcomePacketError(f"Invalid quality_score_fidelity '{self.quality_score_fidelity}'. Expected one of {VALID_FIDELITIES}")

        # Strict scenario & fidelity compatibility
        if self.quality_score_fidelity == "MISLEADING_UNNOTICED":
            if self.scenario not in ("MILD_DEGRADATION", "MODERATE_DEGRADATION", "SEVERE_DEGRADATION"):
                raise OutcomePacketError(f"MISLEADING_UNNOTICED fidelity requires degradation scenario, got {self.scenario}")
        elif self.quality_score_fidelity == "MISLEADING_FALSE_ALARM":
            if self.scenario != "CLEAN":
                raise OutcomePacketError(f"MISLEADING_FALSE_ALARM fidelity requires CLEAN scenario, got {self.scenario}")
        elif self.quality_score_fidelity == "UNCERTAINTY_MISCALIBRATED":
            if self.scenario != "UNCERTAINTY_MISCALIBRATION":
                raise OutcomePacketError(f"UNCERTAINTY_MISCALIBRATED fidelity requires UNCERTAINTY_MISCALIBRATION scenario, got {self.scenario}")

        # Modality consistency
        if self.retina.modality != "retina" or self.foot.modality != "foot" or self.clinical.modality != "clinical":
            raise OutcomePacketError("Modality records channel name mismatch.")

        # Comprehensive validation for all 3 modality records
        for m_name, rec in self.records.items():
            if not isinstance(rec.availability, bool):
                raise OutcomePacketError(f"Modality {m_name} availability must be boolean.")
            for attr in ("risk", "confidence", "uncertainty", "quality", "reliability"):
                val = getattr(rec, attr)
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    raise OutcomePacketError(f"Modality {m_name} {attr} must be a finite float, got {val}")
                if val < 0.0 or val > 1.0:
                    raise OutcomePacketError(f"Modality {m_name} {attr} = {val:.6f} out of bounds [0.0, 1.0]")
            
            # Calibrated probability simplex validation
            if rec.availability and rec.calibrated_probability is not None:
                p_cal = rec.calibrated_probability
                if not (isinstance(p_cal, tuple) and len(p_cal) == 2):
                    raise OutcomePacketError(f"Modality {m_name} calibrated_probability must be a 2-tuple.")
                if abs(sum(p_cal) - 1.0) > 1e-4:
                    raise OutcomePacketError(f"Modality {m_name} calibrated_probability sum {sum(p_cal):.4f} != 1.0")

        # Reject packets where zero modalities are available
        if self.num_available == 0:
            raise OutcomePacketError(f"Packet {self.packet_id} has zero available modalities. At least one modality must be active.")

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

    def to_controlled_decision_packet(self) -> ControlledDecisionPacket:
        """
        Adapts cleanly to a standard ControlledDecisionPacket so existing B1–B6 baselines
        and ACARA-U router evaluate the exact same modality inputs without modification.
        """
        return ControlledDecisionPacket(
            packet_id=self.packet_id,
            retina=self.retina,
            foot=self.foot,
            clinical=self.clinical,
            seed=self.seed,
            packet_type="CONTROLLED_DECISION_PACKET",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "seed": self.seed,
            "oracle_risk": round(float(self.oracle_risk), 6),
            "oracle_tier": self.oracle_tier,
            "scenario": self.scenario,
            "degraded_modality": self.degraded_modality,
            "quality_score_fidelity": self.quality_score_fidelity,
            "retina": self.retina.to_dict(),
            "foot": self.foot.to_dict(),
            "clinical": self.clinical.to_dict(),
            "available_modalities": self.available_modalities,
            "num_available": self.num_available,
        }
