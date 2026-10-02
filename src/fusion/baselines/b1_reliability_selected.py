"""
FusionMedAI - Phase C11.5: Baseline B1 (Reliability-Selected Unimodal Baseline)
Selects the single most reliable available modality based on frozen validation priors:
    i* = argmax_{i in A} R_i
    w_{i*} = 1.0, w_{j != i*} = 0.0
    R_fusion = r_{i*}
"""

from typing import Dict, Any, Optional
import math

from src.fusion.baselines.decision_packet import ControlledDecisionPacket, FusionResult


class ReliabilitySelectedBaseline:
    """
    Baseline B1: Reliability-Selected Unimodal Strategy.
    Assigns 100% authority to the available modality with the highest historical validation reliability prior.
    """

    BASELINE_ID: str = "B1"
    BASELINE_NAME: str = "Reliability_Selected_Unimodal"

    def __init__(self):
        pass

    def evaluate(self, packet: ControlledDecisionPacket) -> FusionResult:
        active = packet.available_modalities
        num_active = len(active)

        # Cross-modality disagreement calculation
        disagreement = self._compute_disagreement(packet)

        # Zero-Modality Safe Rejection
        if num_active == 0:
            return FusionResult(
                baseline_id=self.BASELINE_ID,
                baseline_name=self.BASELINE_NAME,
                packet_id=packet.packet_id,
                weights={"retina": 0.0, "foot": 0.0, "clinical": 0.0},
                r_fusion=0.0,
                active_modalities=[],
                num_active=0,
                routing_entropy=0.0,
                dominant_modality=None,
                disagreement=disagreement,
                status="NO_MODALITY_AVAILABLE",
            )

        # Pick modality with highest frozen reliability prior R_i
        # Ties broken deterministically by order ['retina', 'foot', 'clinical']
        best_mod = max(active, key=lambda m: packet.records[m].reliability)

        weights = {"retina": 0.0, "foot": 0.0, "clinical": 0.0}
        weights[best_mod] = 1.0

        r_fusion = float(packet.records[best_mod].risk)

        return FusionResult(
            baseline_id=self.BASELINE_ID,
            baseline_name=self.BASELINE_NAME,
            packet_id=packet.packet_id,
            weights=weights,
            r_fusion=r_fusion,
            active_modalities=active,
            num_active=num_active,
            routing_entropy=0.0,  # 1.0 * ln(1.0) = 0.0
            dominant_modality=best_mod,
            disagreement=disagreement,
            status="SUCCESS",
        )

    @staticmethod
    def _compute_disagreement(packet: ControlledDecisionPacket) -> Dict[str, float]:
        r_R = packet.retina.risk if packet.retina.availability else None
        r_F = packet.foot.risk if packet.foot.availability else None
        r_C = packet.clinical.risk if packet.clinical.availability else None

        x_rf = abs(r_R - r_F) if (r_R is not None and r_F is not None) else 0.0
        x_rc = abs(r_R - r_C) if (r_R is not None and r_C is not None) else 0.0
        x_fc = abs(r_F - r_C) if (r_F is not None and r_C is not None) else 0.0
        x_max = max(x_rf, x_rc, x_fc)

        return {
            "X_RF": float(x_rf),
            "X_RC": float(x_rc),
            "X_FC": float(x_fc),
            "X_max": float(x_max),
        }
