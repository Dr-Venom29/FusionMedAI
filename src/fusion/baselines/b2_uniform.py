"""
FusionMedAI - Phase C11.5: Baseline B2 (Uniform Average Fusion)
Assigns equal authority to all active modalities:
    w_i = 1 / |A|  for i in A,  w_k = 0.0 for k not in A
    R_fusion = (1 / |A|) * sum_{i in A} r_i
"""

from typing import Dict, Any, Optional
import math

from src.fusion.baselines.decision_packet import ControlledDecisionPacket, FusionResult


class UniformAverageBaseline:
    """
    Baseline B2: Uniform Average Fusion Strategy.
    Non-adaptive baseline distributing authority equally among available modalities.
    """

    BASELINE_ID: str = "B2"
    BASELINE_NAME: str = "Uniform_Average_Fusion"

    def __init__(self):
        pass

    def evaluate(self, packet: ControlledDecisionPacket) -> FusionResult:
        active = packet.available_modalities
        num_active = len(active)
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

        # Explicit independent uniform weighting
        uniform_weight = 1.0 / float(num_active)
        weights = {"retina": 0.0, "foot": 0.0, "clinical": 0.0}
        for m in active:
            weights[m] = uniform_weight

        # Fused risk = convex combination
        r_fusion = sum(weights[m] * float(packet.records[m].risk) for m in active)

        # Routing entropy = ln(|A|)
        entropy = math.log(float(num_active))

        # In uniform weighting, dominant modality is tie-broken by canonical order
        dominant_mod = active[0]

        return FusionResult(
            baseline_id=self.BASELINE_ID,
            baseline_name=self.BASELINE_NAME,
            packet_id=packet.packet_id,
            weights=weights,
            r_fusion=float(r_fusion),
            active_modalities=active,
            num_active=num_active,
            routing_entropy=float(entropy),
            dominant_modality=dominant_mod,
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
