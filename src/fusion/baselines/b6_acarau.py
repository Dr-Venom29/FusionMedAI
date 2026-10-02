"""
FusionMedAI - Phase C11.5: Baseline B6 (Full ACARA-U Dynamic Fusion)
Evaluates the complete full-featured dynamic router:
    z_i = 1.0 * C_i + 1.0 * R_i - 1.0 * U_i + 1.0 * Q_i
    A_i = 0 => z_tilde_i = -inf
    w_i = Softmax(z_tilde_i)
    R_fusion = sum_{i in A} w_i * r_i
"""

from typing import Dict, Any, Optional
import math

from src.fusion.baselines.decision_packet import ControlledDecisionPacket, FusionResult
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.coefficients import RouterCoefficients


class FullACARAUBaseline:
    """
    Baseline B6: Full ACARA-U Strategy (alpha=1, beta=1, gamma=1, eta=1).
    Wraps the frozen ACARAUv2Router engine.
    """

    BASELINE_ID: str = "B6"
    BASELINE_NAME: str = "Full_ACARA_U"

    def __init__(self, coefficients: Optional[RouterCoefficients] = None):
        self.router = ACARAUv2Router(
            coefficients=coefficients if coefficients is not None else RouterCoefficients.full_acarau()
        )

    def evaluate(self, packet: ControlledDecisionPacket) -> FusionResult:
        router_input = packet.to_router_input()
        router_result = self.router.route(router_input)
        disagreement = self._compute_disagreement(packet)

        if router_result.status == "NO_MODALITY_AVAILABLE":
            return FusionResult(
                baseline_id=self.BASELINE_ID,
                baseline_name=self.BASELINE_NAME,
                packet_id=packet.packet_id,
                weights=router_result.weights,
                r_fusion=0.0,
                active_modalities=[],
                num_active=0,
                routing_entropy=0.0,
                dominant_modality=None,
                disagreement=disagreement,
                status="NO_MODALITY_AVAILABLE",
            )

        # Fused risk = convex combination of modality risks
        r_fusion = sum(
            router_result.weights[m] * float(packet.records[m].risk)
            for m in router_result.active_modalities
        )

        return FusionResult(
            baseline_id=self.BASELINE_ID,
            baseline_name=self.BASELINE_NAME,
            packet_id=packet.packet_id,
            weights=router_result.weights,
            r_fusion=float(r_fusion),
            active_modalities=router_result.active_modalities,
            num_active=router_result.num_active,
            routing_entropy=float(router_result.routing_entropy),
            dominant_modality=router_result.dominant_modality,
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
