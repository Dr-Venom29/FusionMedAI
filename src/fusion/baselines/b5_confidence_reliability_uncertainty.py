"""
FusionMedAI - Phase C11.5: Baseline B5 (Confidence + Reliability + Uncertainty Fusion)
Crucial Uncertainty Ablation Strategy:
    z_i = 1.0 * C_i + 1.0 * R_i - 1.0 * U_i
    w_i = Softmax(z_i) over active modalities A
    R_fusion = sum_{i in A} w_i * r_i
"""

from typing import Dict, Any, Optional
import math

from src.fusion.baselines.decision_packet import ControlledDecisionPacket, FusionResult


class ConfidenceReliabilityUncertaintyBaseline:
    """
    Baseline B5: Confidence + Reliability - Uncertainty Strategy (alpha=1, beta=1, gamma=1, eta=0).
    Ablates the predictive uncertainty penalty (-gamma * U_i) without quality bonus.
    """

    BASELINE_ID: str = "B5"
    BASELINE_NAME: str = "Confidence_Reliability_Uncertainty_Fusion"

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

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

        # Compute logits: z_i = alpha * C_i + beta * R_i - gamma * U_i
        logits = {
            m: self.alpha * float(packet.records[m].confidence)
            + self.beta * float(packet.records[m].reliability)
            - self.gamma * float(packet.records[m].uncertainty)
            for m in active
        }
        max_logit = max(logits.values())

        # Softmax
        exp_scores = {m: math.exp(logits[m] - max_logit) for m in active}
        sum_exp = sum(exp_scores.values())

        weights = {"retina": 0.0, "foot": 0.0, "clinical": 0.0}
        for m in active:
            weights[m] = float(exp_scores[m] / sum_exp)

        # Fused risk
        r_fusion = sum(weights[m] * float(packet.records[m].risk) for m in active)

        # Entropy
        entropy = -sum(weights[m] * math.log(weights[m]) for m in active if weights[m] > 1e-12)

        # Dominant modality
        dominant_mod = max(active, key=lambda m: weights[m])

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
