"""
FusionMedAI - Phase C11.16: ACARA-U Ablation Evaluation Engine
Executes decision packet evaluation across ablation variants,
supporting full cohort evaluation, regime masking, and fault-injection negative controls.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
from pathlib import Path

from src.fusion.baselines.decision_packet import ControlledDecisionPacket
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router.router_input import RouterInput, ModalityChannelInput
from src.fusion.router.router_result import RouterResult
from src.fusion.dcri.dcri_engine import DCRIEngine
from src.fusion.dcri.dcri_result import DCRIResult
from src.fusion.component_ablation.ablation_config import (
    AblationVariantSpec,
    VARIANTS,
    REGIME_MODALITY_MAP,
    MODALITIES,
    TAU_1_DEFAULT,
    TAU_2_DEFAULT,
    ACTION_ROUTINE_REVIEW,
    ACTION_ADDITIONAL_ASSESSMENT,
    ACTION_ESCALATION,
    ACTION_KEYS,
)


def assign_decision_tier(score: float, tau_1: float = TAU_1_DEFAULT, tau_2: float = TAU_2_DEFAULT) -> int:
    """Assigns decision tier based on thresholds tau_1 and tau_2."""
    if score < tau_1:
        return ACTION_ROUTINE_REVIEW
    elif score < tau_2:
        return ACTION_ADDITIONAL_ASSESSMENT
    else:
        return ACTION_ESCALATION


class AblationEngine:
    """
    Evaluation engine for Phase C11.16 ACARA-U Component Ablation Study.
    """

    def __init__(self, variant: AblationVariantSpec):
        self.variant = variant
        self.coefficients = RouterCoefficients(
            alpha=variant.alpha,
            beta=variant.beta,
            gamma=variant.gamma,
            eta=variant.eta,
        )
        self.router = ACARAUv2Router(
            coefficients=self.coefficients,
            router_version=f"acarau_ablation_{variant.variant_id}",
        )
        self.dcri_engine = DCRIEngine(router=self.router)
        self.packet_exceptions = 0

    def evaluate_packet(
        self,
        packet: ControlledDecisionPacket,
        regime_override: Optional[str] = None,
        force_mask_violation: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates a single packet under this variant's configuration.
        
        Args:
            packet: ControlledDecisionPacket.
            regime_override: If provided, masks channels to match specified regime (e.g. 'RF').
            force_mask_violation: If True (for A6 negative control), deliberately violates hard masking.
            
        Returns:
            Dictionary with evaluation records: logits, weights, r_fusion, dcri, decision_tier, etc.
        """
        # Natural availability from packet vs evaluated availability
        natural_modalities = list(packet.available_modalities)
        is_forced_regime = regime_override is not None
        active_modalities = natural_modalities
        if is_forced_regime:
            active_modalities = list(REGIME_MODALITY_MAP.get(regime_override, ()))

        num_active = len(active_modalities)

        # Edge case: Empty active set (Fail-Closed)
        if num_active == 0 and not force_mask_violation:
            return {
                "packet_id": packet.packet_id,
                "variant_id": self.variant.variant_id,
                "regime": regime_override if regime_override else "EMPTY",
                "natural_modalities": natural_modalities,
                "evaluated_modalities": [],
                "active_modalities": [],
                "is_forced_regime": is_forced_regime,
                "num_active": 0,
                "weights": {"retina": 0.0, "foot": 0.0, "clinical": 0.0},
                "raw_logits": {"retina": 0.0, "foot": 0.0, "clinical": 0.0},
                "masked_logits": {"retina": float("-inf"), "foot": float("-inf"), "clinical": float("-inf")},
                "r_fusion": 0.0,
                "u_sum": 0.0,
                "uncertainty_penalty": 0.0,
                "dcri": 0.0,
                "routing_entropy": 0.0,
                "dominant_modality": None,
                # Fail-closed computational fallback: Tier 0 is a placeholder computational status, not a clinical recommendation
                "decision_tier": ACTION_ROUTINE_REVIEW,
                "decision_tier_name": "computational_null_routine_fallback",
                "status": "NO_MODALITY_AVAILABLE",
                "mask_violation": False,
                "is_finite": True,
            }

        # Build channels
        channels = {}
        for m in MODALITIES:
            rec = getattr(packet, m)
            is_avail = (m in active_modalities)
            channels[m] = ModalityChannelInput(
                modality=m,
                confidence=float(rec.confidence),
                reliability=float(rec.reliability),
                uncertainty=float(rec.uncertainty),
                quality=float(rec.quality) if is_avail else 0.0,
                availability=is_avail,
            )

        router_input = RouterInput(
            retina=channels["retina"],
            foot=channels["foot"],
            clinical=channels["clinical"],
        )

        if force_mask_violation:
            # Negative control A6: deliberate violation of hard availability masking
            router_result = self.router.route(router_input)
            corrupted_weights = dict(router_result.weights)
            
            # Find inactive modality and give it deliberate non-zero weight
            inactive = [m for m in MODALITIES if m not in active_modalities]
            injected_channel = "clinical"
            if inactive:
                injected_channel = inactive[0]
            corrupted_weights[injected_channel] = 0.20
            # Renormalize active to sum to 0.80 so sum is 1.0, but inactive has 0.20
            for m in active_modalities:
                if m != injected_channel:
                    corrupted_weights[m] *= 0.80

            r_fusion = sum(corrupted_weights[m] * float(getattr(packet, m).risk) for m in MODALITIES)
            u_sum = sum(float(getattr(packet, m).uncertainty) for m in active_modalities)
            penalty = self.variant.delta * u_sum
            dcri = r_fusion - penalty
            tier = assign_decision_tier(dcri)

            # Validate finiteness
            import math
            is_finite = math.isfinite(r_fusion) and math.isfinite(dcri) and all(math.isfinite(w) for w in corrupted_weights.values())

            return {
                "packet_id": packet.packet_id,
                "variant_id": self.variant.variant_id,
                "regime": regime_override if regime_override else "RFC",
                "natural_modalities": natural_modalities,
                "evaluated_modalities": active_modalities,
                "active_modalities": active_modalities,
                "is_forced_regime": is_forced_regime,
                "num_active": num_active,
                "weights": corrupted_weights,
                "raw_logits": router_result.logits,
                "masked_logits": router_result.masked_logits,
                "r_fusion": float(r_fusion),
                "u_sum": float(u_sum),
                "uncertainty_penalty": float(penalty),
                "dcri": float(dcri),
                "routing_entropy": router_result.routing_entropy,
                "dominant_modality": max(MODALITIES, key=lambda m: corrupted_weights[m]),
                "decision_tier": tier,
                "decision_tier_name": ACTION_KEYS[tier],
                "status": "SUCCESS_WITH_MASK_VIOLATION",
                "mask_violation": True,
                "injected_violation_channel": injected_channel,
                "injected_weight": corrupted_weights[injected_channel],
                "is_finite": is_finite,
            }

        # Standard execution
        router_result = self.router.route(router_input)
        weights = router_result.weights

        # Compute fused risk
        r_fusion = sum(weights[m] * float(getattr(packet, m).risk) for m in active_modalities)
        
        # Uncertainty sum across active modalities
        u_sum = sum(float(getattr(packet, m).uncertainty) for m in active_modalities)
        penalty = self.variant.delta * u_sum
        dcri = r_fusion - penalty
        tier = assign_decision_tier(dcri)

        # Validate finiteness
        import math
        is_finite = math.isfinite(r_fusion) and math.isfinite(dcri) and all(math.isfinite(w) for w in weights.values())
        if not is_finite:
            raise ValueError(f"Non-finite output detected for packet {packet.packet_id} under variant {self.variant.variant_id}")

        return {
            "packet_id": packet.packet_id,
            "variant_id": self.variant.variant_id,
            "regime": regime_override if regime_override else "RFC",
            "natural_modalities": natural_modalities,
            "evaluated_modalities": active_modalities,
            "active_modalities": active_modalities,
            "is_forced_regime": is_forced_regime,
            "num_active": num_active,
            "weights": {m: float(weights.get(m, 0.0)) for m in MODALITIES},
            "raw_logits": {m: float(router_result.logits.get(m, 0.0)) for m in MODALITIES},
            "masked_logits": {m: float(router_result.masked_logits.get(m, 0.0)) for m in MODALITIES},
            "r_fusion": float(r_fusion),
            "u_sum": float(u_sum),
            "uncertainty_penalty": float(penalty),
            "dcri": float(dcri),
            "routing_entropy": float(router_result.routing_entropy),
            "dominant_modality": router_result.dominant_modality,
            "decision_tier": tier,
            "decision_tier_name": ACTION_KEYS[tier],
            "status": router_result.status,
            "mask_violation": False,
            "is_finite": is_finite,
        }

    def evaluate_cohort(
        self,
        cohort: List[ControlledDecisionPacket],
        regime_override: Optional[str] = None,
        force_mask_violation: bool = False,
        raise_on_error: bool = False,
    ) -> List[Dict[str, Any]]:
        """Evaluates all packets in the cohort under this variant."""
        results = []
        for pkt in cohort:
            try:
                res = self.evaluate_packet(
                    pkt,
                    regime_override=regime_override,
                    force_mask_violation=force_mask_violation,
                )
                results.append(res)
            except Exception as e:
                self.packet_exceptions += 1
                if raise_on_error:
                    raise e
                # Record structured failure record so runner reaches artifact-based failure analysis
                results.append({
                    "packet_id": getattr(pkt, "packet_id", "UNKNOWN"),
                    "variant_id": self.variant.variant_id,
                    "regime": regime_override if regime_override else "RFC",
                    "natural_modalities": getattr(pkt, "available_modalities", []),
                    "evaluated_modalities": [],
                    "active_modalities": [],
                    "is_forced_regime": regime_override is not None,
                    "num_active": 0,
                    "weights": {"retina": 0.0, "foot": 0.0, "clinical": 0.0},
                    "raw_logits": {"retina": 0.0, "foot": 0.0, "clinical": 0.0},
                    "masked_logits": {"retina": 0.0, "foot": 0.0, "clinical": 0.0},
                    "r_fusion": None,
                    "u_sum": None,
                    "uncertainty_penalty": None,
                    "dcri": None,
                    "routing_entropy": None,
                    "dominant_modality": None,
                    "decision_tier": -1,
                    "decision_tier_name": "EXCEPTION_ERROR",
                    "status": f"EXCEPTION_OCCURRED: {type(e).__name__}",
                    "error": str(e),
                    "mask_violation": False,
                    "is_finite": False,
                })
        return results
