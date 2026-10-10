"""
FusionMedAI - Phase C11.14: DCRI Policy Evaluation Engine
Evaluates decision action assignments under Policy A (DCRI_0.10) and Policy B (R_fusion reference)
across the frozen cohort (N=500, seed 115).
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np

from src.fusion.baselines.decision_packet import ControlledDecisionPacket
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router.router_input import RouterInput, ModalityChannelInput
from .policy_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    DELTA_FROZEN,
    TAU_1_DEFAULT,
    TAU_2_DEFAULT,
    ACTION_ROUTINE_REVIEW,
    ACTION_ADDITIONAL_ASSESSMENT,
    ACTION_ESCALATION,
    ACTION_NAMES,
    ACTION_KEYS,
    MODALITIES,
    FLOAT_TOLERANCE,
    RESIDUAL_TOLERANCE,
)


def assign_action(score: float, tau_1: float, tau_2: float) -> int:
    """
    Assigns a decision action tier based on score and pre-specified thresholds.
    
    Rule:
        - score < tau_1           -> 0 (Routine Review)
        - tau_1 <= score < tau_2  -> 1 (Additional Assessment)
        - score >= tau_2          -> 2 (Escalation for Human Review)
        
    Note: Negative DCRI scores fall strictly into Routine Review (tier 0).
    
    Args:
        score: Continuous risk/decision index (DCRI_delta or R_fusion).
        tau_1: Low risk threshold.
        tau_2: High risk threshold (tau_2 > tau_1).
        
    Returns:
        Action category integer (0, 1, or 2).
    """
    if not math.isfinite(score):
        raise ValueError(f"Score must be a finite float, got: {score}")
    if not math.isfinite(tau_1) or not math.isfinite(tau_2):
        raise ValueError(f"Thresholds must be finite floats, got tau_1={tau_1}, tau_2={tau_2}")
    if tau_1 >= tau_2:
        raise ValueError(f"tau_1 ({tau_1}) must be strictly less than tau_2 ({tau_2})")
        
    if score < tau_1:
        return ACTION_ROUTINE_REVIEW
    elif score < tau_2:
        return ACTION_ADDITIONAL_ASSESSMENT
    else:
        return ACTION_ESCALATION


class PolicyEvaluationResult:
    """Encapsulates aggregate results of a policy comparison."""
    def __init__(
        self,
        tau_1: float,
        tau_2: float,
        delta: float,
        n_packets: int,
        policy_a_counts: Dict[str, int],
        policy_a_pcts: Dict[str, float],
        policy_b_counts: Dict[str, int],
        policy_b_pcts: Dict[str, float],
        transition_matrix: List[List[int]],
        reclassification_count: int,
        reclassification_rate: float,
        downgraded_count: int,
        downgraded_rate: float,
        upgraded_count: int,
        upgraded_rate: float,
        escalation_reduction_count: int,
        escalation_reduction_rate: float,
        per_packet_details: List[Dict[str, Any]],
    ):
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.delta = delta
        self.n_packets = n_packets
        self.policy_a_counts = policy_a_counts
        self.policy_a_pcts = policy_a_pcts
        self.policy_b_counts = policy_b_counts
        self.policy_b_pcts = policy_b_pcts
        self.transition_matrix = transition_matrix  # [action_b][action_a]
        self.reclassification_count = reclassification_count
        self.reclassification_rate = reclassification_rate
        self.downgraded_count = downgraded_count
        self.downgraded_rate = downgraded_rate
        self.upgraded_count = upgraded_count
        self.upgraded_rate = upgraded_rate
        self.escalation_reduction_count = escalation_reduction_count
        self.escalation_reduction_rate = escalation_reduction_rate
        self.per_packet_details = per_packet_details

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tau_1": self.tau_1,
            "tau_2": self.tau_2,
            "delta": self.delta,
            "n_packets": self.n_packets,
            "policy_a_dcri": {
                "counts": self.policy_a_counts,
                "percentages": self.policy_a_pcts,
            },
            "policy_b_fused_risk": {
                "counts": self.policy_b_counts,
                "percentages": self.policy_b_pcts,
            },
            "transition_matrix_fused_to_dcri": self.transition_matrix,
            "reclassifications": {
                "total_reclassified_count": self.reclassification_count,
                "total_reclassified_rate": self.reclassification_rate,
                "downgraded_count": self.downgraded_count,
                "downgraded_rate": self.downgraded_rate,
                "upgraded_count": self.upgraded_count,
                "upgraded_rate": self.upgraded_rate,
                "escalation_reduction_count": self.escalation_reduction_count,
                "escalation_reduction_rate": self.escalation_reduction_rate,
            },
        }


class PolicyEvaluator:
    """
    Evaluates policy assignments over the frozen cohort.
    Ensures upstream weights, calibrated risks, and uncertainties are computed once.
    """

    def __init__(self, delta: float = DELTA_FROZEN):
        self.delta = delta
        self.coeff = RouterCoefficients(
            alpha=ALPHA_REF,
            beta=BETA_REF,
            gamma=GAMMA_REF,
            eta=ETA_REF,
        )
        self.router = ACARAUv2Router(coefficients=self.coeff)

    def precompute_packet_base_state(
        self,
        cohort: List[ControlledDecisionPacket],
    ) -> List[Dict[str, Any]]:
        """
        Precomputes base fusion state (w_i, r_i, U_i, R_fusion, U_sum) once per packet.
        """
        base_states: List[Dict[str, Any]] = []

        for packet in cohort:
            router_input = packet.to_router_input()
            decision_output = self.router.route(router_input)

            # Active modalities
            active_mods = []
            for mod_name in MODALITIES:
                rec = getattr(packet, mod_name, None)
                if rec is not None and rec.availability:
                    active_mods.append(mod_name)

            weights = {m: decision_output.weights.get(m, 0.0) for m in active_mods}
            risks = {}
            uncertainties = {}
            r_fusion = 0.0
            u_sum = 0.0

            for m in active_mods:
                rec = getattr(packet, m)
                r_i = float(rec.risk)
                u_i = float(rec.uncertainty)
                w_i = float(weights[m])
                risks[m] = r_i
                uncertainties[m] = u_i
                r_fusion += w_i * r_i
                u_sum += u_i

            # Bitwise invariant checks
            sum_w = sum(weights.values())
            if abs(sum_w - 1.0) > 1e-6:
                raise RuntimeError(
                    f"Packet {packet.packet_id} violated simplex constraint: sum(w) = {sum_w}"
                )

            dcri = r_fusion - self.delta * u_sum

            # Exact analytical identity check
            residual = abs((r_fusion - self.delta * u_sum) - dcri)
            if residual > RESIDUAL_TOLERANCE:
                raise RuntimeError(
                    f"Packet {packet.packet_id} residual tolerance violation: {residual}"
                )

            base_states.append({
                "packet_id": packet.packet_id,
                "weights": weights,
                "risks": risks,
                "uncertainties": uncertainties,
                "r_fusion": float(r_fusion),
                "u_sum": float(u_sum),
                "dcri": float(dcri),
                "active_modalities": active_mods,
                "regime_cardinality": len(active_mods),
            })

        return base_states

    def evaluate_policies(
        self,
        base_states: List[Dict[str, Any]],
        tau_1: float = TAU_1_DEFAULT,
        tau_2: float = TAU_2_DEFAULT,
    ) -> PolicyEvaluationResult:
        """
        Evaluates Policy A (DCRI) vs Policy B (Fused Risk) for a given threshold pair.
        """
        n = len(base_states)
        if n == 0:
            raise ValueError("base_states list cannot be empty.")

        policy_a_counts = {ACTION_KEYS[0]: 0, ACTION_KEYS[1]: 0, ACTION_KEYS[2]: 0}
        policy_b_counts = {ACTION_KEYS[0]: 0, ACTION_KEYS[1]: 0, ACTION_KEYS[2]: 0}
        transition_matrix = [[0 for _ in range(3)] for _ in range(3)]  # [b][a]

        reclass_count = 0
        downgrade_count = 0
        upgrade_count = 0
        escalation_reduction_count = 0

        per_packet_details: List[Dict[str, Any]] = []

        for item in base_states:
            r_fusion = item["r_fusion"]
            dcri = item["dcri"]

            action_b = assign_action(r_fusion, tau_1, tau_2)  # Fused risk reference
            action_a = assign_action(dcri, tau_1, tau_2)      # DCRI policy

            policy_b_counts[ACTION_KEYS[action_b]] += 1
            policy_a_counts[ACTION_KEYS[action_a]] += 1
            transition_matrix[action_b][action_a] += 1

            is_reclassified = (action_a != action_b)
            delta_action = action_a - action_b

            if is_reclassified:
                reclass_count += 1
                if action_a < action_b:
                    downgrade_count += 1
                else:
                    upgrade_count += 1

            if action_b == ACTION_ESCALATION and action_a < ACTION_ESCALATION:
                escalation_reduction_count += 1

            per_packet_details.append({
                "packet_id": item["packet_id"],
                "r_fusion": r_fusion,
                "u_sum": item["u_sum"],
                "dcri": dcri,
                "action_fused_risk": action_b,
                "action_fused_name": ACTION_NAMES[action_b],
                "action_dcri": action_a,
                "action_dcri_name": ACTION_NAMES[action_a],
                "is_reclassified": is_reclassified,
                "action_shift": delta_action,
                "regime_cardinality": item["regime_cardinality"],
            })

        policy_a_pcts = {k: round(v / n * 100.0, 4) for k, v in policy_a_counts.items()}
        policy_b_pcts = {k: round(v / n * 100.0, 4) for k, v in policy_b_counts.items()}

        return PolicyEvaluationResult(
            tau_1=tau_1,
            tau_2=tau_2,
            delta=self.delta,
            n_packets=n,
            policy_a_counts=policy_a_counts,
            policy_a_pcts=policy_a_pcts,
            policy_b_counts=policy_b_counts,
            policy_b_pcts=policy_b_pcts,
            transition_matrix=transition_matrix,
            reclassification_count=reclass_count,
            reclassification_rate=round(reclass_count / n, 6),
            downgraded_count=downgrade_count,
            downgraded_rate=round(downgrade_count / n, 6),
            upgraded_count=upgrade_count,
            upgraded_rate=round(upgrade_count / n, 6),
            escalation_reduction_count=escalation_reduction_count,
            escalation_reduction_rate=round(escalation_reduction_count / n, 6),
            per_packet_details=per_packet_details,
        )
