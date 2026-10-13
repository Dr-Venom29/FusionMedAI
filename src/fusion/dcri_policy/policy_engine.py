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
    AMBIGUITY_U_SUM_THRESHOLD,
    AMBIGUITY_MIN_QUALITY_THRESHOLD,
    FLAG_REPEAT_EXAM_AMBIGUOUS_EVIDENCE,
    REASON_CLEAR_EVIDENCE,
    REASON_HIGH_UNCERTAINTY,
    REASON_DEGRADED_QUALITY,
    REASON_UNCERTAINTY_AND_QUALITY,
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


def detect_evidence_ambiguity(
    u_sum: float,
    qualities: Optional[Dict[str, float]] = None,
    u_sum_threshold: float = AMBIGUITY_U_SUM_THRESHOLD,
    min_quality_threshold: float = AMBIGUITY_MIN_QUALITY_THRESHOLD,
) -> Tuple[bool, str]:
    """
    Evaluates evidence ambiguity based on cumulative uncertainty and modality signal qualities.

    Boundary Semantics:
        - High uncertainty: u_sum >= u_sum_threshold (inclusive: equality at u_sum_threshold is ambiguous).
        - Degraded quality: q < min_quality_threshold (strict: equality at min_quality_threshold is acceptable).

    Returns:
        (is_ambiguous, reason_code)
    """
    if not math.isfinite(u_sum):
        raise ValueError(f"u_sum must be a finite float, got: {u_sum}")
    if not math.isfinite(u_sum_threshold) or not math.isfinite(min_quality_threshold):
        raise ValueError(f"Ambiguity thresholds must be finite floats, got: u_sum_threshold={u_sum_threshold}, min_quality_threshold={min_quality_threshold}")

    high_u = (u_sum >= u_sum_threshold)
    degraded_q = False
    if qualities:
        for mod_name, q in qualities.items():
            if not math.isfinite(q):
                raise ValueError(f"Modality quality for {mod_name} must be a finite float, got: {q}")
            if q < min_quality_threshold:
                degraded_q = True
                break

    if high_u and degraded_q:
        return True, REASON_UNCERTAINTY_AND_QUALITY
    elif high_u:
        return True, REASON_HIGH_UNCERTAINTY
    elif degraded_q:
        return True, REASON_DEGRADED_QUALITY
    else:
        return False, REASON_CLEAR_EVIDENCE



def apply_safeguard_policy(
    action_base: int,
    action_dcri: int,
    is_ambiguous: bool,
    ambiguity_reason: str,
) -> Tuple[int, bool, Optional[str], Optional[str]]:
    """
    Applies the decision-policy safeguard against uncertainty-induced downgrades.
    
    Rule:
        - If is_ambiguous and action_dcri < action_base:
            Block downgrade -> action_final = action_base
            downgrade_blocked = True
        - Else:
            action_final = action_dcri
            downgrade_blocked = False
        - If is_ambiguous:
            flag = FLAG_REPEAT_EXAM_AMBIGUOUS_EVIDENCE (preserved even if base tier is low)
            
    Returns:
        (action_final, downgrade_blocked, flag, reason)
    """
    flag = FLAG_REPEAT_EXAM_AMBIGUOUS_EVIDENCE if is_ambiguous else None
    
    if is_ambiguous and (action_dcri < action_base):
        action_final = action_base
        downgrade_blocked = True
        reason = f"BLOCKED_DOWNGRADE_{ambiguity_reason}"
    else:
        action_final = action_dcri
        downgrade_blocked = False
        reason = ambiguity_reason if is_ambiguous else REASON_CLEAR_EVIDENCE
        
    return action_final, downgrade_blocked, flag, reason



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
            qualities = {}
            r_fusion = 0.0
            u_sum = 0.0

            for m in active_mods:
                rec = getattr(packet, m)
                r_i = float(rec.risk)
                u_i = float(rec.uncertainty)
                q_i = float(rec.quality) if hasattr(rec, "quality") else 1.0
                w_i = float(weights[m])
                risks[m] = r_i
                uncertainties[m] = u_i
                qualities[m] = q_i
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
                "qualities": qualities,
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

    def evaluate_safeguarded_policy(
        self,
        base_states: List[Dict[str, Any]],
        tau_1: float = TAU_1_DEFAULT,
        tau_2: float = TAU_2_DEFAULT,
        u_sum_threshold: float = AMBIGUITY_U_SUM_THRESHOLD,
        min_quality_threshold: float = AMBIGUITY_MIN_QUALITY_THRESHOLD,
    ) -> "SafeguardedPolicyEvaluationResult":
        """
        Evaluates the Safeguarded DCRI Policy (Policy S) alongside unconstrained DCRI (Policy A)
        and base fused risk (Policy B).
        
        Under Policy S:
            - If evidence is ambiguous (u_sum >= u_sum_threshold or min(quality) < min_quality_threshold):
                Downgrade below base-risk tier is blocked: T_final = max(T_base, T_dcri) = T_base
                downgrade_blocked = True
                flag = REPEAT_EXAM_AMBIGUOUS_EVIDENCE
            - Else:
                T_final = T_dcri
                downgrade_blocked = False
                flag = None
        """
        n = len(base_states)
        if n == 0:
            raise ValueError("base_states list cannot be empty.")

        policy_s_counts = {ACTION_KEYS[0]: 0, ACTION_KEYS[1]: 0, ACTION_KEYS[2]: 0}
        policy_a_counts = {ACTION_KEYS[0]: 0, ACTION_KEYS[1]: 0, ACTION_KEYS[2]: 0}
        policy_b_counts = {ACTION_KEYS[0]: 0, ACTION_KEYS[1]: 0, ACTION_KEYS[2]: 0}

        attempted_downgrades_count = 0
        blocked_downgrades_count = 0
        permitted_downgrades_count = 0
        ambiguous_flags_count = 0
        reclass_count_s_vs_b = 0
        downgrades_s_vs_b = 0

        per_packet_details: List[Dict[str, Any]] = []

        for item in base_states:
            r_fusion = item["r_fusion"]
            dcri = item["dcri"]
            u_sum = item["u_sum"]
            qualities = item.get("qualities", {})

            action_b = assign_action(r_fusion, tau_1, tau_2)
            action_a = assign_action(dcri, tau_1, tau_2)

            is_ambiguous, ambiguity_reason = detect_evidence_ambiguity(
                u_sum=u_sum,
                qualities=qualities,
                u_sum_threshold=u_sum_threshold,
                min_quality_threshold=min_quality_threshold,
            )

            action_s, downgrade_blocked, flag, reason = apply_safeguard_policy(
                action_base=action_b,
                action_dcri=action_a,
                is_ambiguous=is_ambiguous,
                ambiguity_reason=ambiguity_reason,
            )

            policy_b_counts[ACTION_KEYS[action_b]] += 1
            policy_a_counts[ACTION_KEYS[action_a]] += 1
            policy_s_counts[ACTION_KEYS[action_s]] += 1

            is_attempted_downgrade = (action_a < action_b)
            if is_attempted_downgrade:
                attempted_downgrades_count += 1
                if downgrade_blocked:
                    blocked_downgrades_count += 1
                else:
                    permitted_downgrades_count += 1

            if flag is not None:
                ambiguous_flags_count += 1

            if action_s != action_b:
                reclass_count_s_vs_b += 1
                if action_s < action_b:
                    downgrades_s_vs_b += 1

            per_packet_details.append({
                "packet_id": item["packet_id"],
                "r_fusion": r_fusion,
                "u_sum": u_sum,
                "dcri": dcri,
                "action_base": action_b,
                "action_base_name": ACTION_NAMES[action_b],
                "action_dcri": action_a,
                "action_dcri_name": ACTION_NAMES[action_a],
                "action_safeguarded": action_s,
                "action_safeguarded_name": ACTION_NAMES[action_s],
                "is_ambiguous": is_ambiguous,
                "ambiguity_reason": ambiguity_reason,
                "is_attempted_downgrade": is_attempted_downgrade,
                "downgrade_blocked": downgrade_blocked,
                "flag": flag,
                "decision_reason": reason,
            })

        policy_s_pcts = {k: round(v / n * 100.0, 4) for k, v in policy_s_counts.items()}
        policy_a_pcts = {k: round(v / n * 100.0, 4) for k, v in policy_a_counts.items()}
        policy_b_pcts = {k: round(v / n * 100.0, 4) for k, v in policy_b_counts.items()}

        return SafeguardedPolicyEvaluationResult(
            tau_1=tau_1,
            tau_2=tau_2,
            delta=self.delta,
            u_sum_threshold=u_sum_threshold,
            min_quality_threshold=min_quality_threshold,
            n_packets=n,
            policy_b_counts=policy_b_counts,
            policy_b_pcts=policy_b_pcts,
            policy_a_counts=policy_a_counts,
            policy_a_pcts=policy_a_pcts,
            policy_s_counts=policy_s_counts,
            policy_s_pcts=policy_s_pcts,
            attempted_downgrades_count=attempted_downgrades_count,
            blocked_downgrades_count=blocked_downgrades_count,
            permitted_downgrades_count=permitted_downgrades_count,
            ambiguous_flags_count=ambiguous_flags_count,
            reclass_count_s_vs_b=reclass_count_s_vs_b,
            downgrades_s_vs_b=downgrades_s_vs_b,
            per_packet_details=per_packet_details,
        )


class SafeguardedPolicyEvaluationResult:
    """Encapsulates results of safeguarded policy evaluation."""
    def __init__(
        self,
        tau_1: float,
        tau_2: float,
        delta: float,
        u_sum_threshold: float,
        min_quality_threshold: float,
        n_packets: int,
        policy_b_counts: Dict[str, int],
        policy_b_pcts: Dict[str, float],
        policy_a_counts: Dict[str, int],
        policy_a_pcts: Dict[str, float],
        policy_s_counts: Dict[str, int],
        policy_s_pcts: Dict[str, float],
        attempted_downgrades_count: int,
        blocked_downgrades_count: int,
        permitted_downgrades_count: int,
        ambiguous_flags_count: int,
        reclass_count_s_vs_b: int,
        downgrades_s_vs_b: int,
        per_packet_details: List[Dict[str, Any]],
    ):
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.delta = delta
        self.u_sum_threshold = u_sum_threshold
        self.min_quality_threshold = min_quality_threshold
        self.n_packets = n_packets
        self.policy_b_counts = policy_b_counts
        self.policy_b_pcts = policy_b_pcts
        self.policy_a_counts = policy_a_counts
        self.policy_a_pcts = policy_a_pcts
        self.policy_s_counts = policy_s_counts
        self.policy_s_pcts = policy_s_pcts
        self.attempted_downgrades_count = attempted_downgrades_count
        self.blocked_downgrades_count = blocked_downgrades_count
        self.permitted_downgrades_count = permitted_downgrades_count
        self.ambiguous_flags_count = ambiguous_flags_count
        self.reclass_count_s_vs_b = reclass_count_s_vs_b
        self.downgrades_s_vs_b = downgrades_s_vs_b
        self.per_packet_details = per_packet_details

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tau_1": self.tau_1,
            "tau_2": self.tau_2,
            "delta": self.delta,
            "u_sum_threshold": self.u_sum_threshold,
            "min_quality_threshold": self.min_quality_threshold,
            "n_packets": self.n_packets,
            "policy_b_base_risk": {
                "counts": self.policy_b_counts,
                "percentages": self.policy_b_pcts,
            },
            "policy_a_unconstrained_dcri": {
                "counts": self.policy_a_counts,
                "percentages": self.policy_a_pcts,
            },
            "policy_s_safeguarded_dcri": {
                "counts": self.policy_s_counts,
                "percentages": self.policy_s_pcts,
            },
            "safeguard_metrics": {
                "attempted_downgrades_count": self.attempted_downgrades_count,
                "attempted_downgrades_rate": round(self.attempted_downgrades_count / self.n_packets, 6),
                "blocked_downgrades_count": self.blocked_downgrades_count,
                "blocked_downgrades_rate": round(self.blocked_downgrades_count / self.n_packets, 6),
                "permitted_downgrades_count": self.permitted_downgrades_count,
                "permitted_downgrades_rate": round(self.permitted_downgrades_count / self.n_packets, 6),
                "blocked_ratio_vs_attempted": round(self.blocked_downgrades_count / self.attempted_downgrades_count, 6) if self.attempted_downgrades_count > 0 else 0.0,
                "ambiguous_flags_count": self.ambiguous_flags_count,
                "ambiguous_flags_rate": round(self.ambiguous_flags_count / self.n_packets, 6),
                "reclass_count_s_vs_b": self.reclass_count_s_vs_b,
                "reclass_rate_s_vs_b": round(self.reclass_count_s_vs_b / self.n_packets, 6),
                "downgrades_s_vs_b": self.downgrades_s_vs_b,
                "downgrades_rate_s_vs_b": round(self.downgrades_s_vs_b / self.n_packets, 6),
            },
        }

