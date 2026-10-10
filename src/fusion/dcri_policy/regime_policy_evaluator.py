"""
FusionMedAI - Phase C11.14: DCRI Regime Policy Evaluator
Evaluates decision policy behavior across all 7 active modality availability regimes
plus the fail-closed EMPTY regime under fixed delta=0.10 and reference thresholds (0.20, 0.40).
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np

from src.fusion.baselines.decision_packet import ControlledDecisionPacket
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router.router_input import RouterInput, ModalityChannelInput
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)
from .policy_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    DELTA_FROZEN,
    TAU_1_DEFAULT,
    TAU_2_DEFAULT,
    ACTIVE_REGIMES,
    ALL_REGIMES,
    MODALITIES,
    ACTION_KEYS,
)
from .policy_engine import assign_action

REGIME_MODALITY_MAP: Dict[str, Tuple[str, ...]] = {
    "R": ("retina",),
    "F": ("foot",),
    "C": ("clinical",),
    "RF": ("retina", "foot"),
    "RC": ("retina", "clinical"),
    "FC": ("foot", "clinical"),
    "RFC": ("retina", "foot", "clinical"),
}


class RegimePolicyEvaluator:
    """
    Evaluates policy assignments stratified by modality availability regime.
    """

    def __init__(
        self,
        delta: float = DELTA_FROZEN,
        tau_1: float = TAU_1_DEFAULT,
        tau_2: float = TAU_2_DEFAULT,
    ):
        self.delta = delta
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.coeff = RouterCoefficients(
            alpha=ALPHA_REF,
            beta=BETA_REF,
            gamma=GAMMA_REF,
            eta=ETA_REF,
        )
        self.router = ACARAUv2Router(coefficients=self.coeff)

    def evaluate_regimes(
        self,
        cohort: List[ControlledDecisionPacket],
    ) -> Dict[str, Any]:
        """
        Evaluates all active regimes across the cohort.
        
        Returns:
            Dictionary containing metrics and policy distributions per regime.
        """
        regime_results: Dict[str, Any] = {}
        n_cohort = len(cohort)

        for regime in ACTIVE_REGIMES:
            active_mods = REGIME_MODALITY_MAP[regime]
            r_fusions: List[float] = []
            u_sums: List[float] = []
            dcris: List[float] = []

            policy_a_counts = {ACTION_KEYS[0]: 0, ACTION_KEYS[1]: 0, ACTION_KEYS[2]: 0}
            policy_b_counts = {ACTION_KEYS[0]: 0, ACTION_KEYS[1]: 0, ACTION_KEYS[2]: 0}
            transition_matrix = [[0 for _ in range(3)] for _ in range(3)]

            reclass_count = 0
            downgrade_count = 0
            negative_count = 0
            escalation_red_count = 0

            for pkt in cohort:
                ch_map = {}
                for m in MODALITIES:
                    rec = getattr(pkt, m)
                    is_active = (m in active_mods)
                    ch_map[m] = ModalityChannelInput(
                        modality=m,
                        confidence=float(rec.confidence),
                        reliability=float(rec.reliability),
                        uncertainty=float(rec.uncertainty),
                        quality=float(rec.quality) if is_active else 0.0,
                        availability=is_active,
                    )
                r_in = RouterInput(retina=ch_map["retina"], foot=ch_map["foot"], clinical=ch_map["clinical"])
                r_out = self.router.route(r_in)

                weights = {m: r_out.weights.get(m, 0.0) for m in active_mods}
                r_fusion = 0.0
                u_sum = 0.0

                for m in active_mods:
                    rec = getattr(pkt, m)
                    r_i = float(rec.risk)
                    u_i = float(rec.uncertainty)
                    w_i = float(weights[m])
                    r_fusion += w_i * r_i
                    u_sum += u_i

                # Simplex verification
                sum_w = sum(weights.values())
                if abs(sum_w - 1.0) > 1e-5:
                    raise RuntimeError(f"Simplex violated in regime {regime}: {sum_w}")

                dcri = r_fusion - self.delta * u_sum

                if dcri < 0.0:
                    negative_count += 1

                action_b = assign_action(r_fusion, self.tau_1, self.tau_2)
                action_a = assign_action(dcri, self.tau_1, self.tau_2)

                policy_b_counts[ACTION_KEYS[action_b]] += 1
                policy_a_counts[ACTION_KEYS[action_a]] += 1
                transition_matrix[action_b][action_a] += 1

                if action_a != action_b:
                    reclass_count += 1
                    if action_a < action_b:
                        downgrade_count += 1

                if action_b == 2 and action_a < 2:
                    escalation_red_count += 1

                r_fusions.append(float(r_fusion))
                u_sums.append(float(u_sum))
                dcris.append(float(dcri))

            mean_r_fusion = float(np.mean(r_fusions))
            mean_u_sum = float(np.mean(u_sums))
            mean_dcri = float(np.mean(dcris))

            regime_results[regime] = {
                "regime": regime,
                "n_packets": n_cohort,
                "cardinality": len(regime),
                "mean_r_fusion": round(mean_r_fusion, 6),
                "mean_u_sum": round(mean_u_sum, 6),
                "mean_dcri": round(mean_dcri, 6),
                "negative_count": negative_count,
                "negative_rate": round(negative_count / n_cohort, 6),
                "policy_a_dcri": {
                    "counts": policy_a_counts,
                    "percentages": {k: round(v / n_cohort * 100.0, 4) for k, v in policy_a_counts.items()},
                },
                "policy_b_fused_risk": {
                    "counts": policy_b_counts,
                    "percentages": {k: round(v / n_cohort * 100.0, 4) for k, v in policy_b_counts.items()},
                },
                "reclassifications": {
                    "total_count": reclass_count,
                    "rate": round(reclass_count / n_cohort, 6),
                    "downgraded_count": downgrade_count,
                    "downgraded_rate": round(downgrade_count / n_cohort, 6),
                    "escalation_reduction_count": escalation_red_count,
                    "escalation_reduction_rate": round(escalation_red_count / n_cohort, 6),
                },
                "transition_matrix": transition_matrix,
            }

        # EMPTY Regime Fail-Closed Verification
        empty_input = RouterInput(
            retina=ModalityChannelInput(modality="retina", confidence=0.0, reliability=FROZEN_RETINA_RELIABILITY, uncertainty=1.0, quality=0.0, availability=False),
            foot=ModalityChannelInput(modality="foot", confidence=0.0, reliability=FROZEN_FOOT_RELIABILITY, uncertainty=1.0, quality=0.0, availability=False),
            clinical=ModalityChannelInput(modality="clinical", confidence=0.0, reliability=FROZEN_CLINICAL_RELIABILITY, uncertainty=1.0, quality=0.0, availability=False),
        )
        empty_output = self.router.route(empty_input)
        regime_results["EMPTY"] = {
            "regime": "EMPTY",
            "n_packets": n_cohort,
            "cardinality": 0,
            "fail_closed_status": empty_output.status,
            "decision_output_available": False,
            "sentinel_risk": 0.0,
            "routing_weights": empty_output.weights,
        }

        return regime_results
