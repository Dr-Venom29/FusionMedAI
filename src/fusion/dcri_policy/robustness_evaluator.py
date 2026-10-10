"""
FusionMedAI - Phase C11.14: DCRI Robustness Evaluator
Evaluates decision policy robustness under controlled perturbations:
1. Uncertainty scaling (+/- 20%)
2. Threshold boundary jitter (+/- 0.02)
Note: Modality availability regimes are evaluated independently in regime_policy_evaluator.py.
"""

from typing import Dict, List, Tuple, Any
import numpy as np

from .policy_config import (
    DELTA_FROZEN,
    TAU_1_DEFAULT,
    TAU_2_DEFAULT,
    ACTION_KEYS,
)
from .policy_engine import assign_action, PolicyEvaluator


class RobustnessEvaluator:
    """
    Evaluates policy robustness under controlled operational perturbations.
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
        self.evaluator = PolicyEvaluator(delta=delta)

    def evaluate_uncertainty_scaling(
        self,
        base_states: List[Dict[str, Any]],
        scale_factors: Tuple[float, ...] = (0.80, 0.90, 1.00, 1.10, 1.20),
    ) -> List[Dict[str, Any]]:
        """
        Evaluates policy action shifts under global uncertainty scaling U_sum -> U_sum * scale.
        """
        results: List[Dict[str, Any]] = []
        n = len(base_states)

        for scale in scale_factors:
            policy_a_counts = {ACTION_KEYS[0]: 0, ACTION_KEYS[1]: 0, ACTION_KEYS[2]: 0}
            reclass_count = 0
            downgrade_count = 0
            negative_count = 0

            for item in base_states:
                r_fusion = item["r_fusion"]
                u_sum_scaled = item["u_sum"] * scale
                dcri_scaled = r_fusion - self.delta * u_sum_scaled

                if dcri_scaled < 0.0:
                    negative_count += 1

                action_b = assign_action(r_fusion, self.tau_1, self.tau_2)
                action_a = assign_action(dcri_scaled, self.tau_1, self.tau_2)

                policy_a_counts[ACTION_KEYS[action_a]] += 1

                if action_a != action_b:
                    reclass_count += 1
                    if action_a < action_b:
                        downgrade_count += 1

            results.append({
                "scale_factor": scale,
                "n_packets": n,
                "negative_count": negative_count,
                "negative_rate": round(negative_count / n, 6),
                "policy_a_counts": policy_a_counts,
                "policy_a_percentages": {k: round(v / n * 100.0, 4) for k, v in policy_a_counts.items()},
                "reclassification_count": reclass_count,
                "reclassification_rate": round(reclass_count / n, 6),
                "downgraded_count": downgrade_count,
                "downgraded_rate": round(downgrade_count / n, 6),
            })

        return results

    def evaluate_threshold_jitter(
        self,
        base_states: List[Dict[str, Any]],
        jitter_offsets: Tuple[float, ...] = (-0.02, -0.01, 0.00, +0.01, +0.02),
    ) -> List[Dict[str, Any]]:
        """
        Evaluates policy sensitivity to joint threshold translation perturbations
        where both decision boundaries shift simultaneously: (tau_1 + jitter, tau_2 + jitter).
        """
        results: List[Dict[str, Any]] = []
        n = len(base_states)

        for jitter in jitter_offsets:
            tau_1 = round(self.tau_1 + jitter, 4)
            tau_2 = round(self.tau_2 + jitter, 4)

            eval_res = self.evaluator.evaluate_policies(
                base_states=base_states,
                tau_1=tau_1,
                tau_2=tau_2,
            )

            results.append({
                "jitter_offset": jitter,
                "tau_1": tau_1,
                "tau_2": tau_2,
                "policy_a_counts": eval_res.policy_a_counts,
                "policy_a_percentages": eval_res.policy_a_pcts,
                "policy_b_counts": eval_res.policy_b_counts,
                "policy_b_percentages": eval_res.policy_b_pcts,
                "reclassification_count": eval_res.reclassification_count,
                "reclassification_rate": eval_res.reclassification_rate,
                "escalation_reduction_count": eval_res.escalation_reduction_count,
                "escalation_reduction_rate": eval_res.escalation_reduction_rate,
            })

        return results
