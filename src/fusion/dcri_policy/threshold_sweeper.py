"""
FusionMedAI - Phase C11.14: DCRI Threshold Sensitivity Sweeper
Sweeps all pre-specified (tau_1, tau_2) threshold pairs across the frozen cohort.
"""

from typing import Dict, List, Tuple, Any
from .policy_config import (
    TAU_1_GRID,
    TAU_2_GRID,
    DELTA_FROZEN,
)
from .policy_engine import PolicyEvaluator, PolicyEvaluationResult


class ThresholdSensitivitySweeper:
    """
    Evaluates policy sensitivity across a pre-specified 2-D threshold grid.
    """

    def __init__(
        self,
        tau_1_grid: Tuple[float, ...] = TAU_1_GRID,
        tau_2_grid: Tuple[float, ...] = TAU_2_GRID,
        delta: float = DELTA_FROZEN,
    ):
        self.tau_1_grid = tau_1_grid
        self.tau_2_grid = tau_2_grid
        self.delta = delta
        self.evaluator = PolicyEvaluator(delta=delta)

    def sweep(
        self,
        base_states: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Executes threshold grid sweep.
        
        Returns:
            List of evaluation results for each valid (tau_1, tau_2) pair.
        """
        results: List[Dict[str, Any]] = []

        for tau_1 in self.tau_1_grid:
            for tau_2 in self.tau_2_grid:
                if tau_1 >= tau_2:
                    continue  # Enforce strict monotonicity tau_1 < tau_2

                eval_res = self.evaluator.evaluate_policies(
                    base_states=base_states,
                    tau_1=tau_1,
                    tau_2=tau_2,
                )
                
                results.append({
                    "tau_1": tau_1,
                    "tau_2": tau_2,
                    "delta": self.delta,
                    "policy_a_dcri": {
                        "counts": eval_res.policy_a_counts,
                        "percentages": eval_res.policy_a_pcts,
                    },
                    "policy_b_fused_risk": {
                        "counts": eval_res.policy_b_counts,
                        "percentages": eval_res.policy_b_pcts,
                    },
                    "reclassifications": {
                        "total_count": eval_res.reclassification_count,
                        "rate": eval_res.reclassification_rate,
                        "downgraded_count": eval_res.downgraded_count,
                        "downgraded_rate": eval_res.downgraded_rate,
                        "upgraded_count": eval_res.upgraded_count,
                        "upgraded_rate": eval_res.upgraded_rate,
                        "escalation_reduction_count": eval_res.escalation_reduction_count,
                        "escalation_reduction_rate": eval_res.escalation_reduction_rate,
                    },
                    "transition_matrix": eval_res.transition_matrix,
                })

        return results
