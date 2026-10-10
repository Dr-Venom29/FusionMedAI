"""
FusionMedAI - Phase C11.14: DCRI Statistical Analysis Module
Computes non-parametric paired bootstrap confidence intervals (B=1000),
Wilson score intervals for proportions, and paired reclassification statistics.
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np
from scipy import stats

from .policy_config import (
    N_BOOTSTRAPS,
    SEED,
    BOOTSTRAP_CI_ALPHA,
    ACTION_KEYS,
)


def compute_wilson_ci(
    count: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Computes Wilson score confidence interval for a binomial proportion.
    """
    if n == 0:
        return (0.0, 0.0)
    p = count / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_std = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)

    lower = (centre_adjusted_probability - z * adjusted_std) / denominator
    upper = (centre_adjusted_probability + z * adjusted_std) / denominator
    return (max(0.0, float(lower)), min(1.0, float(upper)))


class PolicyStatisticalAnalyzer:
    """
    Computes paired statistical summaries across policy assignments.
    """

    def __init__(
        self,
        n_bootstraps: int = N_BOOTSTRAPS,
        seed: int = SEED,
        alpha: float = BOOTSTRAP_CI_ALPHA,
    ):
        self.n_bootstraps = n_bootstraps
        self.seed = seed
        self.alpha = alpha

    def compute_paired_statistics(
        self,
        base_states: List[Dict[str, Any]],
        per_packet_details: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Runs paired bootstrap and calculates comprehensive statistical summaries.
        """
        if len(base_states) != len(per_packet_details):
            raise ValueError(
                f"Input length mismatch: base_states ({len(base_states)}) != per_packet_details ({len(per_packet_details)})"
            )
        if len(per_packet_details) == 0:
            raise ValueError("Input lists cannot be empty")

        for i, (b, p) in enumerate(zip(base_states, per_packet_details)):
            if b.get("packet_id") != p.get("packet_id"):
                raise ValueError(
                    f"Packet ID alignment mismatch at index {i}: base_state '{b.get('packet_id')}' != per_packet_detail '{p.get('packet_id')}'"
                )

        n = len(per_packet_details)
        rng = np.random.default_rng(self.seed)

        # Arrays for bootstrap
        r_fusions = np.array([p["r_fusion"] for p in per_packet_details], dtype=np.float64)
        dcris = np.array([p["dcri"] for p in per_packet_details], dtype=np.float64)
        reclass_flags = np.array([1 if p["is_reclassified"] else 0 for p in per_packet_details], dtype=np.int32)
        downgrade_flags = np.array([1 if p["action_shift"] < 0 else 0 for p in per_packet_details], dtype=np.int32)
        escalation_red_flags = np.array([1 if (p["action_fused_risk"] == 2 and p["action_dcri"] < 2) else 0 for p in per_packet_details], dtype=np.int32)

        # Observed differences
        score_diffs = r_fusions - dcris  # == delta * u_sum
        obs_mean_diff = float(np.mean(score_diffs))
        obs_reclass_rate = float(np.mean(reclass_flags))
        obs_downgrade_rate = float(np.mean(downgrade_flags))
        obs_escalation_red_rate = float(np.mean(escalation_red_flags))

        # Bootstrap resampling
        boot_diff_means: List[float] = []
        boot_reclass_rates: List[float] = []
        boot_downgrade_rates: List[float] = []
        boot_escalation_red_rates: List[float] = []

        for _ in range(self.n_bootstraps):
            idx = rng.choice(n, size=n, replace=True)
            boot_diff_means.append(float(np.mean(score_diffs[idx])))
            boot_reclass_rates.append(float(np.mean(reclass_flags[idx])))
            boot_downgrade_rates.append(float(np.mean(downgrade_flags[idx])))
            boot_escalation_red_rates.append(float(np.mean(escalation_red_flags[idx])))

        # Percentile intervals
        lower_pct = 100.0 * (self.alpha / 2.0)
        upper_pct = 100.0 * (1.0 - self.alpha / 2.0)

        diff_ci = (
            float(np.percentile(boot_diff_means, lower_pct)),
            float(np.percentile(boot_diff_means, upper_pct)),
        )
        reclass_ci = (
            float(np.percentile(boot_reclass_rates, lower_pct)),
            float(np.percentile(boot_reclass_rates, upper_pct)),
        )
        downgrade_ci = (
            float(np.percentile(boot_downgrade_rates, lower_pct)),
            float(np.percentile(boot_downgrade_rates, upper_pct)),
        )
        escalation_red_ci = (
            float(np.percentile(boot_escalation_red_rates, lower_pct)),
            float(np.percentile(boot_escalation_red_rates, upper_pct)),
        )

        # Wilson score intervals for primary rates
        reclass_wilson = compute_wilson_ci(int(np.sum(reclass_flags)), n)
        downgrade_wilson = compute_wilson_ci(int(np.sum(downgrade_flags)), n)
        escalation_red_wilson = compute_wilson_ci(int(np.sum(escalation_red_flags)), n)

        return {
            "n_packets": n,
            "seed": self.seed,
            "n_bootstraps": self.n_bootstraps,
            "score_discount": {
                "observed_mean_difference": round(obs_mean_diff, 6),
                "bootstrap_mean": round(float(np.mean(boot_diff_means)), 6),
                "bootstrap_std_error": round(float(np.std(boot_diff_means, ddof=1)), 6),
                "bootstrap_95_ci": [round(diff_ci[0], 6), round(diff_ci[1], 6)],
            },
            "reclassifications": {
                "observed_rate": round(obs_reclass_rate, 6),
                "bootstrap_mean_rate": round(float(np.mean(boot_reclass_rates)), 6),
                "bootstrap_95_ci": [round(reclass_ci[0], 6), round(reclass_ci[1], 6)],
                "wilson_95_ci": [round(reclass_wilson[0], 6), round(reclass_wilson[1], 6)],
            },
            "downgrades": {
                "observed_rate": round(obs_downgrade_rate, 6),
                "bootstrap_mean_rate": round(float(np.mean(boot_downgrade_rates)), 6),
                "bootstrap_95_ci": [round(downgrade_ci[0], 6), round(downgrade_ci[1], 6)],
                "wilson_95_ci": [round(downgrade_wilson[0], 6), round(downgrade_wilson[1], 6)],
            },
            "escalation_reductions": {
                "observed_rate": round(obs_escalation_red_rate, 6),
                "bootstrap_mean_rate": round(float(np.mean(boot_escalation_red_rates)), 6),
                "bootstrap_95_ci": [round(escalation_red_ci[0], 6), round(escalation_red_ci[1], 6)],
                "wilson_95_ci": [round(escalation_red_wilson[0], 6), round(escalation_red_wilson[1], 6)],
            },
        }
