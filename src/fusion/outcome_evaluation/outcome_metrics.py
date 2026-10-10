"""
FusionMedAI - Outcome Evaluation Metrics & Statistical Analysis
Computes predictive error metrics (MAE, RMSE), decision loss against an explicit action cost matrix,
paired effect sizes, and true 2-stage hierarchical cluster bootstrap confidence intervals.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import math
import scipy.stats as stats


# =============================================================================
# Action Cost Matrix for Decision Policy Analysis
# =============================================================================
VALID_ACTION_TIERS = ("TIER_0_ROUTINE", "TIER_1_ASSESSMENT", "TIER_2_ESCALATION")

ACTION_COST_MATRIX: Dict[str, Dict[str, float]] = {
    "TIER_0_ROUTINE": {
        "TIER_0_ROUTINE": 0.0,
        "TIER_1_ASSESSMENT": 1.0,   # Moderate risk under-assessed
        "TIER_2_ESCALATION": 5.0,   # Critical high risk missed completely
    },
    "TIER_1_ASSESSMENT": {
        "TIER_0_ROUTINE": 0.5,      # Low risk gets unnecessary assessment
        "TIER_1_ASSESSMENT": 0.0,
        "TIER_2_ESCALATION": 2.5,   # High risk delayed to intermediate tier
    },
    "TIER_2_ESCALATION": {
        "TIER_0_ROUTINE": 2.0,      # Low risk unnecessarily escalated to specialist
        "TIER_1_ASSESSMENT": 1.0,   # Moderate risk over-escalated
        "TIER_2_ESCALATION": 0.0,
    },
}


def assign_action_tier(score: float, tau1: float = 0.20, tau2: float = 0.40) -> str:
    """
    Assigns 3-tier action given decision score and thresholds.
    
    Partitions continuous score into:
      - score < tau1: TIER_0_ROUTINE
      - tau1 <= score < tau2: TIER_1_ASSESSMENT
      - score >= tau2: TIER_2_ESCALATION
    """
    if not isinstance(score, (int, float)) or math.isnan(score) or math.isinf(score):
        raise ValueError(f"Score must be a finite float, got {score}")
    if tau1 >= tau2:
        raise ValueError(f"Threshold tau1 ({tau1}) must be strictly less than tau2 ({tau2})")

    if score < tau1:
        return "TIER_0_ROUTINE"
    elif score < tau2:
        return "TIER_1_ASSESSMENT"
    else:
        return "TIER_2_ESCALATION"


def compute_decision_loss(assigned_tier: str, true_oracle_tier: str) -> float:
    """Computes cost of assigning a tier relative to ground-truth oracle tier."""
    if assigned_tier not in VALID_ACTION_TIERS:
        raise ValueError(f"Invalid assigned_tier '{assigned_tier}'. Expected one of {VALID_ACTION_TIERS}")
    if true_oracle_tier not in VALID_ACTION_TIERS:
        raise ValueError(f"Invalid true_oracle_tier '{true_oracle_tier}'. Expected one of {VALID_ACTION_TIERS}")

    return float(ACTION_COST_MATRIX[assigned_tier][true_oracle_tier])


def compute_prediction_metrics(
    predictions: List[float], targets: List[float]
) -> Dict[str, float]:
    """Computes MAE, RMSE, and mean error against target with defensive input checks."""
    if len(predictions) == 0 or len(targets) == 0:
        raise ValueError("Cannot compute prediction metrics on empty list.")
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions length ({len(predictions)}) != targets length ({len(targets)})")

    y_pred = np.array(predictions, dtype=float)
    y_true = np.array(targets, dtype=float)

    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("Predictions contain NaN or Inf values.")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("Targets contain NaN or Inf values.")

    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mean_error = float(np.mean(diff))

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mean_error": round(mean_error, 6),
    }


def compute_hierarchical_bootstrap_ci(
    cohort_diffs: List[List[float]],
    n_bootstrap: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes a true 2-stage hierarchical cluster bootstrap confidence interval
    and an exact two-tailed Student's t-test with calculated p-value across cohorts.
    
    Procedure:
    1. Resample cohorts with replacement (outer stage).
    2. Within each resampled cohort, resample packet paired differences with replacement (inner stage).
    3. Compute the grand mean difference for each bootstrap replicate.
    4. Derive empirical percentile confidence interval [alpha/2, 1 - alpha/2].
    5. Compute cohort-level Student's t-statistic and exact two-tailed p-value.
    
    Args:
        cohort_diffs: List of cohorts, where each cohort is a list of per-packet differences (e.g. error_B6 - error_B5).
        n_bootstrap: Number of bootstrap iterations (must be >= 10).
        seed: Random seed for reproducibility (must be >= 0).
        alpha: Significance level (0.0 < alpha < 1.0, default 0.05 for 95% CI).
        
    Returns:
        Dictionary containing grand mean, CI bounds, exact t-statistic and p-value, and statistical flags.
    """
    if not isinstance(cohort_diffs, list) or len(cohort_diffs) == 0:
        raise ValueError("cohort_diffs must be a non-empty list of cohorts.")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 10:
        raise ValueError(f"n_bootstrap must be an integer >= 10, got {n_bootstrap}")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    if not isinstance(alpha, (int, float)) or not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0.0, 1.0), got {alpha}")

    for idx, c in enumerate(cohort_diffs):
        if not isinstance(c, list) or len(c) == 0:
            raise ValueError(f"Cohort {idx} contains zero packet differences.")
        arr = np.array(c, dtype=float)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError(f"Cohort {idx} contains non-finite (NaN/Inf) packet differences.")

    k_cohorts = len(cohort_diffs)
    cohort_arrays = [np.array(c, dtype=float) for c in cohort_diffs]
    
    # Compute flat mean across all packets
    all_flat = np.concatenate(cohort_arrays)
    grand_mean = float(np.mean(all_flat))

    # Cohort-level means and sample standard deviation
    cohort_means = [float(np.mean(arr)) for arr in cohort_arrays]
    mean_of_cohort_means = float(np.mean(cohort_means))
    std_of_cohort_means = float(np.std(cohort_means, ddof=1)) if k_cohorts > 1 else 0.0

    rng = np.random.RandomState(seed)
    boot_grand_means = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        # Stage 1: Resample cohort indices with replacement
        sampled_cohort_idx = rng.choice(k_cohorts, size=k_cohorts, replace=True)
        
        # Stage 2: Resample packets within each chosen cohort
        resampled_packet_means = []
        for c_idx in sampled_cohort_idx:
            c_data = cohort_arrays[c_idx]
            n_packets = len(c_data)
            sampled_packets = rng.choice(c_data, size=n_packets, replace=True)
            resampled_packet_means.append(np.mean(sampled_packets))
            
        boot_grand_means[b] = np.mean(resampled_packet_means)

    ci_lower = float(np.percentile(boot_grand_means, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_grand_means, 100 * (1.0 - alpha / 2.0)))

    # Monte Carlo resolution bound: 1 / n_bootstrap
    mc_resolution = float(1.0 / n_bootstrap)

    # Empirical proportion of replicates crossing zero
    if grand_mean < 0:
        prop_cross_zero = float(np.mean(boot_grand_means >= 0.0))
    else:
        prop_cross_zero = float(np.mean(boot_grand_means <= 0.0))

    # Exact Cohort-level Student's t-test with calculated p-value (df = k_cohorts - 1)
    if k_cohorts > 1 and std_of_cohort_means > 1e-12:
        df = k_cohorts - 1
        se = std_of_cohort_means / math.sqrt(k_cohorts)
        t_stat = mean_of_cohort_means / se
        # Exact two-tailed p-value from Student's t distribution survival function
        p_val = float(stats.t.sf(abs(t_stat), df=df) * 2.0)
        t_test_significant = bool(p_val < alpha)
    else:
        t_stat = 0.0
        p_val = 1.0
        t_test_significant = False

    return {
        "grand_mean_diff": round(grand_mean, 6),
        "cohort_mean_of_means": round(mean_of_cohort_means, 6),
        "cohort_std_of_means": round(std_of_cohort_means, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "n_bootstrap": n_bootstrap,
        "mc_resolution": round(mc_resolution, 6),
        "prop_bootstrap_crossing_zero": round(prop_cross_zero, 6),
        "cohort_t_stat": round(t_stat, 6),
        "cohort_t_pvalue": p_val,
        "cohort_t_test_significant": t_test_significant,
        "is_significant_95": bool(ci_upper < 0.0 or ci_lower > 0.0),
    }
