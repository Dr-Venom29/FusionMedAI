"""
src/fusion/combination_analysis/combination_metrics.py
Phase C11.9: Decision-Level Combination Metrics & Statistical Summaries

Computes comprehensive decision-level metrics for each modality combination:
1. Routing statistics: mean weights (w_R, w_F, w_C), entropy H(w), max weight w_max, dominant rate.
2. Risk statistics: R_fusion (mean, std, median, IQR, P90, P95).
3. Decision index: DCRI_delta (mean, std, median, IQR, P90, P95).
4. Uncertainty & Conflict: U_sum, Delta_max, Delta_mean, sigma_w.
5. Tri-modal sensitivity: Delta R relative to RFC, Delta DCRI relative to RFC.

Enforces minimum sample size threshold (N >= 5).
"""

from typing import Dict, Any, List, Sequence, Optional
import math
import numpy as np

MIN_SAMPLE_SIZE = 5


def calc_distribution_stats(values: Sequence[float]) -> Dict[str, float]:
    """
    Computes standard summary statistics for a 1D sequence of float values.
    Returns zeroed stats if values sequence is empty.
    """
    if len(values) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "iqr": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    arr = np.asarray(values, dtype=np.float64)
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "iqr": float(p75 - p25),
        "p25": p25,
        "p75": p75,
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_routing_entropy(weights: Dict[str, float]) -> float:
    """Computes Shannon entropy H(w) = -sum(w_i * ln(w_i)) over positive weights."""
    h = 0.0
    for w in weights.values():
        if w > 1e-12:
            h -= w * math.log(w)
    return max(0.0, h)


def compute_conflict_metrics(
    risks: Dict[str, float],
    weights: Dict[str, float],
    active_modalities: Sequence[str],
) -> Dict[str, float]:
    """
    Computes pairwise max disagreement Delta_max, mean disagreement Delta_mean,
    and weighted consensus std sigma_w over active modalities.
    """
    active = [m for m in active_modalities if m in risks and m in weights and weights[m] > 0.0]
    p = len(active)

    if p <= 1:
        return {
            "delta_max": 0.0,
            "delta_mean": 0.0,
            "sigma_w": 0.0,
        }

    # Pairwise differences
    diffs: List[float] = []
    for i in range(p):
        for j in range(i + 1, p):
            diffs.append(abs(risks[active[i]] - risks[active[j]]))

    delta_max = float(max(diffs)) if diffs else 0.0
    delta_mean = float(np.mean(diffs)) if diffs else 0.0

    # Weighted consensus variance
    r_fusion = sum(weights[m] * risks[m] for m in active)
    sigma_w_sq = sum(weights[m] * ((risks[m] - r_fusion) ** 2) for m in active)
    sigma_w = float(math.sqrt(max(0.0, sigma_w_sq)))

    return {
        "delta_max": delta_max,
        "delta_mean": delta_mean,
        "sigma_w": sigma_w,
    }
