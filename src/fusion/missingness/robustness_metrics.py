"""
src/fusion/missingness/robustness_metrics.py
Phase C11.8: Missing Modality Robustness & Sensitivity Metrics

Defines quantitative measures for authority redistribution, risk continuity,
DCRI sensitivity, uncertainty burden shift, entropy, and bootstrap confidence intervals.
"""

from typing import Dict, Any, List, Optional, Tuple, Sequence
import math
import numpy as np


def calc_stats(values: Sequence[float]) -> Dict[str, float]:
    """Computes standard distributional summary statistics."""
    if not values:
        return {
            "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "p25": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "iqr": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    q75, q25 = np.percentile(arr, [75, 25])
    return {
        "mean": round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "p25": round(float(q25), 6),
        "p75": round(float(q75), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "iqr": round(float(q75 - q25), 6),
    }


def compute_risk_delta(r_full: float, r_subset: float) -> float:
    """Computes absolute risk shift: Delta R = |R_fusion^full - R_fusion^subset|."""
    return abs(float(r_full) - float(r_subset))


def compute_dcri_delta(dcri_full: float, dcri_subset: float) -> float:
    """Computes absolute DCRI shift: Delta DCRI = |DCRI_full - DCRI_subset|."""
    return abs(float(dcri_full) - float(dcri_subset))


def compute_authority_redistribution(
    weights_full: Dict[str, float],
    weights_subset: Dict[str, float],
) -> Dict[str, float]:
    """
    Computes authority redistribution for each modality:
    Delta w_j = w_j^subset - w_j^full.
    """
    all_mods = set(weights_full.keys()) | set(weights_subset.keys())
    return {
        m: round(float(weights_subset.get(m, 0.0)) - float(weights_full.get(m, 0.0)), 6)
        for m in sorted(all_mods)
    }


def compute_dcri_decomposition(
    r_full: float,
    r_subset: float,
    u_sum_full: float,
    u_sum_subset: float,
    delta: float = 0.20,
) -> Dict[str, float]:
    """
    Decomposes signed DCRI shift into risk authority shift and uncertainty discount shift:
    Delta DCRI_signed = DCRI_full - DCRI_subset
                     = (R_full - delta * U_full) - (R_subset - delta * U_subset)
                     = (R_full - R_subset) - delta * (U_full - U_subset)
                     = Delta R_signed - Delta P_U
    """
    delta_r_signed = float(r_full) - float(r_subset)
    delta_u_sum = float(u_sum_full) - float(u_sum_subset)
    delta_penalty = float(delta) * delta_u_sum
    delta_dcri_signed = delta_r_signed - delta_penalty

    return {
        "delta_r_signed": round(delta_r_signed, 6),
        "delta_u_sum": round(delta_u_sum, 6),
        "delta_penalty": round(delta_penalty, 6),
        "delta_dcri_signed": round(delta_dcri_signed, 6),
        "delta_r_abs": round(abs(delta_r_signed), 6),
        "delta_dcri_abs": round(abs(delta_dcri_signed), 6),
    }


def compute_routing_entropy(weights: Dict[str, float]) -> float:
    """
    Computes Shannon entropy of router authority weights over active channels:
    H(w) = - sum_{i in A, w_i > 0} w_i ln(w_i).
    """
    h = 0.0
    for w in weights.values():
        if w > 1e-12:
            h -= w * math.log(w)
    return round(float(h), 6)


def compute_max_authority(weights: Dict[str, float]) -> float:
    """Computes peak authority weight: w_max = max_{i} w_i."""
    if not weights:
        return 0.0
    return round(float(max(weights.values())), 6)


def compute_bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 115,
) -> Dict[str, float]:
    """
    Computes percentile bootstrap confidence interval for the sample mean.
    """
    if not values:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(arr)

    # Vectorized bootstrap resampling
    indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = np.mean(arr[indices], axis=1)

    alpha = 1.0 - confidence_level
    ci_lower = np.percentile(resampled_means, 100.0 * (alpha / 2.0))
    ci_upper = np.percentile(resampled_means, 100.0 * (1.0 - alpha / 2.0))

    return {
        "mean": round(float(np.mean(arr)), 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "confidence_level": confidence_level,
        "n_resamples": n_resamples,
        "seed": seed,
    }
