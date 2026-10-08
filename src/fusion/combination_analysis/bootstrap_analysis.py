"""
src/fusion/combination_analysis/bootstrap_analysis.py
Phase C11.9: Reproducible Bootstrap Confidence Interval Estimation

Computes 95% empirical bootstrap confidence intervals (B=1,000 resamples, seed=115)
for all combination and tier-level metrics.
"""

from typing import Sequence, Tuple, Dict, Any, List
import numpy as np


def compute_bootstrap_ci_1d(
    values: Sequence[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 115,
) -> Tuple[float, float]:
    """
    Computes (lower, upper) 1 - alpha bootstrap confidence interval for the sample mean.
    Returns (0.0, 0.0) if sample has fewer than 2 elements.
    """
    if len(values) < 2:
        val = float(values[0]) if len(values) == 1 else 0.0
        return val, val

    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    rng = np.random.RandomState(seed)

    boot_means = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[b] = np.mean(sample)

    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def compute_bootstrap_ci_std(
    values: Sequence[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 115,
) -> Tuple[float, float]:
    """
    Computes (lower, upper) bootstrap confidence interval for the sample standard deviation.
    """
    if len(values) < 2:
        return 0.0, 0.0

    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    rng = np.random.RandomState(seed)

    boot_stds = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stds[b] = np.std(sample, ddof=1)

    low = float(np.percentile(boot_stds, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_stds, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def compute_paired_bootstrap_ci_diff(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 115,
) -> Tuple[float, float, float]:
    """
    Computes (mean_diff, lower_ci, upper_ci) for paired difference (A - B) across resamples.
    """
    arr_a = np.asarray(values_a, dtype=np.float64)
    arr_b = np.asarray(values_b, dtype=np.float64)
    if len(arr_a) != len(arr_b) or len(arr_a) < 2:
        diff = float(np.mean(arr_a) - np.mean(arr_b)) if len(arr_a) == len(arr_b) and len(arr_a) > 0 else 0.0
        return diff, diff, diff

    diffs = arr_a - arr_b
    mean_diff = float(np.mean(diffs))
    n = len(diffs)
    rng = np.random.RandomState(seed)

    boot_mean_diffs = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_mean_diffs[b] = np.mean(sample)

    low = float(np.percentile(boot_mean_diffs, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_mean_diffs, 100.0 * (1.0 - alpha / 2.0)))
    return mean_diff, low, high
