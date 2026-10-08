"""
FusionMedAI - Phase C11.12: Paired Bootstrap Statistical Resampling
Computes deterministic paired difference bootstrap confidence intervals (B=1,000, seed=115)
comparing each perturbed configuration against the frozen reference Theta_0.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """
    Immutable container for paired bootstrap difference results.
    """
    config_id: str
    metric_name: str
    mean_difference: float
    median_difference: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    excludes_zero: bool
    n_resamples: int
    sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "metric_name": self.metric_name,
            "mean_difference": round(float(self.mean_difference), 6),
            "median_difference": round(float(self.median_difference), 6),
            "ci_lower": round(float(self.ci_lower), 6),
            "ci_upper": round(float(self.ci_upper), 6),
            "confidence_level": float(self.confidence_level),
            "excludes_zero": bool(self.excludes_zero),
            "n_resamples": int(self.n_resamples),
            "sample_size": int(self.sample_size),
        }


def compute_paired_bootstrap(
    metric_perturbed: List[float],
    metric_reference: List[float],
    config_id: str,
    metric_name: str,
    n_bootstraps: int = 1000,
    seed: int = 115,
    ci_level: float = 0.95,
    packet_ids_perturbed: Optional[List[str]] = None,
    packet_ids_reference: Optional[List[str]] = None,
) -> BootstrapResult:
    """
    Performs deterministic paired bootstrap on the difference D = M_perturbed - M_reference.
    
    Args:
        metric_perturbed: Array of metric values for perturbed configuration across N packets.
        metric_reference: Array of metric values for reference configuration across the same N packets.
        config_id: Identifier of perturbed configuration.
        metric_name: Name of evaluated metric.
        n_bootstraps: Number of bootstrap resamples (default 1000).
        seed: Random seed for deterministic reproducibility (default 115).
        ci_level: Confidence level (default 0.95).
        packet_ids_perturbed: Optional list of packet IDs for perturbed evaluation (to verify exact pairing).
        packet_ids_reference: Optional list of packet IDs for reference evaluation (to verify exact pairing).
        
    Returns:
        BootstrapResult with mean, median, CI bounds, and zero-exclusion status.
    """
    if packet_ids_perturbed is not None and packet_ids_reference is not None:
        if packet_ids_perturbed != packet_ids_reference:
            raise ValueError(f"Packet ID mismatch between perturbed and reference cohorts in {config_id}")

    arr_p = np.asarray(metric_perturbed, dtype=np.float64)
    arr_r = np.asarray(metric_reference, dtype=np.float64)

    if len(arr_p) != len(arr_r):
        raise ValueError(f"Length mismatch: perturbed={len(arr_p)}, reference={len(arr_r)}")

    n = len(arr_p)
    differences = arr_p - arr_r

    mean_diff = float(np.mean(differences))
    median_diff = float(np.median(differences))

    # Deterministic resampling
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = np.mean(differences[idx])

    alpha = 1.0 - ci_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))

    excludes_zero = (ci_lower > 0.0 and ci_upper > 0.0) or (ci_lower < 0.0 and ci_upper < 0.0)

    return BootstrapResult(
        config_id=config_id,
        metric_name=metric_name,
        mean_difference=mean_diff,
        median_difference=median_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=ci_level,
        excludes_zero=excludes_zero,
        n_resamples=n_bootstraps,
        sample_size=n,
    )
