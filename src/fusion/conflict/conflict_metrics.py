"""
src/fusion/conflict/conflict_metrics.py
Phase C11.7: Cross-Modality Conflict & Discordance Analysis

Implements continuous conflict metrics:
- Maximum Pairwise Disagreement: Delta_max = max_{j < k} |r_j - r_k|
- Mean Pairwise Disagreement: Delta_mean = (1/P) sum_{j < k} |r_j - r_k|
- Weighted Consensus Variance: V_w = sum w_i (r_i - R_fusion)^2
- Weighted Consensus Standard Deviation: sigma_w = sqrt(V_w)
- Routing Weight Entropy: H = -sum w_i ln(w_i)
"""

from typing import Dict, Optional, Tuple
import math
from .conflict_result import PairwiseRecord


def compute_max_disagreement(
    pairwise_records: Tuple[PairwiseRecord, ...]
) -> Tuple[Optional[float], Optional[Tuple[str, str]]]:
    """
    Computes maximum absolute pairwise risk disagreement Delta_max
    and identifies the dominant conflicting modality pair.
    """
    if not pairwise_records:
        return None, None

    max_val = -1.0
    dominant_pair: Optional[Tuple[str, str]] = None

    for rec in pairwise_records:
        if rec.absolute_difference > max_val:
            max_val = rec.absolute_difference
            dominant_pair = (rec.modality_a, rec.modality_b)

    return float(max_val), dominant_pair


def compute_mean_disagreement(
    pairwise_records: Tuple[PairwiseRecord, ...]
) -> Optional[float]:
    """
    Computes mean absolute pairwise risk disagreement Delta_mean.
    Delta_mean = (1 / P) * sum_{j < k} |r_j - r_k|
    """
    if not pairwise_records:
        return None

    total_diff = sum(rec.absolute_difference for rec in pairwise_records)
    mean_val = total_diff / len(pairwise_records)
    return float(mean_val)


def compute_weighted_dispersion(
    active_modalities: Tuple[str, ...],
    risks: Dict[str, float],
    weights: Dict[str, float],
    r_fusion: float,
) -> Tuple[float, float]:
    """
    Computes weighted variance and standard deviation around the ACARA-U consensus:
    V_w = sum_{i in A} w_i * (r_i - R_fusion)^2
    sigma_w = sqrt(V_w)
    """
    if not active_modalities:
        return 0.0, 0.0

    if len(active_modalities) == 1:
        return 0.0, 0.0

    var_w = 0.0
    for mod in active_modalities:
        w_i = weights[mod]
        r_i = risks[mod]
        diff = r_i - r_fusion
        var_w += w_i * (diff ** 2)

    # Ensure numerical stability within theoretical bound [0.0, 0.25]
    var_w = max(0.0, min(0.25, float(var_w)))
    std_w = math.sqrt(var_w)

    return var_w, std_w


def compute_weight_entropy(
    active_modalities: Tuple[str, ...],
    weights: Dict[str, float],
) -> float:
    """
    Computes Shannon entropy of router authority weights:
    H = -sum_{i in A} w_i * ln(w_i)
    """
    if not active_modalities:
        return 0.0

    entropy = 0.0
    for mod in active_modalities:
        w = weights.get(mod, 0.0)
        if w > 1e-12:
            entropy -= w * math.log(w)

    return max(0.0, float(entropy))
