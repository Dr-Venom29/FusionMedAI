"""
src/fusion/degradation/response_metrics.py
Phase C11.10: Metrics & Statistical Analysis Engine for Input Degradation

Calculates:
- Delta Q_i, Delta w_i, Relative Authority Reduction (RAR_i)
- Quality-Authority Response Slope: S_QW = Delta w_i / Delta Q_i
- Quality-Routing Spearman Alignment: rho(Q_i, w_i)
- Monotonicity Rates across severity ladder D0 -> D1 -> D2 -> D3
- Risk shifts Delta R, DCRI shifts Delta DCRI
- Routing Entropy H(w) and Conflict Metrics (Delta_max, sigma_w)
- Hard-mask safety invariance check (A_i = 0 => Delta w_active = 0, Delta R = 0)
- 1,000-resample paired non-parametric bootstrap 95% confidence intervals
"""

from typing import Dict, Any, List, Tuple, Sequence, Optional
import numpy as np
from scipy import stats
import math

from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord
from src.fusion.degradation.degradation_spec import (
    SEVERITY_D0_CLEAN,
    SEVERITY_D1_MILD,
    SEVERITY_D2_MODERATE,
    SEVERITY_D3_SEVERE,
    SEVERITY_LEVELS,
)
from src.fusion.degradation.degradation_result import (
    QualityResponseRecord,
    RoutingResponseRecord,
    SlopeRecord,
    MonotonicityRecord,
)


def compute_bootstrap_ci_1d(
    values: Sequence[float],
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 115,
) -> Tuple[float, float]:
    """Computes non-parametric 1D bootstrap confidence interval."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])

    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(arr), size=(n_resamples, len(arr)))
    resampled_means = np.mean(arr[indices], axis=1)

    alpha = 1.0 - ci_level
    low = float(np.percentile(resampled_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(resampled_means, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def compute_paired_bootstrap_ci_diff(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 115,
) -> Tuple[float, float, float]:
    """Computes paired bootstrap confidence interval for difference (A - B)."""
    diffs = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    if len(diffs) == 0:
        return 0.0, 0.0, 0.0
    mean_diff = float(np.mean(diffs))
    low, high = compute_bootstrap_ci_1d(diffs, n_resamples=n_resamples, ci_level=ci_level, seed=seed)
    return mean_diff, low, high


def compute_quality_authority_slope(
    delta_w_list: Sequence[float],
    delta_q_list: Sequence[float],
    eps: float = 1e-6,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Computes Quality-Authority Response Slope S_QW = Delta w_i / Delta Q_i.
    Since Delta Q < 0 and Delta w < 0 under degradation, S_QW > 0.
    """
    slopes: List[float] = []
    for dw, dq in zip(delta_w_list, delta_q_list):
        if abs(dq) > eps:
            slopes.append(dw / dq)
        else:
            slopes.append(0.0)

    arr = np.asarray(slopes, dtype=float)
    mean_s = float(np.mean(arr)) if len(arr) > 0 else 0.0
    median_s = float(np.median(arr)) if len(arr) > 0 else 0.0
    ci_low, ci_high = compute_bootstrap_ci_1d(arr) if len(arr) > 0 else (0.0, 0.0)
    return mean_s, median_s, (ci_low, ci_high)


def compute_spearman_alignment(
    quality_values: Sequence[float],
    weight_values: Sequence[float],
) -> Tuple[float, float]:
    """Computes Spearman rank correlation rho(Q_i, w_i) and p-value."""
    q_arr = np.asarray(quality_values, dtype=float)
    w_arr = np.asarray(weight_values, dtype=float)
    if len(q_arr) < 3 or np.std(q_arr) < 1e-7 or np.std(w_arr) < 1e-7:
        return 0.0, 1.0
    res = stats.spearmanr(q_arr, w_arr)
    return float(res.statistic), float(res.pvalue)


def compute_monotonicity_rates(
    ladder_qualities: Sequence[Sequence[float]],  # Outer: packets, Inner: [D0, D1, D2, D3]
    ladder_weights: Sequence[Sequence[float]],    # Outer: packets, Inner: [D0, D1, D2, D3]
    eps: float = 1e-5,
) -> Tuple[int, int, float, int, float]:
    """
    Computes fraction of packets satisfying monotonic degradation ordering:
    Quality: Q(D0) >= Q(D1) >= Q(D2) >= Q(D3)
    Routing: w(D0) >= w(D1) >= w(D2) >= w(D3)
    """
    n_packets = len(ladder_qualities)
    if n_packets == 0:
        return 0, 0, 0.0, 0, 0.0

    q_mono_count = 0
    w_mono_count = 0

    for q_series, w_series in zip(ladder_qualities, ladder_weights):
        # Check quality monotonicity
        is_q_mono = all(
            (q_series[i] + eps) >= q_series[i + 1]
            for i in range(len(q_series) - 1)
        )
        if is_q_mono:
            q_mono_count += 1

        # Check routing monotonicity
        is_w_mono = all(
            (w_series[i] + eps) >= w_series[i + 1]
            for i in range(len(w_series) - 1)
        )
        if is_w_mono:
            w_mono_count += 1

    return (
        n_packets,
        q_mono_count,
        float(q_mono_count / n_packets),
        w_mono_count,
        float(w_mono_count / n_packets),
    )


def compute_routing_entropy(weights: Dict[str, float]) -> float:
    """Computes Shannon entropy H(w) = -sum w_i ln(w_i) over non-zero weights."""
    h = 0.0
    for w in weights.values():
        if w > 1e-12:
            h -= float(w) * math.log(float(w))
    return float(h)


def compute_conflict_metrics(
    weights: Dict[str, float],
    risks: Dict[str, float],
    active_modalities: Sequence[str],
) -> Tuple[float, float, float]:
    """
    Computes:
    - Delta_max: Maximum pairwise risk difference among active channels
    - Delta_mean: Mean pairwise risk difference
    - sigma_w: Weighted consensus standard deviation around ACARA-U consensus
    """
    if len(active_modalities) <= 1:
        return 0.0, 0.0, 0.0

    active_risks = [risks[m] for m in active_modalities]
    pairwise_diffs = []
    for i in range(len(active_risks)):
        for j in range(i + 1, len(active_risks)):
            pairwise_diffs.append(abs(active_risks[i] - active_risks[j]))

    delta_max = max(pairwise_diffs) if pairwise_diffs else 0.0
    delta_mean = float(np.mean(pairwise_diffs)) if pairwise_diffs else 0.0

    # Weighted consensus standard deviation
    w_vec = np.array([weights[m] for m in active_modalities], dtype=float)
    r_vec = np.array([risks[m] for m in active_modalities], dtype=float)
    w_sum = np.sum(w_vec)
    if w_sum > 0:
        w_norm = w_vec / w_sum
        r_fused = np.sum(w_norm * r_vec)
        v_w = np.sum(w_norm * (r_vec - r_fused) ** 2)
        sigma_w = float(np.sqrt(max(0.0, v_w)))
    else:
        sigma_w = 0.0

    return float(delta_max), float(delta_mean), float(sigma_w)


def check_hard_mask_invariance(
    clean_packet: ControlledDecisionPacket,
    degraded_packet: ControlledDecisionPacket,
    clean_weights: Dict[str, float],
    degraded_weights: Dict[str, float],
    clean_rfused: float,
    degraded_rfused: float,
    tol: float = 1e-6,
) -> bool:
    """
    Checks that if an unavailable modality (A_i = False) has its values corrupted/degraded,
    the active routing weights and fused risk remain EXACTLY invariant.
    """
    for m, rec in clean_packet.records.items():
        if not rec.availability:
            # Unavailable modality must receive exactly zero weight in both
            if clean_weights.get(m, 0.0) > tol or degraded_weights.get(m, 0.0) > tol:
                return False

    # Active weights must match
    for m, rec in clean_packet.records.items():
        if rec.availability:
            cw = clean_weights.get(m, 0.0)
            dw = degraded_weights.get(m, 0.0)
            if abs(cw - dw) > tol:
                return False

    # Fused risk must match
    if abs(clean_rfused - degraded_rfused) > tol:
        return False

    return True
