"""
src/fusion/combination_analysis/tail_robustness.py
Phase C11.9: Tail Robustness & Head-vs-Tail Comparative Evaluation

Computes:
1. Head vs Tail variability: sigma(R_fusion), sigma(DCRI), sigma(Entropy).
2. Head vs Tail sensitivity: mean Delta R relative to RFC, mean Delta DCRI relative to RFC.
3. Head vs Tail uncertainty: mean U_sum and dispersion.
4. Cross-distribution shift: Global expected value shifts across D1 (Balanced), D2 (Moderate), D3 (Strong Tail).
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Sequence, Optional
import numpy as np


@dataclass(frozen=True)
class TierSummaryRecord:
    """Statistical summary for a specific frequency tier (HEAD, MIDDLE, TAIL)."""
    tier: str
    num_packets: int
    participating_combinations: List[str]
    mean_r_fusion: float
    std_r_fusion: float
    mean_dcri: float
    std_dcri: float
    mean_entropy: float
    std_entropy: float
    mean_u_sum: float
    std_u_sum: float
    mean_delta_r: float          # Sensitivity relative to RFC
    std_delta_r: float
    mean_delta_dcri: float       # DCRI shift relative to RFC
    std_delta_dcri: float


@dataclass(frozen=True)
class TailRobustnessComparison:
    """Head vs Tail comparative contrast record."""
    distribution_id: str
    head_summary: TierSummaryRecord
    tail_summary: TierSummaryRecord
    delta_std_r_fusion: float    # std(Tail) - std(Head)
    delta_std_dcri: float        # std(Tail) - std(Head)
    delta_mean_sensitivity_r: float  # mean_delta_r(Tail) - mean_delta_r(Head)
    delta_mean_sensitivity_dcri: float
    delta_mean_u_sum: float      # mean_u_sum(Tail) - mean_u_sum(Head)


def compute_tier_summary(
    tier_name: str,
    combs: Sequence[str],
    comb_results: Dict[str, Dict[str, Any]],
) -> TierSummaryRecord:
    """Aggregates combination-level results into a tier-level summary."""
    r_list: List[float] = []
    dcri_list: List[float] = []
    h_list: List[float] = []
    u_list: List[float] = []
    dr_list: List[float] = []
    ddcri_list: List[float] = []
    n_total = 0

    for c in combs:
        if c in comb_results and comb_results[c].get("status") == "SUCCESS":
            res = comb_results[c]
            n_c = res["n_packets"]
            n_total += n_c
            r_list.extend(res["r_fusion_raw"])
            dcri_list.extend(res["dcri_raw"])
            h_list.extend(res["entropy_raw"])
            u_list.extend(res["u_sum_raw"])
            dr_list.extend(res["delta_r_raw"])
            ddcri_list.extend(res["delta_dcri_raw"])

    if n_total == 0:
        return TierSummaryRecord(
            tier=tier_name,
            num_packets=0,
            participating_combinations=list(combs),
            mean_r_fusion=0.0,
            std_r_fusion=0.0,
            mean_dcri=0.0,
            std_dcri=0.0,
            mean_entropy=0.0,
            std_entropy=0.0,
            mean_u_sum=0.0,
            std_u_sum=0.0,
            mean_delta_r=0.0,
            std_delta_r=0.0,
            mean_delta_dcri=0.0,
            std_delta_dcri=0.0,
        )

    r_arr = np.asarray(r_list, dtype=np.float64)
    dcri_arr = np.asarray(dcri_list, dtype=np.float64)
    h_arr = np.asarray(h_list, dtype=np.float64)
    u_arr = np.asarray(u_list, dtype=np.float64)
    dr_arr = np.asarray(dr_list, dtype=np.float64)
    ddcri_arr = np.asarray(ddcri_list, dtype=np.float64)

    return TierSummaryRecord(
        tier=tier_name,
        num_packets=n_total,
        participating_combinations=list(combs),
        mean_r_fusion=float(np.mean(r_arr)),
        std_r_fusion=float(np.std(r_arr, ddof=1)) if len(r_arr) > 1 else 0.0,
        mean_dcri=float(np.mean(dcri_arr)),
        std_dcri=float(np.std(dcri_arr, ddof=1)) if len(dcri_arr) > 1 else 0.0,
        mean_entropy=float(np.mean(h_arr)),
        std_entropy=float(np.std(h_arr, ddof=1)) if len(h_arr) > 1 else 0.0,
        mean_u_sum=float(np.mean(u_arr)),
        std_u_sum=float(np.std(u_arr, ddof=1)) if len(u_arr) > 1 else 0.0,
        mean_delta_r=float(np.mean(dr_arr)),
        std_delta_r=float(np.std(dr_arr, ddof=1)) if len(dr_arr) > 1 else 0.0,
        mean_delta_dcri=float(np.mean(ddcri_arr)),
        std_delta_dcri=float(np.std(ddcri_arr, ddof=1)) if len(ddcri_arr) > 1 else 0.0,
    )


def compute_tail_robustness_comparison(
    dist_id: str,
    head_summary: TierSummaryRecord,
    tail_summary: TierSummaryRecord,
) -> TailRobustnessComparison:
    """Computes Head-vs-Tail differential contrast."""
    return TailRobustnessComparison(
        distribution_id=dist_id,
        head_summary=head_summary,
        tail_summary=tail_summary,
        delta_std_r_fusion=float(tail_summary.std_r_fusion - head_summary.std_r_fusion),
        delta_std_dcri=float(tail_summary.std_dcri - head_summary.std_dcri),
        delta_mean_sensitivity_r=float(tail_summary.mean_delta_r - head_summary.mean_delta_r),
        delta_mean_sensitivity_dcri=float(tail_summary.mean_delta_dcri - head_summary.mean_delta_dcri),
        delta_mean_u_sum=float(tail_summary.mean_u_sum - head_summary.mean_u_sum),
    )
