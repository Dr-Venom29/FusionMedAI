"""
src/fusion/combination_analysis/baseline_evaluator.py
Phase C11.9: Comparative Baseline Tail-Robustness Evaluator

Evaluates Baselines B1–B6 across all controlled combinations and distributions:
B1: Reliability-Selected
B2: Uniform Average
B3: Confidence-Weighted
B4: Confidence + Reliability
B5: Confidence + Reliability - Uncertainty
B6: Full ACARA-U

Computes comparative tail sensitivity D_tail and tail variability sigma(R_tail) under identical packet assignments.
"""

from typing import Dict, Any, List, Sequence, Tuple
import numpy as np

from src.fusion.baselines.decision_packet import ControlledDecisionPacket
from src.fusion.baselines.fusion_runner import FusionRunner
from src.fusion.combination_analysis.combination_definition import (
    NON_EMPTY_COMBINATIONS,
    COMBINATION_RFC,
    apply_combination_to_packet,
)
from src.fusion.combination_analysis.distribution_generator import AssignedPacket
from src.fusion.combination_analysis.bootstrap_analysis import compute_paired_bootstrap_ci_diff


BASELINE_IDS = ("B1", "B2", "B3", "B4", "B5", "B6")
BASELINE_NAMES = {
    "B1": "Reliability-Selected",
    "B2": "Uniform Average",
    "B3": "Confidence-Weighted",
    "B4": "Confidence + Reliability",
    "B5": "Conf + Rel - Uncertainty",
    "B6": "ACARA-U (Full Router)",
}


def evaluate_baselines_on_combinations(
    assigned_cohort: Sequence[AssignedPacket],
    full_cohort: Sequence[ControlledDecisionPacket],
    head_combinations: Sequence[str],
    tail_combinations: Sequence[str],
) -> Dict[str, Any]:
    """
    Evaluates all Baselines B1–B6 across the assigned cohort and compares their
    tail sensitivity D_tail and tail dispersion relative to RFC.
    """
    runner = FusionRunner()
    
    # Pre-evaluate RFC baseline risks on the full cohort for reference: packet_id -> {baseline_id -> r_fusion_rfc}
    rfc_refs: Dict[str, Dict[str, float]] = {}
    for pkt in full_cohort:
        rfc_pkt = apply_combination_to_packet(pkt, COMBINATION_RFC)
        rfc_evals = runner.evaluate_all_baselines(rfc_pkt)
        rfc_refs[pkt.packet_id] = {b_id: res.r_fusion for b_id, res in rfc_evals.items()}

    # Evaluate all assigned packets
    baseline_records: Dict[str, Dict[str, Any]] = {
        b_id: {
            "name": BASELINE_NAMES[b_id],
            "by_combination": {c: {"r_fusion": [], "delta_r": [], "weights": {m: [] for m in ("retina", "foot", "clinical")}} for c in NON_EMPTY_COMBINATIONS},
            "head_r_fusion": [],
            "tail_r_fusion": [],
            "head_delta_r": [],
            "tail_delta_r": [],
        }
        for b_id in BASELINE_IDS
    }

    for ap in assigned_cohort:
        comb_id = ap.assigned_combination
        pkt = ap.packet
        orig_id = ap.original_packet_id
        evals = runner.evaluate_all_baselines(pkt)

        for b_id, res in evals.items():
            r_f = res.r_fusion
            rfc_r = rfc_refs[orig_id][b_id]
            delta_r = abs(rfc_r - r_f)

            b_rec = baseline_records[b_id]
            b_rec["by_combination"][comb_id]["r_fusion"].append(r_f)
            b_rec["by_combination"][comb_id]["delta_r"].append(delta_r)
            for m in ("retina", "foot", "clinical"):
                b_rec["by_combination"][comb_id]["weights"][m].append(res.weights.get(m, 0.0))

            if comb_id in head_combinations:
                b_rec["head_r_fusion"].append(r_f)
                b_rec["head_delta_r"].append(delta_r)
            elif comb_id in tail_combinations:
                b_rec["tail_r_fusion"].append(r_f)
                b_rec["tail_delta_r"].append(delta_r)

    # Compute comparative summary table
    summary_table: Dict[str, Any] = {}
    b6_tail_dr = np.asarray(baseline_records["B6"]["tail_delta_r"], dtype=np.float64)

    for b_id in BASELINE_IDS:
        b_rec = baseline_records[b_id]
        head_r = np.asarray(b_rec["head_r_fusion"], dtype=np.float64)
        tail_r = np.asarray(b_rec["tail_r_fusion"], dtype=np.float64)
        head_dr = np.asarray(b_rec["head_delta_r"], dtype=np.float64)
        tail_dr = np.asarray(b_rec["tail_delta_r"], dtype=np.float64)

        paired_diff_info = None
        if len(tail_dr) > 1 and len(b6_tail_dr) == len(tail_dr) and b_id != "B6":
            mean_d, ci_l, ci_h = compute_paired_bootstrap_ci_diff(tail_dr, b6_tail_dr, n_resamples=1000, seed=115)
            paired_diff_info = {
                "mean_difference": mean_d,
                "ci_95": [ci_l, ci_h],
            }

        summary_table[b_id] = {
            "baseline_id": b_id,
            "name": b_rec["name"],
            "head_mean_r": float(np.mean(head_r)) if len(head_r) > 0 else 0.0,
            "head_std_r": float(np.std(head_r, ddof=1)) if len(head_r) > 1 else 0.0,
            "tail_mean_r": float(np.mean(tail_r)) if len(tail_r) > 0 else 0.0,
            "tail_std_r": float(np.std(tail_r, ddof=1)) if len(tail_r) > 1 else 0.0,
            "head_mean_delta_r": float(np.mean(head_dr)) if len(head_dr) > 0 else 0.0,
            "tail_mean_delta_r": float(np.mean(tail_dr)) if len(tail_dr) > 0 else 0.0,
            "tail_sensitivity_d_tail": float(np.mean(tail_dr)) if len(tail_dr) > 0 else 0.0,
            "tail_std_diff": float(np.std(tail_r, ddof=1) - np.std(head_r, ddof=1)) if len(tail_r) > 1 and len(head_r) > 1 else 0.0,
            "paired_diff_vs_b6": paired_diff_info,
        }

    return {
        "summary_table": summary_table,
        "detailed_records": baseline_records,
    }
