"""
src/fusion/combination_analysis/combination_engine.py
Phase C11.9: Modality-Combination Distribution & Tail-Robustness Orchestration Engine

Coordinates:
1. Packet generation under controlled distributions (D1 Balanced, D2 Moderate, D3 Strong Tail).
2. Evaluation through frozen ACARA-U router and DCRIEngine (delta=0.20).
3. Tier classification (Head, Middle, Tail).
4. Combination-level metrics, uncertainty, and conflict quantification.
5. Baseline benchmarking across B1–B6.
6. Bootstrap confidence interval estimation (1,000 resamples).
7. Cross-distribution sensitivity analysis.
"""

from dataclasses import asdict
from typing import Dict, Any, List, Tuple, Sequence, Optional
import numpy as np

from src.fusion.baselines.decision_packet import ControlledDecisionPacket
from src.fusion.dcri.dcri_engine import DCRIEngine
from src.fusion.dcri.dcri_result import DCRIResult
from src.fusion.combination_analysis.combination_definition import (
    NON_EMPTY_COMBINATIONS,
    COMBINATION_RFC,
    COMBINATION_EMPTY,
    apply_combination_to_packet,
)
from src.fusion.combination_analysis.distribution_generator import (
    ALL_DISTRIBUTIONS,
    DISTRIBUTION_D1_BALANCED,
    DISTRIBUTION_D2_MODERATE_HEAD_TAIL,
    DISTRIBUTION_D3_STRONG_LONG_TAIL,
    get_distribution_config,
    generate_stratified_distribution_cohort,
    AssignedPacket,
)
from src.fusion.combination_analysis.head_tail_classifier import (
    classify_distribution_tiers,
    DistributionTierAssignment,
)
from src.fusion.combination_analysis.combination_metrics import (
    calc_distribution_stats,
    compute_routing_entropy,
    compute_conflict_metrics,
    MIN_SAMPLE_SIZE,
)
from src.fusion.combination_analysis.tail_robustness import (
    compute_tier_summary,
    compute_tail_robustness_comparison,
    TierSummaryRecord,
    TailRobustnessComparison,
)
from src.fusion.combination_analysis.baseline_evaluator import (
    evaluate_baselines_on_combinations,
)
from src.fusion.combination_analysis.bootstrap_analysis import (
    compute_bootstrap_ci_1d,
)
from src.fusion.combination_analysis.calibration_analysis import (
    analyze_modality_calibration_availability,
)
from src.fusion.combination_analysis.combination_result import (
    CombinationEvaluationRecord,
    DistributionEvaluationRecord,
    CrossDistributionSensitivityRecord,
)


class ModalityCombinationEngine:
    """
    Main orchestration engine for Phase C11.9.
    """

    def __init__(self, delta: float = 0.20, bootstrap_resamples: int = 1000, seed: int = 115):
        self.delta = delta
        self.bootstrap_resamples = bootstrap_resamples
        self.seed = seed
        self.dcri_engine = DCRIEngine()

    def evaluate_distribution(
        self,
        cohort: Sequence[ControlledDecisionPacket],
        dist_id: str,
    ) -> DistributionEvaluationRecord:
        """
        Executes complete evaluation for a single frequency distribution over the cohort.
        """
        dist_config = get_distribution_config(dist_id)
        tier_assignment = classify_distribution_tiers(dist_config)

        # 1. Generate stratified assigned packets
        assigned_cohort = generate_stratified_distribution_cohort(
            cohort=cohort,
            dist_id=dist_id,
            seed=self.seed,
        )

        # 2. Pre-evaluate full RFC reference for each original packet: orig_id -> (r_fusion_rfc, dcri_rfc)
        rfc_references: Dict[str, Tuple[float, float]] = {}
        for pkt in cohort:
            rfc_pkt = apply_combination_to_packet(pkt, COMBINATION_RFC)
            rfc_res = self.dcri_engine.evaluate_packet(rfc_pkt, delta=self.delta)
            rfc_references[pkt.packet_id] = (rfc_res.r_fusion, rfc_res.dcri)

        # 3. Group evaluations by combination
        raw_by_comb: Dict[str, Dict[str, List[float]]] = {
            c: {
                "w_retina": [],
                "w_foot": [],
                "w_clinical": [],
                "entropy": [],
                "r_fusion": [],
                "dcri": [],
                "u_sum": [],
                "delta_max": [],
                "delta_mean": [],
                "sigma_w": [],
                "delta_r": [],
                "delta_dcri": [],
            }
            for c in NON_EMPTY_COMBINATIONS
        }

        global_r_list: List[float] = []
        global_dcri_list: List[float] = []
        global_entropy_list: List[float] = []
        global_u_sum_list: List[float] = []
        global_delta_max_list: List[float] = []

        for ap in assigned_cohort:
            c = ap.assigned_combination
            pkt = ap.packet
            orig_id = ap.original_packet_id

            dcri_res: DCRIResult = self.dcri_engine.evaluate_packet(pkt, delta=self.delta)
            w_dict = dcri_res.modality_weights
            r_f = dcri_res.r_fusion
            dcri_val = dcri_res.dcri
            u_s = dcri_res.u_sum
            h = compute_routing_entropy(w_dict)

            # Conflict
            risks = {
                "retina": pkt.retina.risk,
                "foot": pkt.foot.risk,
                "clinical": pkt.clinical.risk,
            }
            active_m = [m for m, av in [("retina", pkt.retina.availability), ("foot", pkt.foot.availability), ("clinical", pkt.clinical.availability)] if av]
            conf = compute_conflict_metrics(risks, w_dict, active_m)

            rfc_r, rfc_dcri = rfc_references[orig_id]
            dr = abs(rfc_r - r_f)
            ddcri = abs(rfc_dcri - dcri_val)

            raw = raw_by_comb[c]
            raw["w_retina"].append(w_dict.get("retina", 0.0))
            raw["w_foot"].append(w_dict.get("foot", 0.0))
            raw["w_clinical"].append(w_dict.get("clinical", 0.0))
            raw["entropy"].append(h)
            raw["r_fusion"].append(r_f)
            raw["dcri"].append(dcri_val)
            raw["u_sum"].append(u_s)
            raw["delta_max"].append(conf["delta_max"])
            raw["delta_mean"].append(conf["delta_mean"])
            raw["sigma_w"].append(conf["sigma_w"])
            raw["delta_r"].append(dr)
            raw["delta_dcri"].append(ddcri)

            global_r_list.append(r_f)
            global_dcri_list.append(dcri_val)
            global_entropy_list.append(h)
            global_u_sum_list.append(u_s)
            global_delta_max_list.append(conf["delta_max"])

        # 4. Construct CombinationEvaluationRecord per combination
        comb_records: Dict[str, CombinationEvaluationRecord] = {}
        comb_results_for_tier: Dict[str, Dict[str, Any]] = {}

        for c in NON_EMPTY_COMBINATIONS:
            raw = raw_by_comb[c]
            n_pkts = len(raw["r_fusion"])
            t_rec = tier_assignment.records[c]

            r_stats = calc_distribution_stats(raw["r_fusion"])
            dcri_stats = calc_distribution_stats(raw["dcri"])
            h_stats = calc_distribution_stats(raw["entropy"])
            u_stats = calc_distribution_stats(raw["u_sum"])
            dr_stats = calc_distribution_stats(raw["delta_r"])

            # Bootstrap CIs
            ci_r = compute_bootstrap_ci_1d(raw["r_fusion"], n_resamples=self.bootstrap_resamples, seed=self.seed)
            ci_dcri = compute_bootstrap_ci_1d(raw["dcri"], n_resamples=self.bootstrap_resamples, seed=self.seed)
            ci_dr = compute_bootstrap_ci_1d(raw["delta_r"], n_resamples=self.bootstrap_resamples, seed=self.seed)

            status = "SUCCESS" if n_pkts >= MIN_SAMPLE_SIZE else "INSUFFICIENT_SAMPLE"

            comb_records[c] = CombinationEvaluationRecord(
                combination_id=c,
                distribution_id=dist_id,
                n_packets=n_pkts,
                frequency_probability=t_rec.probability,
                rank=t_rec.rank,
                tier=t_rec.tier,
                mean_w_retina=float(np.mean(raw["w_retina"])) if raw["w_retina"] else 0.0,
                mean_w_foot=float(np.mean(raw["w_foot"])) if raw["w_foot"] else 0.0,
                mean_w_clinical=float(np.mean(raw["w_clinical"])) if raw["w_clinical"] else 0.0,
                mean_entropy=h_stats["mean"],
                std_entropy=h_stats["std"],
                mean_r_fusion=r_stats["mean"],
                std_r_fusion=r_stats["std"],
                mean_dcri=dcri_stats["mean"],
                std_dcri=dcri_stats["std"],
                mean_u_sum=u_stats["mean"],
                std_u_sum=u_stats["std"],
                mean_delta_max=float(np.mean(raw["delta_max"])) if raw["delta_max"] else 0.0,
                mean_delta_mean=float(np.mean(raw["delta_mean"])) if raw["delta_mean"] else 0.0,
                mean_sigma_w=float(np.mean(raw["sigma_w"])) if raw["sigma_w"] else 0.0,
                mean_delta_r=dr_stats["mean"],
                std_delta_r=dr_stats["std"],
                ci_95_r_fusion=ci_r,
                ci_95_dcri=ci_dcri,
                ci_95_delta_r=ci_dr,
                status=status,
            )

            comb_results_for_tier[c] = {
                "status": "SUCCESS",
                "n_packets": n_pkts,
                "r_fusion_raw": raw["r_fusion"],
                "dcri_raw": raw["dcri"],
                "entropy_raw": raw["entropy"],
                "u_sum_raw": raw["u_sum"],
                "delta_r_raw": raw["delta_r"],
                "delta_dcri_raw": raw["delta_dcri"],
            }

        # 5. Tier Summaries (Head, Middle, Tail)
        head_summary = compute_tier_summary("HEAD", tier_assignment.head_combinations, comb_results_for_tier)
        middle_summary = compute_tier_summary("MIDDLE", tier_assignment.middle_combinations, comb_results_for_tier)
        tail_summary = compute_tier_summary("TAIL", tier_assignment.tail_combinations, comb_results_for_tier)

        tail_vs_head = compute_tail_robustness_comparison(dist_id, head_summary, tail_summary)

        # 6. Comparative Baseline Benchmark
        baseline_eval = evaluate_baselines_on_combinations(
            assigned_cohort=assigned_cohort,
            full_cohort=cohort,
            head_combinations=tier_assignment.head_combinations,
            tail_combinations=tier_assignment.tail_combinations,
        )

        return DistributionEvaluationRecord(
            distribution_id=dist_id,
            name=dist_config.name,
            total_packets=len(assigned_cohort),
            head_to_tail_ratio=tier_assignment.head_to_tail_ratio,
            combinations=comb_records,
            head_summary=asdict(head_summary),
            middle_summary=asdict(middle_summary),
            tail_summary=asdict(tail_summary),
            tail_vs_head_contrast=asdict(tail_vs_head),
            baseline_comparison=baseline_eval,
            global_mean_r_fusion=float(np.mean(global_r_list)),
            global_std_r_fusion=float(np.std(global_r_list, ddof=1)),
            global_mean_dcri=float(np.mean(global_dcri_list)),
            global_std_dcri=float(np.std(global_dcri_list, ddof=1)),
            global_mean_entropy=float(np.mean(global_entropy_list)),
            global_mean_u_sum=float(np.mean(global_u_sum_list)),
            global_mean_delta_max=float(np.mean(global_delta_max_list)),
        )

    def evaluate_all_distributions(
        self,
        cohort: Sequence[ControlledDecisionPacket],
    ) -> Tuple[Dict[str, DistributionEvaluationRecord], CrossDistributionSensitivityRecord]:
        """
        Executes evaluation across all three pre-registered distributions (D1, D2, D3)
        and computes cross-distribution global sensitivity.
        """
        dist_records: Dict[str, DistributionEvaluationRecord] = {}
        for dist_id in ALL_DISTRIBUTIONS:
            dist_records[dist_id] = self.evaluate_distribution(cohort, dist_id)

        cross_sens = CrossDistributionSensitivityRecord(
            distributions_evaluated=list(ALL_DISTRIBUTIONS),
            global_r_fusion_by_dist={d: rec.global_mean_r_fusion for d, rec in dist_records.items()},
            global_dcri_by_dist={d: rec.global_mean_dcri for d, rec in dist_records.items()},
            global_entropy_by_dist={d: rec.global_mean_entropy for d, rec in dist_records.items()},
            global_u_sum_by_dist={d: rec.global_mean_u_sum for d, rec in dist_records.items()},
            global_delta_max_by_dist={d: rec.global_mean_delta_max for d, rec in dist_records.items()},
            tail_sensitivity_by_dist={
                d: rec.tail_summary.get("mean_delta_r", 0.0) for d, rec in dist_records.items()
            },
            tail_std_r_fusion_by_dist={
                d: rec.tail_summary.get("std_r_fusion", 0.0) for d, rec in dist_records.items()
            },
        )

        return dist_records, cross_sens
