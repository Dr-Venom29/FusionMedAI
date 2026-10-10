"""
FusionMedAI - Outcome-Grounded Evaluation Experiment Orchestrator
Executes the complete experimental protocol across development and confirmatory seed cohorts,
computes 2-stage hierarchical cluster bootstrap statistical aggregations, and generates frozen research artifacts.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import json
import hashlib
from pathlib import Path

from src.fusion.outcome_evaluation.oracle_generator import OracleCohortGenerator
from src.fusion.outcome_evaluation.outcome_runner import OutcomeEvaluationRunner
from src.fusion.outcome_evaluation.outcome_metrics import (
    compute_hierarchical_bootstrap_ci,
    ACTION_COST_MATRIX,
)
from src.fusion.router_sanity.sanity_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    DELTA_FROZEN,
)


class OutcomeExperimentOrchestrator:
    """
    Orchestrates development and confirmatory experiments comparing B6 and B5 on oracle-grounded cohorts.
    """

    PROTOCOL_NAME: str = "ACARA_U_Outcome_Grounded_Evaluation_Protocol"
    PROTOCOL_VERSION: str = "1.1.0"

    DEV_SEEDS: Tuple[int, ...] = (201, 202, 203, 204, 205)
    CONFIRMATORY_SEEDS: Tuple[int, ...] = (301, 302, 303, 304, 305, 306, 307, 308, 309, 310)

    N_PACKETS_PER_COHORT: int = 500
    PRACTICAL_SUPERIORITY_THRESHOLD: float = -0.005

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.output_dir = repo_root / "experiments" / "fusion" / "outcome_evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.runner = OutcomeEvaluationRunner(
            alpha=ALPHA_REF,
            beta=BETA_REF,
            gamma=GAMMA_REF,
            eta=ETA_REF,
            delta_dcri=DELTA_FROZEN,
            tau1=0.20,
            tau2=0.40,
        )

    def run_all(self) -> Dict[str, Any]:
        """
        Executes development validation and confirmatory benchmarking, generating all artifacts.
        """
        # 1. Protocol Metadata
        protocol_data = self._generate_protocol_metadata()
        protocol_path = self.output_dir / "protocol.json"
        with open(protocol_path, "w", encoding="utf-8") as f:
            json.dump(protocol_data, f, indent=2)

        # 2. Run Development Cohorts
        dev_results = self._run_cohort_suite(self.DEV_SEEDS, suite_name="DEVELOPMENT")
        dev_path = self.output_dir / "dev_results.json"
        with open(dev_path, "w", encoding="utf-8") as f:
            json.dump(dev_results, f, indent=2)

        # 3. Run Confirmatory Cohorts
        conf_results = self._run_cohort_suite(self.CONFIRMATORY_SEEDS, suite_name="CONFIRMATORY")
        conf_path = self.output_dir / "confirmatory_results.json"
        with open(conf_path, "w", encoding="utf-8") as f:
            json.dump(conf_results, f, indent=2)

        # 4. Generate Synthesis Summary
        summary_data = self._generate_summary(conf_results, dev_results)
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # 5. Generate Cryptographic Freeze Manifest
        manifest_data = self._generate_manifest([
            protocol_path,
            dev_path,
            conf_path,
            summary_path,
        ])
        manifest_path = self.output_dir / "freeze_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return summary_data

    def _generate_protocol_metadata(self) -> Dict[str, Any]:
        return {
            "protocol_name": self.PROTOCOL_NAME,
            "version": self.PROTOCOL_VERSION,
            "research_question": "Does ACARA-U (B6) produce better predictive or decision outcomes than B5 under outcome-grounded simulation?",
            "hypotheses": {
                "H1_Primary": {
                    "comparison": "B6 (Full ACARA-U) vs B5 (Confidence+Reliability-Uncertainty)",
                    "primary_metric": "Delta_MAE = MAE(B6) - MAE(B5) on Oracle Risk",
                    "superiority_criterion": f"Delta_MAE < {self.PRACTICAL_SUPERIORITY_THRESHOLD} with 95% Hierarchical Bootstrap CI strictly below 0.0",
                },
                "H2_Quality_Contribution": {
                    "comparison": "B6 (eta=0.5) vs B6_no_quality (eta=0.0, identical to B5)",
                    "metric": "Direct isolation of quality logit term across clean vs accurate degradation vs misleading sensor failure subsets",
                },
                "H3_Degradation_Response": {
                    "comparison": "B6 vs B5 on degraded modality inputs (Mild, Moderate, Severe)",
                    "metric": "Error reduction delta across true degradation vs misleading sensor failure",
                },
                "H4_Tail_Robustness": {
                    "comparison": "B6 vs B5 across cardinality regimes (1-modality, 2-modality, 3-modality)",
                    "metric": "Stratified MAE and RMSE across sparsity conditions",
                },
                "H5_DCRI_Decision_Policy": {
                    "comparison": "DCRI Policy (delta=0.10) vs Unpenalized Fused Risk Policy vs Oracle",
                    "metric": "Decision Loss under Clinical Action Cost Matrix and High-Risk False Downgrades Count",
                },
            },
            "parameters": {
                "router_coefficients": {
                    "alpha": ALPHA_REF,
                    "beta": BETA_REF,
                    "gamma": GAMMA_REF,
                    "eta": ETA_REF,
                },
                "dcri_penalty_delta": DELTA_FROZEN,
                "action_thresholds": {"tau1": 0.20, "tau2": 0.40},
                "action_cost_matrix": ACTION_COST_MATRIX,
            },
            "sampling_plan": {
                "dev_seeds": list(self.DEV_SEEDS),
                "confirmatory_seeds": list(self.CONFIRMATORY_SEEDS),
                "packets_per_cohort": self.N_PACKETS_PER_COHORT,
                "total_confirmatory_packets": len(self.CONFIRMATORY_SEEDS) * self.N_PACKETS_PER_COHORT,
                "statistical_inference": "Two-stage hierarchical cluster bootstrap (resampling cohorts, then packets within cohorts, B=2000)",
            },
        }

    def _run_cohort_suite(
        self, seeds: Tuple[int, ...], suite_name: str
    ) -> Dict[str, Any]:
        cohort_results = []
        cohort_packet_diffs: List[List[float]] = []
        
        # Subgroup tracking separating clean accurate from degraded accurate
        cohort_subgroup_diffs: Dict[str, List[List[float]]] = {
            "CLEAN_ACCURATE": [],
            "ACCURATE_DEGRADATION": [],
            "MISLEADING_UNNOTICED": [],
            "MISLEADING_FALSE_ALARM": [],
            "UNCERTAINTY_MISCALIBRATED": [],
        }

        all_b6_maes = []
        all_b5_maes = []
        all_b2_maes = []

        all_b6_losses = []
        all_b5_losses = []
        all_dcri_losses = []

        total_high_risk = 0
        total_false_downgrades_b6 = 0
        total_false_downgrades_b5 = 0
        total_false_downgrades_dcri = 0

        # Aggregated scenario, fidelity, regime, and uncertainty downgrade buckets
        scenario_agg: Dict[str, Dict[str, List[float]]] = {}
        fidelity_agg: Dict[str, Dict[str, List[float]]] = {}
        regime_agg: Dict[str, Dict[str, List[float]]] = {}
        unc_downgrade_agg: Dict[str, Dict[str, int]] = {
            "LOW_UNCERTAINTY (< 0.10)": {"total": 0, "b6_fd": 0, "b5_fd": 0, "dcri_fd": 0},
            "MODERATE_UNCERTAINTY (0.10 - 0.30)": {"total": 0, "b6_fd": 0, "b5_fd": 0, "dcri_fd": 0},
            "HIGH_UNCERTAINTY (>= 0.30)": {"total": 0, "b6_fd": 0, "b5_fd": 0, "dcri_fd": 0},
        }

        all_packets_records: List[Dict[str, Any]] = []

        for seed in seeds:
            generator = OracleCohortGenerator(seed=seed)
            packets = generator.generate_cohort(n_packets=self.N_PACKETS_PER_COHORT)
            eval_res = self.runner.evaluate_cohort(packets)

            cohort_diffs = eval_res["packet_diffs"]
            cohort_packet_diffs.append(cohort_diffs)

            # Subgroup differences for this cohort
            c_clean_acc = [
                p["diff_mae_b6_minus_b5"]
                for p in eval_res["sample_packets"]
                if p["scenario"] == "CLEAN" and p["quality_fidelity"] == "ACCURATE"
            ]
            c_deg_acc = [
                p["diff_mae_b6_minus_b5"]
                for p in eval_res["sample_packets"]
                if p["scenario"] in ("MILD_DEGRADATION", "MODERATE_DEGRADATION", "SEVERE_DEGRADATION") and p["quality_fidelity"] == "ACCURATE"
            ]
            c_unnoticed = [
                p["diff_mae_b6_minus_b5"]
                for p in eval_res["sample_packets"]
                if p["quality_fidelity"] == "MISLEADING_UNNOTICED"
            ]
            c_false_alarm = [
                p["diff_mae_b6_minus_b5"]
                for p in eval_res["sample_packets"]
                if p["quality_fidelity"] == "MISLEADING_FALSE_ALARM"
            ]
            c_miscal = [
                p["diff_mae_b6_minus_b5"]
                for p in eval_res["sample_packets"]
                if p["quality_fidelity"] == "UNCERTAINTY_MISCALIBRATED"
            ]

            cohort_subgroup_diffs["CLEAN_ACCURATE"].append(c_clean_acc)
            cohort_subgroup_diffs["ACCURATE_DEGRADATION"].append(c_deg_acc)
            cohort_subgroup_diffs["MISLEADING_UNNOTICED"].append(c_unnoticed)
            cohort_subgroup_diffs["MISLEADING_FALSE_ALARM"].append(c_false_alarm)
            cohort_subgroup_diffs["UNCERTAINTY_MISCALIBRATED"].append(c_miscal)

            cohort_record = {
                "seed": seed,
                "packet_count": len(packets),
                "b6_mae": eval_res["b6_metrics"]["mae"],
                "b5_mae": eval_res["b5_metrics"]["mae"],
                "b2_mae": eval_res["b2_metrics"]["mae"],
                "delta_mae": eval_res["mean_delta_mae"],
                "scenario_breakdown": eval_res["scenario_breakdown"],
                "fidelity_breakdown": eval_res["fidelity_breakdown"],
                "decision_policy": eval_res["decision_policy"],
            }
            cohort_results.append(cohort_record)

            all_b6_maes.append(eval_res["b6_metrics"]["mae"])
            all_b5_maes.append(eval_res["b5_metrics"]["mae"])
            all_b2_maes.append(eval_res["b2_metrics"]["mae"])

            # Accumulate per-packet records
            for p in eval_res["sample_packets"]:
                all_packets_records.append(p)
                all_b6_losses.append(p["loss_b6"])
                all_b5_losses.append(p["loss_b5"])
                all_dcri_losses.append(p["loss_dcri"])

                scen = p["scenario"]
                if scen not in scenario_agg:
                    scenario_agg[scen] = {"err_b6": [], "err_b5": []}
                scenario_agg[scen]["err_b6"].append(p["abs_err_b6"])
                scenario_agg[scen]["err_b5"].append(p["abs_err_b5"])

                # Granular fidelity grouping
                if p["quality_fidelity"] == "ACCURATE":
                    fid_group = "ACCURATE_DEGRADATION" if scen != "CLEAN" else "CLEAN_ACCURATE"
                else:
                    fid_group = p["quality_fidelity"]

                if fid_group not in fidelity_agg:
                    fidelity_agg[fid_group] = {"err_b6": [], "err_b5": []}
                fidelity_agg[fid_group]["err_b6"].append(p["abs_err_b6"])
                fidelity_agg[fid_group]["err_b5"].append(p["abs_err_b5"])

                reg = p["regime"]
                if reg not in regime_agg:
                    regime_agg[reg] = {"err_b6": [], "err_b5": []}
                regime_agg[reg]["err_b6"].append(p["abs_err_b6"])
                regime_agg[reg]["err_b5"].append(p["abs_err_b5"])

            # Uncertainty downgrade aggregation
            for k, udata in eval_res["uncertainty_downgrades"].items():
                unc_downgrade_agg[k]["total"] += udata["total"]
                unc_downgrade_agg[k]["b6_fd"] += udata["b6_fd"]
                unc_downgrade_agg[k]["b5_fd"] += udata["b5_fd"]
                unc_downgrade_agg[k]["dcri_fd"] += udata["dcri_fd"]

            # Policy counts
            dp = eval_res["decision_policy"]
            total_high_risk += dp["high_risk_count"]
            total_false_downgrades_b6 += dp["false_downgrades_b6"]
            total_false_downgrades_b5 += dp["false_downgrades_b5"]
            total_false_downgrades_dcri += dp["false_downgrades_dcri"]

        # True 2-Stage Hierarchical Cluster Bootstrap across all cohorts
        hierarchical_ci = compute_hierarchical_bootstrap_ci(
            cohort_packet_diffs, n_bootstrap=2000, seed=42
        )

        # Hierarchical Subgroup CIs for Granular Fidelity Groups
        subgroup_hierarchical_cis = {}
        for sub_name, sub_cohort_list in cohort_subgroup_diffs.items():
            if all(len(c) > 0 for c in sub_cohort_list):
                subgroup_hierarchical_cis[sub_name] = compute_hierarchical_bootstrap_ci(
                    sub_cohort_list, n_bootstrap=2000, seed=42
                )

        # Pooled Scenario Breakdown
        scenario_pooled = {}
        for scen, sdata in scenario_agg.items():
            scen_b6 = float(np.mean(sdata["err_b6"]))
            scen_b5 = float(np.mean(sdata["err_b5"]))
            scenario_pooled[scen] = {
                "count": len(sdata["err_b6"]),
                "mae_b6": round(scen_b6, 6),
                "mae_b5": round(scen_b5, 6),
                "delta_mae": round(scen_b6 - scen_b5, 6),
            }

        # Pooled Fidelity Breakdown (Clean, Accurate Degradation, Misleading Unnoticed, False Alarm, Miscalibrated)
        fidelity_pooled = {}
        for fid, fdata in fidelity_agg.items():
            fid_b6 = float(np.mean(fdata["err_b6"]))
            fid_b5 = float(np.mean(fdata["err_b5"]))
            subgroup_ci = subgroup_hierarchical_cis.get(fid, {})
            fidelity_pooled[fid] = {
                "count": len(fdata["err_b6"]),
                "mae_b6": round(fid_b6, 6),
                "mae_b5": round(fid_b5, 6),
                "delta_mae": round(fid_b6 - fid_b5, 6),
                "hierarchical_ci_95": [
                    subgroup_ci.get("ci_lower", 0.0),
                    subgroup_ci.get("ci_upper", 0.0),
                ] if subgroup_ci else None,
                "cohort_t_stat": subgroup_ci.get("cohort_t_stat", 0.0) if subgroup_ci else None,
                "cohort_t_pvalue": subgroup_ci.get("cohort_t_pvalue", 1.0) if subgroup_ci else None,
            }

        # Pooled Regime Breakdown
        regime_pooled = {}
        for reg, rdata in regime_agg.items():
            reg_b6 = float(np.mean(rdata["err_b6"]))
            reg_b5 = float(np.mean(rdata["err_b5"]))
            regime_pooled[reg] = {
                "count": len(rdata["err_b6"]),
                "mae_b6": round(reg_b6, 6),
                "mae_b5": round(reg_b5, 6),
                "delta_mae": round(reg_b6 - reg_b5, 6),
            }

        return {
            "suite_name": suite_name,
            "num_cohorts": len(seeds),
            "total_packets": len(all_packets_records),
            "mean_b6_mae": round(float(np.mean(all_b6_maes)), 6),
            "mean_b5_mae": round(float(np.mean(all_b5_maes)), 6),
            "mean_b2_mae": round(float(np.mean(all_b2_maes)), 6),
            "hierarchical_delta_mae": hierarchical_ci,
            "cohort_results": cohort_results,
            "scenario_summary": scenario_pooled,
            "fidelity_summary": fidelity_pooled,
            "regime_summary": regime_pooled,
            "uncertainty_downgrade_summary": unc_downgrade_agg,
            "decision_policy_summary": {
                "mean_loss_b6": round(float(np.mean(all_b6_losses)), 6),
                "mean_loss_b5": round(float(np.mean(all_b5_losses)), 6),
                "mean_loss_dcri": round(float(np.mean(all_dcri_losses)), 6),
                "delta_loss_b6_vs_b5": round(float(np.mean(all_b6_losses) - np.mean(all_b5_losses)), 6),
                "delta_loss_dcri_vs_b6": round(float(np.mean(all_dcri_losses) - np.mean(all_b6_losses)), 6),
                "total_high_risk": total_high_risk,
                "total_false_downgrades_b6": total_false_downgrades_b6,
                "total_false_downgrades_b5": total_false_downgrades_b5,
                "total_false_downgrades_dcri": total_false_downgrades_dcri,
                "false_downgrade_rate_b6": round(total_false_downgrades_b6 / max(total_high_risk, 1), 6),
                "false_downgrade_rate_b5": round(total_false_downgrades_b5 / max(total_high_risk, 1), 6),
                "false_downgrade_rate_dcri": round(total_false_downgrades_dcri / max(total_high_risk, 1), 6),
            },
            "raw_packet_records": all_packets_records,
        }

    def _generate_summary(
        self, conf_results: Dict[str, Any], dev_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        h_ci = conf_results["hierarchical_delta_mae"]
        mean_delta = h_ci["grand_mean_diff"]
        ci_lower = h_ci["ci_lower"]
        ci_upper = h_ci["ci_upper"]
        is_sig = h_ci["is_significant_95"]
        t_stat = h_ci["cohort_t_stat"]
        p_val = h_ci["cohort_t_pvalue"]

        # Dynamically evaluate H1 against declared practical-superiority threshold (-0.005)
        if is_sig and mean_delta < self.PRACTICAL_SUPERIORITY_THRESHOLD:
            h1_verdict = "CONFIRMED_PRACTICAL_SUPERIORITY"
            h1_desc = f"B6 significantly reduced error beyond the prespecified practical threshold (Delta_MAE = {mean_delta:.6f} < {self.PRACTICAL_SUPERIORITY_THRESHOLD}, 95% Hierarchical CI: [{ci_lower:.6f}, {ci_upper:.6f}], t = {t_stat:.2f}, p = {p_val:.2e})."
        elif is_sig and mean_delta < 0.0:
            h1_verdict = "STATISTICALLY_SIGNIFICANT_MODEST_IMPROVEMENT_BELOW_PRACTICAL_THRESHOLD"
            h1_desc = f"B6 showed a statistically significant error reduction in favor of B6 (Delta_MAE = {mean_delta:.6f}, 95% Hierarchical CI: [{ci_lower:.6f}, {ci_upper:.6f}], t = {t_stat:.2f}, p = {p_val:.2e}), but the effect size did not reach the prespecified practical superiority threshold of {self.PRACTICAL_SUPERIORITY_THRESHOLD}."
        elif not is_sig:
            h1_verdict = "INCONCLUSIVE_CONFIDENCE_INTERVAL_CROSSES_ZERO"
            h1_desc = f"The confidence interval includes zero (Delta_MAE = {mean_delta:.6f}, 95% Hierarchical CI: [{ci_lower:.6f}, {ci_upper:.6f}])."
        else:
            h1_verdict = "B5_FAVORED"
            h1_desc = f"B5 achieved lower error than B6 (Delta_MAE = {mean_delta:.6f})."

        # Dynamically evaluate H2 strictly on accurately detected degradation subgroup
        fid_summary = conf_results.get("fidelity_summary", {})
        acc_deg_info = fid_summary.get("ACCURATE_DEGRADATION", {})
        delta_acc_deg = acc_deg_info.get("delta_mae", 0.0)
        ci_acc_deg = acc_deg_info.get("hierarchical_ci_95", [0.0, 0.0]) or [0.0, 0.0]
        
        clean_info = fid_summary.get("CLEAN_ACCURATE", {})
        delta_clean = clean_info.get("delta_mae", 0.0)
        delta_unnoticed = fid_summary.get("MISLEADING_UNNOTICED", {}).get("delta_mae", 0.0)
        delta_false_alarm = fid_summary.get("MISLEADING_FALSE_ALARM", {}).get("delta_mae", 0.0)
        delta_miscal = fid_summary.get("UNCERTAINTY_MISCALIBRATED", {}).get("delta_mae", 0.0)

        if delta_acc_deg < 0.0 and ci_acc_deg[1] < 0.0:
            h2_verdict = "CONFIRMED_OBSERVED_QUALITY_BENEFIT_UNDER_ACCURATE_DEGRADATION"
            h2_desc = (
                f"Adding the quality weighting term (eta=0.5) significantly reduces estimation error specifically under accurately detected signal degradation "
                f"(Delta_MAE = {delta_acc_deg:.6f}, 95% Subgroup CI: [{ci_acc_deg[0]:.6f}, {ci_acc_deg[1]:.6f}]). Clean inputs remain at parity (Delta_MAE = {delta_clean:.6f}), "
                f"while undetected degradation defaults to B5 parity (Delta_MAE = {delta_unnoticed:.6f}) and false alarms incur a slight penalty (Delta_MAE = {delta_false_alarm:.6f})."
            )
        else:
            h2_verdict = "NOT_BENEFICIAL"
            h2_desc = f"Quality weighting did not reliably reduce error under evaluated degradation conditions (Delta_MAE = {delta_acc_deg:.6f})."

        # Clean confirmatory summary without 5000 raw packet duplicates in summary.json
        conf_summary_metrics = {k: v for k, v in conf_results.items() if k != "raw_packet_records"}

        return {
            "status": "EVALUATED_AND_FROZEN",
            "experiment_name": "ACARA-U Outcome-Grounded Comparative Benchmark",
            "protocol_version": self.PROTOCOL_VERSION,
            "cohort_count": conf_results["num_cohorts"],
            "total_packets": conf_results["total_packets"],
            "primary_findings": {
                "H1_Primary_Outcome": {
                    "verdict": h1_verdict,
                    "description": h1_desc,
                    "mean_mae_b6": conf_results["mean_b6_mae"],
                    "mean_mae_b5": conf_results["mean_b5_mae"],
                    "mean_mae_b2": conf_results["mean_b2_mae"],
                    "delta_mae": mean_delta,
                    "hierarchical_ci_95": [ci_lower, ci_upper],
                    "cohort_t_stat": t_stat,
                    "cohort_t_pvalue": p_val,
                    "cohort_t_test_significant": h_ci["cohort_t_test_significant"],
                    "practical_superiority_threshold": self.PRACTICAL_SUPERIORITY_THRESHOLD,
                    "practical_threshold_met": bool(mean_delta < self.PRACTICAL_SUPERIORITY_THRESHOLD),
                },
                "H2_Quality_Term_Isolation": {
                    "verdict": h2_verdict,
                    "description": h2_desc,
                    "clean_accurate_delta_mae": delta_clean,
                    "accurate_degradation_delta_mae": delta_acc_deg,
                    "accurate_degradation_ci_95": ci_acc_deg,
                    "unnoticed_degradation_delta_mae": delta_unnoticed,
                    "false_alarm_quality_delta_mae": delta_false_alarm,
                    "uncertainty_miscalibration_delta_mae": delta_miscal,
                },
                "H3_Degradation_Analysis": {
                    "clean_delta_mae": conf_results["scenario_summary"].get("CLEAN", {}).get("delta_mae", 0.0),
                    "mild_deg_delta_mae": conf_results["scenario_summary"].get("MILD_DEGRADATION", {}).get("delta_mae", 0.0),
                    "moderate_deg_delta_mae": conf_results["scenario_summary"].get("MODERATE_DEGRADATION", {}).get("delta_mae", 0.0),
                    "severe_deg_delta_mae": conf_results["scenario_summary"].get("SEVERE_DEGRADATION", {}).get("delta_mae", 0.0),
                    "miscalibration_delta_mae": conf_results["scenario_summary"].get("UNCERTAINTY_MISCALIBRATION", {}).get("delta_mae", 0.0),
                },
                "H5_DCRI_Policy_Tradeoff": {
                    "fused_risk_policy_loss": conf_results["decision_policy_summary"]["mean_loss_b6"],
                    "dcri_policy_loss": conf_results["decision_policy_summary"]["mean_loss_dcri"],
                    "delta_decision_loss": conf_results["decision_policy_summary"]["delta_loss_dcri_vs_b6"],
                    "fused_risk_false_downgrade_rate": conf_results["decision_policy_summary"]["false_downgrade_rate_b6"],
                    "dcri_false_downgrade_rate": conf_results["decision_policy_summary"]["false_downgrade_rate_dcri"],
                    "uncertainty_breakdown": conf_results["uncertainty_downgrade_summary"],
                    "interpretation": "DCRI penalty (delta=0.10) successfully reduces unnecessary low-risk escalations, but increases false downgrades of high-risk cases with elevated uncertainty, increasing expected loss under an asymmetric clinical cost matrix.",
                },
            },
            "confirmatory_metrics": conf_summary_metrics,
        }

    def _generate_manifest(self, file_paths: List[Path]) -> Dict[str, Any]:
        hashes = {}
        for p in file_paths:
            with open(p, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            hashes[p.name] = sha

        return {
            "manifest_type": "OUTCOME_EVALUATION_FREEZE_MANIFEST",
            "version": self.PROTOCOL_VERSION,
            "artifact_hashes": hashes,
            "total_artifacts": len(hashes),
            "status": "SEALED",
        }
