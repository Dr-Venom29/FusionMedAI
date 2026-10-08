"""
src/fusion/combination_analysis/run_combination_experiments.py
Phase C11.9: Modality-Combination Distribution & Tail-Robustness Experiment Execution Script

Executes the complete experimental battery over the frozen N=500 controlled decision cohort (seed 115)
and generates all 16 sealed research artifacts in experiments/fusion/combination_analysis/.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import hashlib
import numpy as np

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.combination_analysis.combination_definition import (
    ALL_COMBINATIONS,
    NON_EMPTY_COMBINATIONS,
    COMBINATION_MASKS,
    COMBINATION_CARDINALITY,
    COMBINATION_ACTIVE_MODALITIES,
    get_combination_spec,
)
from src.fusion.combination_analysis.distribution_generator import (
    ALL_DISTRIBUTIONS,
    DISTRIBUTION_D1_BALANCED,
    DISTRIBUTION_D2_MODERATE_HEAD_TAIL,
    DISTRIBUTION_D3_STRONG_LONG_TAIL,
    get_distribution_config,
    generate_stratified_distribution_cohort,
)
from src.fusion.combination_analysis.head_tail_classifier import (
    classify_distribution_tiers,
)
from src.fusion.combination_analysis.combination_engine import (
    ModalityCombinationEngine,
)
from src.fusion.combination_analysis.calibration_analysis import (
    analyze_modality_calibration_availability,
)


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_all_combination_experiments(repo_root: Path = Path(".")) -> Dict[str, Path]:
    output_dir = repo_root / "experiments" / "fusion" / "combination_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("FusionMedAI Phase C11.9: Modality-Combination Distribution & Tail Robustness")
    print("Cohort Size: N = 500 | PRNG Seed: 115 | Status: RUNNING")
    print("================================================================================")

    # 1. Load Frozen Cohort
    cohort = load_frozen_cohort(repo_root=repo_root, n_packets=500, seed=115)
    delta_eval = 0.20
    engine = ModalityCombinationEngine(delta=delta_eval, bootstrap_resamples=1000, seed=115)

    # Run evaluations across all distributions
    dist_records, cross_sens = engine.evaluate_all_distributions(cohort)

    # --------------------------------------------------------------------------
    # Artifact 1: Configuration Metadata
    # --------------------------------------------------------------------------
    config_data = {
        "phase": "C11.9",
        "name": "Modality-Combination Distribution & Tail-Robustness Analysis",
        "cohort_size": len(cohort),
        "cohort_seed": 115,
        "delta_evaluation_point": delta_eval,
        "bootstrap_resamples": 1000,
        "evaluated_distributions": list(ALL_DISTRIBUTIONS),
        "methodological_boundary": (
            "Controlled decision-packet experiments constructed from independent modality cohorts; "
            "evaluates fusion robustness to combination frequency distributions, not patient-level multimodal prevalence."
        ),
        "registered_hypotheses": {
            "H1": "Decision-level routing and aggregation metrics differ across controlled modality combinations.",
            "H2": "Rare experimental modality combinations exhibit greater variability in selected decision-level metrics than common combinations.",
            "H3": "ACARA-U exhibits lower or more stable tail sensitivity than at least one simpler fusion baseline under the same controlled modality-combination distribution.",
            "H4": "Uncertainty-related outputs differ across modality combinations and contribute to differences in DCRI behavior.",
        },
    }
    f_cfg = output_dir / "experiment_config.json"
    with open(f_cfg, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Artifact 2: Combination Definitions
    # --------------------------------------------------------------------------
    comb_defs = {}
    for c in ALL_COMBINATIONS:
        spec = get_combination_spec(c)
        comb_defs[c] = {
            "combination_id": spec.combination_id,
            "availability_retina": spec.availability_retina,
            "availability_foot": spec.availability_foot,
            "availability_clinical": spec.availability_clinical,
            "active_modalities": list(spec.active_modalities),
            "cardinality": spec.cardinality,
            "is_empty": spec.is_empty,
        }
    f_comb_def = output_dir / "combination_definitions.json"
    with open(f_comb_def, "w", encoding="utf-8") as f:
        json.dump(comb_defs, f, indent=2)

    # --------------------------------------------------------------------------
    # Artifact 3: Distribution Configurations
    # --------------------------------------------------------------------------
    dist_configs_data = {}
    for d_id in ALL_DISTRIBUTIONS:
        d_cfg = get_distribution_config(d_id)
        dist_configs_data[d_id] = {
            "distribution_id": d_cfg.distribution_id,
            "name": d_cfg.name,
            "description": d_cfg.description,
            "probabilities": d_cfg.probabilities,
            "expected_counts_n500": d_cfg.expected_counts_n500,
        }
    f_dist_cfg = output_dir / "distribution_configurations.json"
    with open(f_dist_cfg, "w", encoding="utf-8") as f:
        json.dump(dist_configs_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Artifact 4: Packet Assignment Manifest
    # --------------------------------------------------------------------------
    assignment_manifest = {}
    for d_id in ALL_DISTRIBUTIONS:
        assigned = generate_stratified_distribution_cohort(cohort, d_id, seed=115)
        assignment_manifest[d_id] = [
            {
                "sample_index": ap.sample_index,
                "original_packet_id": ap.original_packet_id,
                "assigned_combination": ap.assigned_combination,
            }
            for ap in assigned
        ]
    f_assign = output_dir / "packet_assignment_manifest.json"
    with open(f_assign, "w", encoding="utf-8") as f:
        json.dump(assignment_manifest, f, indent=2)

    # --------------------------------------------------------------------------
    # Artifact 5: Combination Frequency
    # --------------------------------------------------------------------------
    freq_data = {}
    for d_id in ALL_DISTRIBUTIONS:
        d_rec = dist_records[d_id]
        freq_data[d_id] = {
            c: {
                "n_packets": rec.n_packets,
                "target_probability": rec.frequency_probability,
                "empirical_fraction": rec.n_packets / d_rec.total_packets,
                "rank": rec.rank,
                "tier": rec.tier,
            }
            for c, rec in d_rec.combinations.items()
        }
    f_freq = output_dir / "combination_frequency.json"
    with open(f_freq, "w", encoding="utf-8") as f:
        json.dump(freq_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Artifact 6: Head/Tail Assignment
    # --------------------------------------------------------------------------
    head_tail_data = {}
    for d_id in ALL_DISTRIBUTIONS:
        d_cfg = get_distribution_config(d_id)
        tiers = classify_distribution_tiers(d_cfg)
        head_tail_data[d_id] = {
            "head_combinations": list(tiers.head_combinations),
            "middle_combinations": list(tiers.middle_combinations),
            "tail_combinations": list(tiers.tail_combinations),
            "head_to_tail_ratio": tiers.head_to_tail_ratio,
            "records": {
                c: {
                    "rank": r.rank,
                    "tier": r.tier,
                    "probability": r.probability,
                    "threshold_tier": r.threshold_tier,
                }
                for c, r in tiers.records.items()
            },
        }
    f_ht = output_dir / "head_tail_assignment.json"
    with open(f_ht, "w", encoding="utf-8") as f:
        json.dump(head_tail_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Artifact 7: Combination Metrics (Full Profile)
    # --------------------------------------------------------------------------
    metrics_data = {}
    for d_id in ALL_DISTRIBUTIONS:
        d_rec = dist_records[d_id]
        metrics_data[d_id] = {
            c: {
                "n_packets": rec.n_packets,
                "tier": rec.tier,
                "weights": {
                    "retina": rec.mean_w_retina,
                    "foot": rec.mean_w_foot,
                    "clinical": rec.mean_w_clinical,
                },
                "entropy": {"mean": rec.mean_entropy, "std": rec.std_entropy},
                "r_fusion": {"mean": rec.mean_r_fusion, "std": rec.std_r_fusion, "ci_95": list(rec.ci_95_r_fusion)},
                "dcri": {"mean": rec.mean_dcri, "std": rec.std_dcri, "ci_95": list(rec.ci_95_dcri)},
                "u_sum": {"mean": rec.mean_u_sum, "std": rec.std_u_sum},
                "conflict": {
                    "delta_max": rec.mean_delta_max,
                    "delta_mean": rec.mean_delta_mean,
                    "sigma_w": rec.mean_sigma_w,
                },
                "sensitivity_relative_to_rfc": {
                    "mean_delta_r": rec.mean_delta_r,
                    "std_delta_r": rec.std_delta_r,
                    "ci_95_delta_r": list(rec.ci_95_delta_r),
                },
                "status": rec.status,
            }
            for c, rec in d_rec.combinations.items()
        }
    f_metrics = output_dir / "combination_metrics.json"
    with open(f_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Artifact 8: Uncertainty by Combination
    # --------------------------------------------------------------------------
    unc_data = {}
    for d_id in ALL_DISTRIBUTIONS:
        d_rec = dist_records[d_id]
        unc_data[d_id] = {
            "by_combination": {
                c: {
                    "mean_u_sum": rec.mean_u_sum,
                    "std_u_sum": rec.std_u_sum,
                    "tier": rec.tier,
                }
                for c, rec in d_rec.combinations.items()
            },
            "by_tier": {
                "head_mean_u_sum": d_rec.head_summary["mean_u_sum"],
                "middle_mean_u_sum": d_rec.middle_summary["mean_u_sum"],
                "tail_mean_u_sum": d_rec.tail_summary["mean_u_sum"],
            },
        }
    f_unc = output_dir / "uncertainty_by_combination.json"
    with open(f_unc, "w", encoding="utf-8") as f:
        json.dump(unc_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Artifact 9: Conflict by Combination
    # --------------------------------------------------------------------------
    conf_data = {}
    for d_id in ALL_DISTRIBUTIONS:
        d_rec = dist_records[d_id]
        conf_data[d_id] = {
            c: {
                "mean_delta_max": rec.mean_delta_max,
                "mean_delta_mean": rec.mean_delta_mean,
                "mean_sigma_w": rec.mean_sigma_w,
                "tier": rec.tier,
            }
            for c, rec in d_rec.combinations.items()
        }
    f_conf = output_dir / "conflict_by_combination.json"
    with open(f_conf, "w", encoding="utf-8") as f:
        json.dump(conf_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Artifact 10: Risk by Combination
    # --------------------------------------------------------------------------
    risk_data = {}
    for d_id in ALL_DISTRIBUTIONS:
        d_rec = dist_records[d_id]
        risk_data[d_id] = {
            c: {
                "mean_r_fusion": rec.mean_r_fusion,
                "std_r_fusion": rec.std_r_fusion,
                "ci_95": list(rec.ci_95_r_fusion),
                "delta_r_relative_to_rfc": rec.mean_delta_r,
                "tier": rec.tier,
            }
            for c, rec in d_rec.combinations.items()
        }
    f_risk = output_dir / "risk_by_combination.json"
    with open(f_risk, "w", encoding="utf-8") as f:
        json.dump(risk_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Artifact 11: DCRI by Combination
    # --------------------------------------------------------------------------
    dcri_data = {}
    for d_id in ALL_DISTRIBUTIONS:
        d_rec = dist_records[d_id]
        dcri_data[d_id] = {
            c: {
                "mean_dcri": rec.mean_dcri,
                "std_dcri": rec.std_dcri,
                "ci_95": list(rec.ci_95_dcri),
                "tier": rec.tier,
            }
            for c, rec in d_rec.combinations.items()
        }
    f_dcri = output_dir / "dcri_by_combination.json"
    with open(f_dcri, "w", encoding="utf-8") as f:
        json.dump(dcri_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Artifact 12: Baseline Comparison (B1–B6)
    # --------------------------------------------------------------------------
    base_comp_data = {}
    for d_id in ALL_DISTRIBUTIONS:
        d_rec = dist_records[d_id]
        base_comp_data[d_id] = d_rec.baseline_comparison["summary_table"]
    f_base = output_dir / "baseline_comparison.json"
    with open(f_base, "w", encoding="utf-8") as f:
        json.dump(base_comp_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Artifact 13: Tail Robustness
    # --------------------------------------------------------------------------
    tail_data = {}
    for d_id in ALL_DISTRIBUTIONS:
        d_rec = dist_records[d_id]
        tail_data[d_id] = {
            "head_summary": d_rec.head_summary,
            "middle_summary": d_rec.middle_summary,
            "tail_summary": d_rec.tail_summary,
            "tail_vs_head_contrast": d_rec.tail_vs_head_contrast,
        }
    f_tail = output_dir / "tail_robustness.json"
    with open(f_tail, "w", encoding="utf-8") as f:
        json.dump(tail_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Artifact 14: Bootstrap Confidence Intervals
    # --------------------------------------------------------------------------
    boot_data = {}
    for d_id in ALL_DISTRIBUTIONS:
        d_rec = dist_records[d_id]
        boot_data[d_id] = {
            c: {
                "ci_95_r_fusion": list(rec.ci_95_r_fusion),
                "ci_95_dcri": list(rec.ci_95_dcri),
                "ci_95_delta_r": list(rec.ci_95_delta_r),
            }
            for c, rec in d_rec.combinations.items()
        }
    f_boot = output_dir / "bootstrap_confidence_intervals.json"
    with open(f_boot, "w", encoding="utf-8") as f:
        json.dump(boot_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Artifact 15: Calibration Analysis
    # --------------------------------------------------------------------------
    cal_data = {}
    for d_id in ALL_DISTRIBUTIONS:
        assigned = generate_stratified_distribution_cohort(cohort, d_id, seed=115)
        cal_data[d_id] = analyze_modality_calibration_availability(assigned)
    f_cal = output_dir / "calibration_analysis.json"
    with open(f_cal, "w", encoding="utf-8") as f:
        json.dump(cal_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Artifact 16: Sensitivity Analysis (Cross-Distribution)
    # --------------------------------------------------------------------------
    sens_data = {
        "distributions_evaluated": cross_sens.distributions_evaluated,
        "global_r_fusion_by_distribution": cross_sens.global_r_fusion_by_dist,
        "global_dcri_by_distribution": cross_sens.global_dcri_by_dist,
        "global_entropy_by_distribution": cross_sens.global_entropy_by_dist,
        "global_u_sum_by_distribution": cross_sens.global_u_sum_by_dist,
        "global_delta_max_by_distribution": cross_sens.global_delta_max_by_dist,
        "tail_sensitivity_by_distribution": cross_sens.tail_sensitivity_by_dist,
        "tail_std_r_fusion_by_distribution": cross_sens.tail_std_r_fusion_by_dist,
    }
    f_sens = output_dir / "sensitivity_analysis.json"
    with open(f_sens, "w", encoding="utf-8") as f:
        json.dump(sens_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Cryptographic Checksum Manifest (SHA-256)
    # --------------------------------------------------------------------------
    all_artifacts = [
        f_cfg, f_comb_def, f_dist_cfg, f_assign, f_freq, f_ht,
        f_metrics, f_unc, f_conf, f_risk, f_dcri, f_base,
        f_tail, f_boot, f_cal, f_sens,
    ]
    manifest = {
        f.name: compute_sha256(f) for f in all_artifacts
    }
    f_manifest = output_dir / "freeze_manifest.json"
    with open(f_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"All 16 artifacts + freeze_manifest.json successfully generated in {output_dir}.")
    print("Phase C11.9 Execution COMPLETE.")

    return {f.name: f for f in all_artifacts + [f_manifest]}


if __name__ == "__main__":
    run_all_combination_experiments(Path("."))
