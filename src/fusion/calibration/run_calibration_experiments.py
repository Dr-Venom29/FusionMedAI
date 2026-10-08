"""
src/fusion/calibration/run_calibration_experiments.py
Phase C11.11: Modality Calibration Impact on Decision-Level Fusion Experiment Runner
"""

import sys
from pathlib import Path
import json
import hashlib
import numpy as np

# Ensure root is in path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.fusion.calibration.calibration_runner import CalibrationExperimentRunner
from src.fusion.calibration.paired_calibration_analysis import (
    compute_paired_bootstrap_cis,
    evaluate_calibration_hypotheses,
)
from src.fusion.calibration.calibration_condition import (
    extract_modality_calibration_state,
)


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("================================================================================")
    print("PHASE C11.11: MODALITY CALIBRATION IMPACT ON DECISION-LEVEL FUSION")
    print("================================================================================\n")

    root_dir = Path(__file__).resolve().parents[3]
    output_dir = root_dir / "experiments" / "fusion" / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize runner
    runner = CalibrationExperimentRunner(
        alpha=1.0,
        beta=1.5,
        gamma=1.0,
        eta=0.5,
        delta=0.20,
        seed=115,
    )

    # 2. Config export
    config = {
        "phase": "C11.11",
        "experiment_title": "Modality Calibration Impact on Decision-Level Fusion",
        "sample_size": 500,
        "seed": 115,
        "router_coefficients": {
            "alpha": 1.0,
            "beta": 1.5,
            "gamma": 1.0,
            "eta": 0.5,
        },
        "dcri_delta": 0.20,
        "bootstrap_resamples": 1000,
        "bootstrap_confidence_level": 0.95,
        "frozen_modality_calibrators": {
            "retina": "Temperature Scaling (T=1.6218)",
            "foot": "Vector Scaling (W=[1.0410, 1.0430, 0.8711, 1.1301], b=[0.0292, 0.0625, 0.0630, -0.1547])",
            "clinical": "Isotonic Regression / Platt Mapping",
        },
        "experimental_conditions": {
            "B0": "Uncalibrated Uniform Fusion (w_i = 1/M)",
            "B1": "Uncalibrated Reliability-Selected Fusion",
            "B2": "Uncalibrated ACARA-U Dynamic Fusion",
            "B3": "Calibrated Uniform Fusion (w_i = 1/M)",
            "B4": "Calibrated Reliability-Selected Fusion",
            "B5": "Calibrated ACARA-U Dynamic Fusion",
        },
        "retina_calibration_fit_split": "validation",
        "foot_calibration_fit_split": "validation",
        "clinical_calibration_fit_split": "validation",
        "evaluation_cohort_type": "controlled_synthetic_decision_packets",
        "calibration_fit_before_evaluation": True,
        "no_evaluation_data_leakage": True,
        "methodological_boundary": (
            "Modality-level calibration metrics are computed on single-modality validation splits. "
            "Decision-level fused risk R_fusion and DCRI are derived decision indices; no clinical "
            "multimodal ground truth is fabricated for fusion-level ECE/Brier."
        ),
    }

    config_path = output_dir / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"1. Exported experiment configuration -> {config_path.name}")

    # 3. Load cohort
    cohort = runner.load_cohort()
    print(f"2. Loaded frozen cohort of N={len(cohort)} decision packets (seed=115)")

    # 4. Modality calibration profiles
    profiles = runner.load_modality_calibration_profiles()
    mod_results_path = output_dir / "modality_calibration_results.json"
    with open(mod_results_path, "w") as f:
        json.dump({k: v.to_dict() for k, v in profiles.items()}, f, indent=2)
    print(f"3. Exported modality calibration profiles -> {mod_results_path.name}")

    # 5. Extract distributions across cohort
    cohort_distributions = {
        "retina": {"raw_risks": [], "cal_risks": [], "delta_risks": [], "raw_max_p": [], "cal_max_p": []},
        "foot": {"raw_risks": [], "cal_risks": [], "delta_risks": [], "raw_max_p": [], "cal_max_p": []},
        "clinical": {"raw_risks": [], "cal_risks": [], "delta_risks": [], "raw_max_p": [], "cal_max_p": []},
    }
    for pkt in cohort:
        for mod, rec in pkt.records.items():
            state = extract_modality_calibration_state(rec, propagate_confidence=True)
            cohort_distributions[mod]["raw_risks"].append(float(state.risk_raw))
            cohort_distributions[mod]["cal_risks"].append(float(state.risk_cal))
            cohort_distributions[mod]["delta_risks"].append(float(state.delta_risk))
            cohort_distributions[mod]["raw_max_p"].append(float(state.confidence_raw))
            cohort_distributions[mod]["cal_max_p"].append(float(state.confidence_cal))

    dist_summary = {}
    for mod, data in cohort_distributions.items():
        dist_summary[mod] = {
            "mean_risk_raw": round(float(np.mean(data["raw_risks"])), 6),
            "mean_risk_calibrated": round(float(np.mean(data["cal_risks"])), 6),
            "mean_delta_risk": round(float(np.mean(data["delta_risks"])), 6),
            "std_delta_risk": round(float(np.std(data["delta_risks"])), 6),
            "mean_confidence_raw": round(float(np.mean(data["raw_max_p"])), 6),
            "mean_confidence_calibrated": round(float(np.mean(data["cal_max_p"])), 6),
        }
    dist_path = output_dir / "calibration_distributions.json"
    with open(dist_path, "w") as f:
        json.dump(dist_summary, f, indent=2)
    print(f"4. Exported cohort probability distributions -> {dist_path.name}")

    # 6. Run Clean Comparison (Experiment A)
    print("5. Running Clean Comparison (Experiment A) across B0–B5 conditions...")
    clean_res = runner.run_clean_comparison(cohort, propagate_confidence=True)
    clean_path = output_dir / "clean_comparison.json"
    with open(clean_path, "w") as f:
        json.dump(clean_res, f, indent=2)
    print(f"   Exported clean comparison results -> {clean_path.name}")

    # 7. Run Degradation Comparison (Experiment B)
    print("6. Running Degradation Comparison (Experiment B) across D0–D3 ladders...")
    deg_res = runner.run_degradation_comparison(cohort, propagate_confidence=True)
    deg_path = output_dir / "degradation_comparison.json"
    with open(deg_path, "w") as f:
        json.dump(deg_res, f, indent=2)
    print(f"   Exported degradation comparison results -> {deg_path.name}")

    # 8. Run 1,000-resample paired bootstrap
    print("7. Computing 1,000-resample paired bootstrap confidence intervals (seed=115)...")
    paired_cis = compute_paired_bootstrap_cis(clean_res, n_bootstraps=1000, seed=115)
    boot_path = output_dir / "paired_bootstrap.json"
    with open(boot_path, "w") as f:
        json.dump({k: v.to_dict() for k, v in paired_cis.items()}, f, indent=2)
    print(f"   Exported paired bootstrap results -> {boot_path.name}")

    # 9. Evaluate Hypotheses H1–H6
    print("8. Evaluating Hypotheses H1–H6...")
    hypotheses = evaluate_calibration_hypotheses(profiles, paired_cis, deg_res)
    hyp_path = output_dir / "hypothesis_results.json"
    with open(hyp_path, "w") as f:
        json.dump(hypotheses, f, indent=2)
    print(f"   Exported hypothesis outcomes -> {hyp_path.name}")

    # 10. Generate Freeze Manifest
    print("9. Generating Cryptographic Freeze Manifest (SHA-256)...")
    manifest_files = [
        "experiment_config.json",
        "modality_calibration_results.json",
        "calibration_distributions.json",
        "clean_comparison.json",
        "degradation_comparison.json",
        "paired_bootstrap.json",
        "hypothesis_results.json",
    ]
    manifest = {
        "phase": "C11.11",
        "title": "Modality Calibration Impact on Decision-Level Fusion",
        "total_artifacts": len(manifest_files),
        "artifacts": {},
    }
    for fname in manifest_files:
        fpath = output_dir / fname
        if fpath.exists():
            manifest["artifacts"][fname] = {
                "sha256": compute_sha256(fpath),
                "size_bytes": fpath.stat().st_size,
            }

    manifest_path = output_dir / "freeze_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"   Exported freeze manifest -> {manifest_path.name}")

    print("\n================================================================================")
    print("PHASE C11.11 EXPERIMENT RUN COMPLETE: ALL 8 ARTIFACTS GENERATED")
    print("================================================================================")


if __name__ == "__main__":
    main()
