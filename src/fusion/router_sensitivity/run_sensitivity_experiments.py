"""
src/fusion/router_sensitivity/run_sensitivity_experiments.py
Phase C11.12: ACARA-U Parameter & Weighting Sensitivity Analysis Runner
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

from src.fusion.router_sensitivity.sensitivity_runner import SensitivityExperimentRunner
from src.fusion.router_sensitivity.parameter_grid import get_sensitivity_parameter_grid, get_reference_config
from src.fusion.router_sensitivity.sensitivity_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    DELTA_PROVISIONAL,
    SEED,
    N_PACKETS,
    N_BOOTSTRAPS,
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
    print("PHASE C11.12: ACARA-U PARAMETER & WEIGHTING SENSITIVITY ANALYSIS")
    print("================================================================================\n")

    root_dir = Path(__file__).resolve().parents[3]
    output_dir = root_dir / "experiments" / "fusion" / "router_sensitivity" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export Experiment Configuration
    config = {
        "phase": "C11.12",
        "title": "ACARA-U Parameter & Weighting Sensitivity Analysis",
        "sample_size": N_PACKETS,
        "seed": SEED,
        "frozen_reference_router": {
            "alpha": ALPHA_REF,
            "beta": BETA_REF,
            "gamma": GAMMA_REF,
            "eta": ETA_REF,
        },
        "dcri_provisional_delta": DELTA_PROVISIONAL,
        "bootstrap_resamples": N_BOOTSTRAPS,
        "bootstrap_confidence_level": 0.95,
        "purpose": (
            "Determine whether the frozen ACARA-U weighting configuration is behaviorally stable "
            "under local perturbations and quantify directional parameter sensitivity. "
            "Explicitly NOT hyperparameter optimization or post-hoc coefficient re-tuning."
        ),
        "methodological_boundary": (
            "Evaluations operate on the frozen N=500 controlled decision packet cohort. "
            "R_fusion and DCRI are derived decision indices; no clinical multimodal ground truth is fabricated."
        ),
    }

    config_path = output_dir / "sensitivity_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"1. Exported experiment configuration -> {config_path.name}")

    # 2. Export Parameter Grid
    grid = get_sensitivity_parameter_grid()
    grid_dict = {item.config_id: item.to_dict() for item in grid}
    grid_path = output_dir / "parameter_grid.json"
    with open(grid_path, "w", encoding="utf-8") as f:
        json.dump(grid_dict, f, indent=2)
    print(f"2. Exported parameter grid ({len(grid)} configurations) -> {grid_path.name}")

    # 3. Initialize Runner & Load Cohort
    runner = SensitivityExperimentRunner(repo_root=root_dir, delta=DELTA_PROVISIONAL, seed=SEED, n_bootstraps=N_BOOTSTRAPS)
    cohort = runner.load_cohort()
    print(f"3. Loaded frozen cohort of N={len(cohort)} decision packets (seed={SEED})")

    # 4. Run Full Grid Analysis
    print(f"4. Evaluating {len(grid)} prespecified configuration IDs (19 unique coefficient settings) across tri-modal cohort...")
    grid_analysis = runner.run_full_grid_analysis(cohort)

    # Export sensitivity results (summaries + slopes + aggregate sensitivities)
    eval_summaries = {k: v["summary"] for k, v in grid_analysis["evaluations"].items()}
    results_payload = {
        "reference_summary": grid_analysis["reference_summary"],
        "evaluations": eval_summaries,
        "sensitivity_slopes": grid_analysis["sensitivity_slopes"],
        "aggregate_sensitivities": grid_analysis["aggregate_sensitivities"],
    }
    results_path = output_dir / "sensitivity_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"   Exported full grid results -> {results_path.name}")

    # Export Reference Comparison Deltas
    ref_comp_path = output_dir / "reference_comparison.json"
    with open(ref_comp_path, "w", encoding="utf-8") as f:
        json.dump(grid_analysis["reference_deltas"], f, indent=2)
    print(f"   Exported reference delta comparisons -> {ref_comp_path.name}")

    # 5. Run Regime Analysis (7 valid combinations + empty)
    print("5. Evaluating 7 availability regimes across all configurations...")
    regime_results = runner.run_regime_analysis(cohort)
    regime_path = output_dir / "regime_results.json"
    with open(regime_path, "w", encoding="utf-8") as f:
        json.dump(regime_results, f, indent=2)
    print(f"   Exported availability regime results -> {regime_path.name}")

    # 6. Run Paired Bootstrap (B=1,000, seed=115)
    print("6. Computing 1,000-resample paired bootstrap confidence intervals...")
    bootstrap_results = runner.run_paired_bootstraps(grid_analysis)
    bootstrap_path = output_dir / "bootstrap_results.json"
    with open(bootstrap_path, "w", encoding="utf-8") as f:
        json.dump(bootstrap_results, f, indent=2)
    print(f"   Exported paired bootstrap results -> {bootstrap_path.name}")

    # 7. Run Mask Invariance Tests
    print("7. Running strong masked-value invariance tests across all configurations...")
    mask_invariance = runner.run_mask_invariance_tests(cohort)

    # 8. Evaluate Hypotheses H1–H8
    print("8. Evaluating Hypotheses H1–H8...")
    hypotheses = runner.evaluate_hypotheses(grid_analysis, bootstrap_results, mask_invariance, cohort)
    hyp_path = output_dir / "hypothesis_results.json"
    with open(hyp_path, "w", encoding="utf-8") as f:
        json.dump(hypotheses, f, indent=2)
    print(f"   Exported hypothesis outcomes -> {hyp_path.name}")

    # 9. Generate Cryptographic Freeze Manifest (SHA-256)
    print("9. Generating Cryptographic Freeze Manifest (SHA-256)...")
    manifest_files = [
        "sensitivity_config.json",
        "parameter_grid.json",
        "sensitivity_results.json",
        "reference_comparison.json",
        "regime_results.json",
        "bootstrap_results.json",
        "hypothesis_results.json",
    ]
    manifest = {
        "phase": "C11.12",
        "title": "ACARA-U Parameter & Weighting Sensitivity Analysis",
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

    manifest_path = output_dir / "sensitivity_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"   Exported freeze manifest -> {manifest_path.name}")

    print("\n================================================================================")
    print("PHASE C11.12 EXPERIMENT RUN COMPLETE: ALL 8 ARTIFACTS GENERATED")
    print("================================================================================")


if __name__ == "__main__":
    main()
