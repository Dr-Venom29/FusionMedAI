"""Tabular Architecture Benchmarking Master Orchestrator (Phase C5).

Executes:
1. Frozen C2 Split loading and C4 Preprocessor transformation (119 features).
2. Baseline reference benchmarking (Logistic Regression, Random Forest, XGBoost, LightGBM).
3. Advanced Architecture evaluation (CatBoost Default & Tuned, TabNet Default & Tuned).
4. Bounded validation Hyperparameter Optimization (HPO).
5. Comprehensive metrics, threshold sweeps, calibration, and subgroup analysis.
6. Complexity and runtime cost profiling.
7. Cryptographic SHA-256 artifact manifest generation.
"""

import os
import sys
import json
import time
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
script_dir = str(Path(__file__).resolve().parent)
while script_dir in sys.path:
    sys.path.remove(script_dir)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clinical.modeling.preprocessing import ClinicalPreprocessor
from src.clinical.modeling.models import (
    BaseClinicalModel,
    get_baseline_models,
)
from src.clinical.benchmarking.catboost import CatBoostModel
from src.clinical.benchmarking.tabnet import TabNetModel
from src.clinical.modeling.metrics import (
    compute_classification_metrics,
    compute_threshold_metrics,
    compute_calibration_metrics,
)
from src.clinical.benchmarking.complexity import audit_model_complexity
from src.clinical.benchmarking.subgroup_analysis import evaluate_clinical_subgroups
from src.clinical.benchmarking.hpo import (
    run_hpo_search,
    run_catboost_hpo,
    run_tabnet_hpo,
)
from src.clinical.benchmarking.runtime import get_runtime_output_root


def sha256_file(filepath: Path) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def build_models_suite(
    model_name: str,
    run_hpo: bool,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
) -> tuple[Dict[str, BaseClinicalModel], pd.DataFrame]:
    """Assemble the models to evaluate based on target model and HPO flag."""
    models: Dict[str, BaseClinicalModel] = {}
    hpo_dfs: List[pd.DataFrame] = []

    # Helper baseline models
    all_baselines = get_baseline_models(random_state=random_state)

    if model_name in ["all", "logistic_regression"]:
        models["logistic_regression"] = all_baselines["logistic_regression"]

    if model_name in ["all", "logistic_regression_elasticnet"]:
        models["logistic_regression_elasticnet"] = all_baselines["logistic_regression_elasticnet"]

    if model_name in ["all", "random_forest"]:
        models["random_forest"] = all_baselines["random_forest"]

    if model_name in ["all", "xgboost"]:
        models["xgboost"] = all_baselines["xgboost"]

    if model_name in ["all", "lightgbm"]:
        models["lightgbm"] = all_baselines["lightgbm"]

    # CatBoost handling
    if model_name in ["all", "catboost"]:
        models["catboost_default"] = CatBoostModel(
            iterations=300, depth=6, learning_rate=0.05, l2_leaf_reg=3.0, random_state=random_state, verbose=0
        )
        if run_hpo:
            print("  -> Running bounded validation HPO for CatBoost (15 trials)...")
            cb_best, cb_df = run_catboost_hpo(X_train, y_train, X_val, y_val, n_trials=15, random_state=random_state)
            hpo_dfs.append(cb_df)
            models["catboost_tuned"] = CatBoostModel(
                iterations=cb_best.get("iterations", 300),
                depth=cb_best.get("depth", 6),
                learning_rate=cb_best.get("learning_rate", 0.05),
                l2_leaf_reg=cb_best.get("l2_leaf_reg", 3.0),
                subsample=cb_best.get("subsample", 0.8),
                random_state=random_state,
                verbose=0,
            )

    # TabNet handling
    if model_name in ["all", "tabnet"]:
        models["tabnet_default"] = TabNetModel(
            n_d=16, n_a=16, n_steps=3, gamma=1.3, learning_rate=0.02, max_epochs=40, patience=8, random_state=random_state, verbose=0
        )
        if run_hpo:
            print("  -> Running bounded validation HPO for TabNet (10 trials)...")
            tn_best, tn_df = run_tabnet_hpo(X_train, y_train, X_val, y_val, n_trials=10, random_state=random_state)
            hpo_dfs.append(tn_df)
            models["tabnet_tuned"] = TabNetModel(
                n_d=tn_best.get("n_d", 16),
                n_a=tn_best.get("n_a", 16),
                n_steps=tn_best.get("n_steps", 3),
                gamma=tn_best.get("gamma", 1.3),
                lambda_sparse=tn_best.get("lambda_sparse", 1e-3),
                learning_rate=tn_best.get("learning_rate", 0.02),
                max_epochs=40,
                patience=8,
                random_state=random_state,
                verbose=0,
            )

    df_hpo = pd.concat(hpo_dfs, ignore_index=True) if hpo_dfs else pd.DataFrame()
    return models, df_hpo


def run_benchmark(model_filter: str = "catboost", run_hpo: bool = False) -> Dict[str, Any]:
    """Execute the clinical architecture benchmarking pipeline."""
    print("=" * 75)
    print("FusionMedAI: Clinical Tabular Architecture Benchmarking Pipeline (Phase C5)")
    print("=" * 75)

    # 1. Directory Structure Setup
    splits_dir = REPO_ROOT / "datasets" / "clinical" / "processed" / "splits"
    modeling_base = get_runtime_output_root(REPO_ROOT)
    configs_dir = modeling_base / "model_configs"
    manifests_dir = modeling_base / "manifests"

    for d in [modeling_base, configs_dir, manifests_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Load Frozen C2 Splits
    print("\n[Step 1/8] Loading Frozen C2 Canonical Splits...")
    df_train = pd.read_csv(splits_dir / "train.csv")
    df_val = pd.read_csv(splits_dir / "val.csv")
    df_test = pd.read_csv(splits_dir / "test.csv")

    print(f"  -> Train Split: {len(df_train):,} encounters ({df_train['patient_nbr'].nunique():,} patients)")
    print(f"  -> Val Split:   {len(df_val):,} encounters ({df_val['patient_nbr'].nunique():,} patients)")
    print(f"  -> Test Split:  {len(df_test):,} encounters ({df_test['patient_nbr'].nunique():,} patients)")

    # 3. Fit Preprocessor Strictly on Train
    print("\n[Step 2/8] Transforming Splits via Locked ClinicalPreprocessor (D=119)...")
    preprocessor = ClinicalPreprocessor(scale_numerical=True)
    preprocessor.fit(df_train)
    X_train, y_train, trace_train = preprocessor.transform(df_train)
    X_val, y_val, trace_val = preprocessor.transform(df_val)
    X_test, y_test, trace_test = preprocessor.transform(df_test)

    # 4. Build Model Suite & Selective HPO
    if model_filter == "all" and run_hpo:
        print("\n" + "!" * 75)
        print("  [WARNING] You have requested '--model all --hpo'.")
        print("  This is an expensive experiment that executes multi-trial Optuna searches")
        print("  for both CatBoost and TabNet architectures sequentially.")
        print("!" * 75)

    print(f"\n[Step 3/8] Building Model Suite (filter='{model_filter}', hpo={run_hpo})...")
    benchmark_models, df_hpo_trials = build_models_suite(
        model_name=model_filter,
        run_hpo=run_hpo,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        random_state=42,
    )
    print(f"  -> Models ready for evaluation: {list(benchmark_models.keys())}")

    # 5. Train, Evaluate & Profile Complexity
    print(f"\n[Step 4/8] Training & Evaluating {len(benchmark_models)} Architectures...")
    benchmark_rows = []
    threshold_records = []
    calibration_records = []
    subgroup_records = []
    complexity_records = []

    for name, model in benchmark_models.items():
        print(f"  -> Fitting '{name}'...")
        t_start = time.perf_counter()
        
        # Fit with validation early stopping for CatBoost / TabNet if supported
        if isinstance(model, (CatBoostModel, TabNetModel)):
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            model.fit(X_train, y_train)
        
        fit_duration = time.perf_counter() - t_start

        # Profile Model Complexity
        comp_profile = audit_model_complexity(
            model_name=name,
            model_obj=model,
            fit_time_seconds=fit_duration,
            X_sample=X_val[:1000],
        )
        complexity_records.append(comp_profile)

        # Save Configuration JSON
        cfg_path = configs_dir / f"{name}_config.json"
        with open(cfg_path, "w") as f:
            json.dump(model.get_config(), f, indent=2)

        # Evaluate on Validation
        y_prob_val = model.predict_proba(X_val)[:, 1]
        val_m50 = compute_classification_metrics(y_val, y_prob_val, threshold=0.50, model_name=name)
        val_m20 = compute_classification_metrics(y_val, y_prob_val, threshold=0.20, model_name=name)
        val_calib = compute_calibration_metrics(y_val, y_prob_val, n_bins=10, model_name=name)

        # Evaluate on Test
        y_prob_test = model.predict_proba(X_test)[:, 1]
        test_m50 = compute_classification_metrics(y_test, y_prob_test, threshold=0.50, model_name=name)
        test_m20 = compute_classification_metrics(y_test, y_prob_test, threshold=0.20, model_name=name)
        test_calib = compute_calibration_metrics(y_test, y_prob_test, n_bins=10, model_name=name)

        benchmark_rows.append({
            "model": name,
            "val_roc_auc": val_m50["roc_auc"],
            "val_pr_auc": val_m50["pr_auc"],
            "val_brier_score": val_m50["brier_score"],
            "val_log_loss": val_m50["log_loss"],
            "val_ece": val_calib["expected_calibration_error"],
            "val_sensitivity_th20": val_m20["sensitivity"],
            "val_specificity_th20": val_m20["specificity"],
            "val_ppv_th20": val_m20["ppv"],
            "test_roc_auc": test_m50["roc_auc"],
            "test_pr_auc": test_m50["pr_auc"],
            "test_brier_score": test_m50["brier_score"],
            "test_log_loss": test_m50["log_loss"],
            "test_ece": test_calib["expected_calibration_error"],
            "test_sensitivity_th20": test_m20["sensitivity"],
            "test_specificity_th20": test_m20["specificity"],
            "test_ppv_th20": test_m20["ppv"],
            "train_time_sec": comp_profile["train_time_seconds"],
            "inference_latency_ms_per_1000": comp_profile["inference_latency_ms_per_1000"],
            "parameter_count": comp_profile["parameter_count"],
        })

        # Threshold sweeps
        for th in [0.10, 0.20, 0.30, 0.40, 0.50]:
            r_val = compute_classification_metrics(y_val, y_prob_val, threshold=th, model_name=name)
            r_val["split"] = "val"
            threshold_records.append(r_val)
            r_test = compute_classification_metrics(y_test, y_prob_test, threshold=th, model_name=name)
            r_test["split"] = "test"
            threshold_records.append(r_test)

        # Calibration bin details
        for b in val_calib["bins"]:
            b_copy = b.copy()
            b_copy["model"] = name
            b_copy["split"] = "val"
            calibration_records.append(b_copy)
        for b in test_calib["bins"]:
            b_copy = b.copy()
            b_copy["model"] = name
            b_copy["split"] = "test"
            calibration_records.append(b_copy)

        # Subgroup evaluations
        sub_val = evaluate_clinical_subgroups(df_val, y_val, y_prob_val, model_name=name, split_name="val", threshold=0.20)
        subgroup_records.extend(sub_val)
        sub_test = evaluate_clinical_subgroups(df_test, y_test, y_prob_test, model_name=name, split_name="test", threshold=0.20)
        subgroup_records.extend(sub_test)

        print(f"     [Val]  ROC-AUC: {val_m50['roc_auc']:.4f} | PR-AUC: {val_m50['pr_auc']:.4f} | Brier: {val_m50['brier_score']:.4f} | Train Time: {fit_duration:.2f}s")
        print(f"     [Test] ROC-AUC: {test_m50['roc_auc']:.4f} | PR-AUC: {test_m50['pr_auc']:.4f} | Brier: {test_m50['brier_score']:.4f}")

    # 6. Export CSV Artifacts
    print("\n[Step 5/8] Exporting Tabular Artifacts...")
    df_benchmark = pd.DataFrame(benchmark_rows)
    df_benchmark.to_csv(modeling_base / "benchmark_results.csv", index=False)

    if not df_hpo_trials.empty:
        df_hpo_trials.to_csv(modeling_base / "hyperparameter_results.csv", index=False)
    else:
        pd.DataFrame(columns=["model", "trial_number", "val_pr_auc", "val_roc_auc"]).to_csv(
            modeling_base / "hyperparameter_results.csv", index=False
        )

    pd.DataFrame(calibration_records).to_csv(modeling_base / "calibration_results.csv", index=False)
    pd.DataFrame(threshold_records).to_csv(modeling_base / "threshold_results.csv", index=False)
    pd.DataFrame(subgroup_records).to_csv(modeling_base / "subgroup_results.csv", index=False)
    pd.DataFrame(complexity_records).to_csv(modeling_base / "complexity_results.csv", index=False)

    # 7. Cryptographic Manifest Generation
    print("\n[Step 6/8] Generating Cryptographic Manifest...")
    manifest_artifacts = {}
    csv_artifacts = [
        modeling_base / "benchmark_results.csv",
        modeling_base / "hyperparameter_results.csv",
        modeling_base / "calibration_results.csv",
        modeling_base / "threshold_results.csv",
        modeling_base / "subgroup_results.csv",
        modeling_base / "complexity_results.csv",
    ]
    csv_artifacts.extend(list(configs_dir.glob("*.json")))

    for fpath in sorted(csv_artifacts):
        try:
            rel = fpath.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel = fpath.as_posix()
        manifest_artifacts[rel] = {
            "size_bytes": fpath.stat().st_size,
            "sha256": sha256_file(fpath),
        }

    manifest_payload = {
        "provenance": {
            "pipeline": "clinical_benchmarking",
            "analysis_population": "train",
            "random_seed": 42,
            "train_rows": len(df_train),
            "val_rows": len(df_val),
            "test_rows": len(df_test),
            "feature_dim": 119,
            "n_models_evaluated": len(benchmark_models),
        },
        "artifacts": manifest_artifacts,
    }

    manifest_file = manifests_dir / "benchmarking_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest_payload, f, indent=2)

    try:
        manifest_rel = manifest_file.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        manifest_rel = manifest_file.as_posix()
    print(f"  -> Manifest locked with {len(manifest_artifacts)} artifacts at {manifest_rel}")
    print("\n" + "=" * 75)
    print("Clinical Architecture Benchmarking Complete!")
    print("=" * 75)

    return manifest_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Tabular Architecture Benchmarking")
    parser.add_argument(
        "--model",
        type=str,
        default="catboost",
        choices=[
            "all",
            "catboost",
            "tabnet",
            "xgboost",
            "lightgbm",
            "random_forest",
            "logistic_regression",
            "logistic_regression_elasticnet",
        ],
        help="Target model architecture to benchmark (default: catboost)",
    )
    parser.add_argument(
        "--hpo",
        action="store_true",
        default=False,
        help="Run bounded validation hyperparameter search (expensive operation, default: False)",
    )
    parser.add_argument(
        "--no-hpo",
        action="store_false",
        dest="hpo",
        help="Skip bounded validation hyperparameter search (default)",
    )
    args = parser.parse_args()

    run_benchmark(model_filter=args.model, run_hpo=args.hpo)
