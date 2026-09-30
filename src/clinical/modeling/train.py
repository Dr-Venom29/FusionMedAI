"""End-to-End Training Pipeline for Clinical Baseline Models.

Orchestrates:
1. Loading frozen C2 patient-grouped splits (train, val, test).
2. Fitting ClinicalPreprocessor strictly on train.csv.
3. Transforming train, val, and test partitions without leakage.
4. Training all 4 baseline models (Logistic Regression, Random Forest, XGBoost, LightGBM).
5. Saving prediction CSVs with trace identifiers (encounter_id, patient_nbr, y_true, y_prob).
6. Evaluating metrics across thresholds, calibration, and subgroup error stratification.
7. Exporting structured JSON/CSV metadata artifacts and SHA-256 manifest.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clinical.modeling.preprocessing import ClinicalPreprocessor
from src.clinical.modeling.models import get_baseline_models
from src.clinical.modeling.evaluate import (
    evaluate_model_on_split,
    compute_stratified_error_analysis,
)


def sha256_file(filepath: str) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_training_pipeline() -> Dict[str, Any]:
    """Execute the end-to-end baseline training, prediction, and evaluation workflow."""
    print("=" * 70)
    print("FusionMedAI: Phase C4 - Baseline Model Training & Evaluation Pipeline")
    print("=" * 70)

    # 1. Paths configuration
    splits_dir = REPO_ROOT / "datasets" / "clinical" / "processed" / "splits"
    out_base = REPO_ROOT / "datasets" / "clinical" / "metadata" / "modeling"
    configs_dir = out_base / "model_configs"
    metrics_dir = out_base / "metrics"
    preds_dir = out_base / "predictions"
    reports_dir = out_base / "reports"

    for d in [configs_dir, metrics_dir, preds_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Load splits
    print("\n[Step 1/7] Loading Frozen C2 Splits...")
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    print(f"  -> Train Split: {len(df_train):,} rows ({df_train['patient_nbr'].nunique():,} patients)")
    print(f"  -> Val Split:   {len(df_val):,} rows ({df_val['patient_nbr'].nunique():,} patients)")
    print(f"  -> Test Split:  {len(df_test):,} rows ({df_test['patient_nbr'].nunique():,} patients)")

    # 3. Fit Preprocessor strictly on train
    print("\n[Step 2/7] Fitting ClinicalPreprocessor Strictly on Train...")
    preprocessor = ClinicalPreprocessor(scale_numerical=True)
    preprocessor.fit(df_train)
    print(f"  -> Preprocessor fitted successfully. Total Feature Dimensions: {preprocessor.feature_dim_}")

    # 4. Transform all splits
    print("\n[Step 3/7] Transforming Splits using Locked Preprocessor...")
    X_train, y_train, trace_train = preprocessor.transform(df_train)
    X_val, y_val, trace_val = preprocessor.transform(df_val)
    X_test, y_test, trace_test = preprocessor.transform(df_test)

    print(f"  -> X_train shape: {X_train.shape} | y_train shape: {y_train.shape} | pos: {np.mean(y_train):.4%}")
    print(f"  -> X_val shape:   {X_val.shape}   | y_val shape:   {y_val.shape}   | pos: {np.mean(y_val):.4%}")
    print(f"  -> X_test shape:  {X_test.shape}  | y_test shape:  {y_test.shape}  | pos: {np.mean(y_test):.4%}")

    # Save Preprocessing Metadata
    prep_meta = {
        "provenance": {
            "analysis_population": "train",
            "train_rows": len(df_train),
            "train_patients": int(df_train["patient_nbr"].nunique()),
            "feature_dim": preprocessor.feature_dim_,
            "feature_names": preprocessor.feature_names_,
        },
        "transformations": {
            "scale_numerical": True,
            "demographics_ohe": ["race", "gender"],
            "demographics_ordinal": ["age"],
            "context_ohe": ["admission_type_id", "admission_source_id", "medical_specialty", "payer_code"],
            "glycemic_labs_ordinal": ["max_glu_serum", "A1Cresult"],
            "diagnoses_icd9_chapters": ["diag_1", "diag_2", "diag_3"],
            "medications_exposure_4level": preprocessor.train_provenance_["top_specialties"],
            "treatment_dynamics_binary": ["change", "diabetesMed"],
        },
    }
    prep_meta_file = out_base / "preprocessing_metadata.json"
    with open(prep_meta_file, "w") as f:
        json.dump(prep_meta, f, indent=2)
    print(f"  -> Saved {prep_meta_file.relative_to(REPO_ROOT)}")

    # 5. Train all baseline models
    print("\n[Step 4/7] Training Baseline Models...")
    models = get_baseline_models(random_state=42)
    
    val_eval_results = {}
    test_eval_results = {}
    threshold_records = []
    calibration_records = []

    for name, model in models.items():
        print(f"  -> Training '{name}'...")
        model.fit(X_train, y_train)

        # Save Model Config
        cfg_file = configs_dir / f"{name}_config.json"
        with open(cfg_file, "w") as f:
            json.dump(model.get_config(), f, indent=2)

        # Predictions on Validation
        y_prob_val = model.predict_proba(X_val)[:, 1]
        df_pred_val = trace_val.copy()
        df_pred_val["y_true"] = y_val
        df_pred_val["y_prob"] = y_prob_val
        val_pred_file = preds_dir / f"{name}_val.csv"
        df_pred_val.to_csv(val_pred_file, index=False)

        # Predictions on Test
        y_prob_test = model.predict_proba(X_test)[:, 1]
        df_pred_test = trace_test.copy()
        df_pred_test["y_true"] = y_test
        df_pred_test["y_prob"] = y_prob_test
        test_pred_file = preds_dir / f"{name}_test.csv"
        df_pred_test.to_csv(test_pred_file, index=False)

        # Evaluate on Validation
        val_res = evaluate_model_on_split(model, X_val, y_val, model_name=name)
        val_eval_results[name] = val_res

        # Evaluate on Test
        test_res = evaluate_model_on_split(model, X_test, y_test, model_name=name)
        test_eval_results[name] = test_res

        # Record threshold metrics
        for row in val_res["threshold_evaluations"]:
            rec = row.copy()
            rec["split"] = "val"
            threshold_records.append(rec)
        for row in test_res["threshold_evaluations"]:
            rec = row.copy()
            rec["split"] = "test"
            threshold_records.append(rec)

        # Record calibration bin metrics
        for b in val_res["calibration"]["bins"]:
            brec = b.copy()
            brec["model"] = name
            brec["split"] = "val"
            calibration_records.append(brec)
        for b in test_res["calibration"]["bins"]:
            brec = b.copy()
            brec["model"] = name
            brec["split"] = "test"
            calibration_records.append(brec)

        print(f"     [Val]  ROC-AUC: {val_res['summary_default_th50']['roc_auc']:.4f} | PR-AUC: {val_res['summary_default_th50']['pr_auc']:.4f} | Brier: {val_res['summary_default_th50']['brier_score']:.4f}")
        print(f"     [Test] ROC-AUC: {test_res['summary_default_th50']['roc_auc']:.4f} | PR-AUC: {test_res['summary_default_th50']['pr_auc']:.4f} | Brier: {test_res['summary_default_th50']['brier_score']:.4f}")

    # 6. Save Metrics and Reports
    print("\n[Step 5/7] Saving Metrics and Aggregated Reports...")
    with open(metrics_dir / "validation_metrics.json", "w") as f:
        json.dump(val_eval_results, f, indent=2)
    with open(metrics_dir / "test_metrics.json", "w") as f:
        json.dump(test_eval_results, f, indent=2)

    df_thresholds = pd.DataFrame(threshold_records)
    df_thresholds.to_csv(metrics_dir / "threshold_metrics.csv", index=False)

    df_calib = pd.DataFrame(calibration_records)
    df_calib.to_csv(metrics_dir / "calibration_metrics.csv", index=False)

    # 7. Stratified Error Analysis (on top LightGBM / XGBoost models)
    print("\n[Step 6/7] Computing Stratified Subgroup Error Analysis...")
    error_analysis_payload = {}
    for name in ["logistic_regression", "random_forest", "xgboost", "lightgbm"]:
        y_prob_val = val_eval_results[name]["threshold_evaluations"][1]  # th=0.20
        prob_vals = pd.read_csv(preds_dir / f"{name}_val.csv")["y_prob"].values
        err_res = compute_stratified_error_analysis(
            df_raw=df_val,
            y_true=y_val,
            y_prob=prob_vals,
            threshold=0.20,
            model_name=name,
        )
        error_analysis_payload[name] = err_res

    with open(reports_dir / "error_analysis_report.json", "w") as f:
        json.dump(error_analysis_payload, f, indent=2)

    # Baseline Comparison Summary Report
    comp_rows = []
    for name in models.keys():
        v = val_eval_results[name]["summary_default_th50"]
        t = test_eval_results[name]["summary_default_th50"]
        comp_rows.append({
            "model": name,
            "val_roc_auc": v["roc_auc"],
            "val_pr_auc": v["pr_auc"],
            "val_brier_score": v["brier_score"],
            "val_log_loss": v["log_loss"],
            "test_roc_auc": t["roc_auc"],
            "test_pr_auc": t["pr_auc"],
            "test_brier_score": t["brier_score"],
            "test_log_loss": t["log_loss"],
        })

    baseline_comparison_report = {
        "provenance": {
            "analysis_population": "train",
            "train_rows": len(df_train),
            "val_rows": len(df_val),
            "test_rows": len(df_test),
            "random_seed": 42,
        },
        "comparison_table": comp_rows,
    }
    with open(reports_dir / "baseline_comparison_report.json", "w") as f:
        json.dump(baseline_comparison_report, f, indent=2)

    # 8. Cryptographic Manifest Freeze
    print("\n[Step 7/7] Generating Cryptographic Manifest...")
    manifest_artifacts = {}
    
    # Collect all generated files
    all_generated_files = [
        out_base / "preprocessing_metadata.json",
        reports_dir / "error_analysis_report.json",
        reports_dir / "baseline_comparison_report.json",
        metrics_dir / "validation_metrics.json",
        metrics_dir / "test_metrics.json",
        metrics_dir / "threshold_metrics.csv",
        metrics_dir / "calibration_metrics.csv",
    ]
    all_generated_files.extend(list(configs_dir.glob("*.json")))
    all_generated_files.extend(list(preds_dir.glob("*.csv")))

    for fpath in sorted(all_generated_files):
        rel = fpath.relative_to(REPO_ROOT).as_posix()
        manifest_artifacts[rel] = {
            "size_bytes": fpath.stat().st_size,
            "sha256": sha256_file(str(fpath)),
        }

    manifest_payload = {
        "provenance": {
            "analysis_population": "train",
            "random_seed": 42,
            "n_models_trained": len(models),
        },
        "artifacts": manifest_artifacts,
    }
    manifest_file = reports_dir / "modeling_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest_payload, f, indent=2)

    print(f"  -> Manifest locked with {len(manifest_artifacts)} artifacts at {manifest_file.relative_to(REPO_ROOT)}")
    print("\n" + "=" * 70)
    print("Phase C4 Baseline Pipeline Execution Complete!")
    print("=" * 70)

    return baseline_comparison_report


if __name__ == "__main__":
    run_training_pipeline()
