"""Subgroup Performance Evaluation for Tabular Architecture Benchmarking.

Audits performance heterogeneity across:
- Age bands
- Race categories
- Gender
- Primary diagnosis ICD-9 chapters
- Prior inpatient utilization (0, 1, 2, >=3)
- Insulin exposure state
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from src.clinical.modeling.schema import map_icd9_to_chapter


def evaluate_clinical_subgroups(
    df_raw: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    split_name: str = "val",
    threshold: float = 0.20,
) -> List[Dict[str, Any]]:
    """Compute subgroup-stratified metrics for a given model's predictions."""
    y_pred = (y_prob >= threshold).astype(int)
    df_eval = df_raw.copy()
    df_eval["y_true"] = y_true
    df_eval["y_prob"] = y_prob
    df_eval["y_pred"] = y_pred
    df_eval["primary_diag_chapter"] = df_eval["diag_1"].apply(map_icd9_to_chapter)
    df_eval["inpatient_prior_group"] = df_eval["number_inpatient"].apply(
        lambda x: "0" if x == 0 else ("1" if x == 1 else ("2" if x == 2 else ">=3"))
    )

    dimensions = {
        "age": "age",
        "race": "race",
        "gender": "gender",
        "primary_diag_chapter": "primary_diag_chapter",
        "inpatient_prior_group": "inpatient_prior_group",
        "insulin": "insulin",
    }

    subgroup_records = []
    for dim_name, col in dimensions.items():
        if col not in df_eval.columns:
            continue
        for group_val, grp in df_eval.groupby(col, observed=True):
            n_grp = len(grp)
            n_pos = int(grp["y_true"].sum())
            if n_grp == 0:
                continue

            y_t = grp["y_true"].values
            y_p = grp["y_prob"].values
            y_hat = grp["y_pred"].values

            tp = int(np.sum((y_t == 1) & (y_hat == 1)))
            fp = int(np.sum((y_t == 0) & (y_hat == 1)))
            tn = int(np.sum((y_t == 0) & (y_hat == 0)))
            fn = int(np.sum((y_t == 1) & (y_hat == 0)))

            sens = round(tp / (tp + fn), 4) if (tp + fn) > 0 else np.nan
            spec = round(tn / (tn + fp), 4) if (tn + fp) > 0 else np.nan
            ppv = round(tp / (tp + fp), 4) if (tp + fp) > 0 else np.nan
            npv = round(tn / (tn + fn), 4) if (tn + fn) > 0 else np.nan

            # Subgroup discrimination where both classes exist
            if len(np.unique(y_t)) > 1:
                sub_roc = round(float(roc_auc_score(y_t, y_p)), 4)
                sub_pr = round(float(average_precision_score(y_t, y_p)), 4)
            else:
                # When only one class is present in the subgroup, ROC-AUC is mathematically undefined
                sub_roc = np.nan
                sub_pr = np.nan

            subgroup_records.append({
                "model": model_name,
                "split": split_name,
                "dimension": dim_name,
                "subgroup": str(group_val),
                "n_samples": n_grp,
                "n_positive": n_pos,
                "prevalence": round(n_pos / n_grp, 4),
                "roc_auc": sub_roc,
                "pr_auc": sub_pr,
                "sensitivity_th20": sens,
                "specificity_th20": spec,
                "ppv_th20": ppv,
                "npv_th20": npv,
                "fp_count": fp,
                "fn_count": fn,
            })

    return subgroup_records


def run_subgroup_analysis_standalone(
    model_name: str = "catboost",
    threshold: float = 0.20,
) -> pd.DataFrame:
    """Run clinical subgroup analysis for a selected model."""
    import argparse
    from pathlib import Path
    import pandas as pd
    from src.clinical.modeling.models import get_baseline_models
    from src.clinical.modeling.preprocessing import ClinicalPreprocessor
    from src.clinical.benchmarking.catboost import CatBoostModel
    from src.clinical.benchmarking.tabnet import TabNetModel

    print("=" * 70)
    print(f"FusionMedAI: Clinical Subgroup Stratified Analysis ({model_name.upper()})")
    print("=" * 70)

    repo_root = Path(__file__).resolve().parents[3]
    splits_dir = repo_root / "datasets" / "clinical" / "processed" / "splits"
    c5_base = repo_root / "datasets" / "clinical" / "metadata" / "modeling" / "c5"
    c5_base.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Loading canonical splits & fitting preprocessor...")
    df_train = pd.read_csv(splits_dir / "train.csv")
    df_val = pd.read_csv(splits_dir / "val.csv")
    df_test = pd.read_csv(splits_dir / "test.csv")

    preprocessor = ClinicalPreprocessor(scale_numerical=True)
    preprocessor.fit(df_train)
    X_train, y_train, _ = preprocessor.transform(df_train)
    X_val, y_val, _ = preprocessor.transform(df_val)
    X_test, y_test, _ = preprocessor.transform(df_test)

    baselines = get_baseline_models(random_state=42)
    if model_name == "catboost":
        model = CatBoostModel(iterations=300, depth=6, random_state=42, verbose=0)
    elif model_name == "tabnet":
        model = TabNetModel(max_epochs=40, patience=8, random_state=42, verbose=0)
    elif model_name in baselines:
        model = baselines[model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"\n[2/3] Fitting '{model_name}' and evaluating subgroups across Val & Test...")
    if isinstance(model, (CatBoostModel, TabNetModel)):
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
    else:
        model.fit(X_train, y_train)

    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    sub_val = evaluate_clinical_subgroups(df_val, y_val, y_prob_val, model_name=model_name, split_name="val", threshold=threshold)
    sub_test = evaluate_clinical_subgroups(df_test, y_test, y_prob_test, model_name=model_name, split_name="test", threshold=threshold)

    all_sub = sub_val + sub_test
    df_sub = pd.DataFrame(all_sub)
    out_path = c5_base / "subgroup_results.csv"
    df_sub.to_csv(out_path, index=False)

    print(f"\n[3/3] Subgroup analysis ({len(df_sub)} rows) saved to: {out_path.relative_to(repo_root)}")
    return df_sub


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standalone Clinical Subgroup Analysis Runner")
    parser.add_argument(
        "--model",
        type=str,
        default="catboost",
        choices=["catboost", "tabnet", "xgboost", "lightgbm", "random_forest", "logistic_regression"],
        help="Target architecture to evaluate (default: catboost)",
    )
    parser.add_argument("--threshold", type=float, default=0.20, help="Classification threshold (default: 0.20)")
    args = parser.parse_args()

    run_subgroup_analysis_standalone(
        model_name=args.model,
        threshold=args.threshold,
    )
