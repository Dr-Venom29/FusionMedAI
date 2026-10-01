"""Clinical feature group taxonomy mapping and aggregate SHAP analysis."""

from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def get_clinical_feature_group(feature_name: str) -> str:
    """Map a 119-dimensional feature name to its clinical taxonomy group."""
    if feature_name.startswith("num_") or feature_name in ["time_in_hospital", "number_diagnoses"]:
        return "Acute Clinical & Encounter Complexity"
    elif feature_name.startswith("number_inpatient") or feature_name.startswith("number_emergency") or feature_name.startswith("number_outpatient"):
        return "Prior Healthcare Utilization"
    elif feature_name.startswith("race_") or feature_name.startswith("gender_"):
        return "Demographics"
    elif feature_name.startswith("age") or feature_name.startswith("max_glu_serum") or feature_name.startswith("A1Cresult"):
        return "Age & Glycemic Monitoring"
    elif feature_name.startswith("diag_1_") or feature_name.startswith("diag_2_") or feature_name.startswith("diag_3_"):
        return "ICD-9 Diagnosis Chapters"
    elif any(feature_name.startswith(m) for m in [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone"
    ]):
        return "Diabetic Medications"
    elif feature_name.startswith("admission_type_id_") or feature_name.startswith("admission_source_id_") or feature_name.startswith("discharge_disposition_id_") or feature_name.startswith("medical_specialty_") or feature_name.startswith("payer_code_"):
        return "Encounter Context & Administrative"
    elif feature_name.startswith("change_") or feature_name.startswith("diabetesMed_"):
        return "Treatment Dynamics"
    else:
        return "Other Clinical Encoded"


def compute_group_importance(
    importance_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate individual feature SHAP importances into clinical taxonomy groups."""
    group_df = importance_df.groupby("feature_group").agg(
        feature_count=("feature", "count"),
        total_mean_abs_shap=("mean_abs_shap_test", "sum"),
        avg_mean_abs_shap=("mean_abs_shap_test", "mean"),
    ).reset_index()

    total_sum = group_df["total_mean_abs_shap"].sum()
    group_df["relative_contribution_pct"] = (group_df["total_mean_abs_shap"] / total_sum) * 100.0
    group_df = group_df.sort_values(by="total_mean_abs_shap", ascending=False).reset_index(drop=True)
    group_df["group_rank"] = np.arange(1, len(group_df) + 1)
    return group_df


def plot_group_importance(
    group_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot horizontal bar chart of clinical feature group contributions."""
    group_sorted = group_df.sort_values("relative_contribution_pct", ascending=True)
    plt.figure(figsize=(10, 6), dpi=300)
    palette = sns.color_palette("viridis", len(group_sorted))
    plt.barh(group_sorted["feature_group"], group_sorted["relative_contribution_pct"], color=palette, edgecolor="black", alpha=0.85)

    for i, v in enumerate(group_sorted["relative_contribution_pct"]):
        count = group_sorted["feature_count"].iloc[i]
        plt.text(v + 0.5, i, f"{v:.1f}% ({count} feats)", va="center", fontsize=9, fontweight="bold")

    plt.xlabel("Aggregate Relative SHAP Contribution (%)", fontsize=12, fontweight="bold")
    plt.title("Clinical Feature Group Relative Importance (D=119)", fontsize=14, fontweight="bold", pad=15)
    plt.xlim(0, max(group_sorted["relative_contribution_pct"]) + 8)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
