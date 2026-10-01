"""
Publication Figure Generation for Clinical Phase C9: Robustness & Distribution Shift.
Generates 7 publication-standard (300 DPI) figures.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_missingness_degradation(
    df_missingness: pd.DataFrame,
    save_path: Path,
):
    """Plot performance degradation and uncertainty inflation vs missingness scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Filter random MCAR scenarios for continuous curve
    mcar_scenarios = df_missingness[df_missingness["scenario"].str.startswith("M") & ~df_missingness["scenario"].str.contains("Targeted")].copy()
    labels = ["Nominal (0%)", "+10% Missing", "+25% Missing", "+50% Missing"]
    x = np.arange(len(mcar_scenarios))

    # Panel A: Discrimination & Calibration Degradation
    axes[0].plot(x, mcar_scenarios["roc_auc"], marker="o", color="#1f77b4", linewidth=2.2, label="ROC-AUC")
    axes[0].plot(x, mcar_scenarios["pr_auc"] * 3.0, marker="s", color="#ff7f0e", linewidth=2.0, linestyle="--", label="PR-AUC (scaled x3)")
    axes[0].plot(x, mcar_scenarios["calibration_slope"], marker="^", color="#2ca02c", linewidth=2.0, linestyle=":", label="Calibration Slope")

    axes[0].set_title("A. Metric Degradation Under Random Missingness", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("Metric Value", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower left", fontsize=9)

    # Panel B: Uncertainty Inflation
    axes[1].plot(x, mcar_scenarios["mean_uncertainty"] * 100, marker="o", color="#d62728", linewidth=2.2, label="Mean Uncertainty (σ_p x100)")
    axes[1].plot(x, mcar_scenarios["p95_uncertainty"] * 100, marker="^", color="#9467bd", linewidth=2.0, linestyle="--", label="95th Percentile Uncertainty (x100)")
    axes[1].plot(x, mcar_scenarios["error_rate"] * 100, marker="d", color="#8c564b", linewidth=2.0, linestyle=":", label="Error Rate (%)")

    axes[1].set_title("B. Uncertainty Inflation & Error Response", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("Percentage / Uncertainty (x100)", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_subgroup_shift(
    df_subgroups: pd.DataFrame,
    save_path: Path,
):
    """Plot discrimination and calibration reliability across demographic cohorts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Filter demographic cohorts
    df_demo = df_subgroups[df_subgroups["category"].isin(["Gender", "Age", "Race"])].copy()
    y_pos = np.arange(len(df_demo))

    # Panel A: ROC-AUC and PR-AUC
    width = 0.35
    axes[0].barh(y_pos - width / 2, df_demo["roc_auc"], width, label="ROC-AUC", color="#2b83ba", alpha=0.85)
    axes[0].barh(y_pos + width / 2, df_demo["pr_auc"] * 2.5, width, label="PR-AUC (scaled x2.5)", color="#fdae61", alpha=0.85)
    axes[0].axvline(0.6504, color="#d7191c", linestyle="--", linewidth=1.2, label="Nominal Test ROC-AUC (0.650)")

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(df_demo["subgroup_name"], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title("A. Discrimination Across Demographic Subgroups", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Metric Value", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.5, axis="x")
    axes[0].legend(loc="lower right", fontsize=8)

    # Panel B: Calibration Slope & Mean Uncertainty
    axes[1].barh(y_pos - width / 2, df_demo["calibration_slope"], width, label="Calibration Slope", color="#abdda4", alpha=0.85)
    axes[1].barh(y_pos + width / 2, df_demo["mean_uncertainty"] * 40.0, width, label="Mean Uncertainty (scaled x40)", color="#d7191c", alpha=0.85)
    axes[1].axvline(1.0, color="gray", linestyle=":", linewidth=1.0, label="Ideal Slope (1.0)")

    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(df_demo["subgroup_name"], fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_title("B. Calibration Slope & Uncertainty Across Subgroups", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Slope / Scaled Uncertainty", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.5, axis="x")
    axes[1].legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_encounter_shift(
    df_strata: pd.DataFrame,
    save_path: Path,
):
    """Plot performance across encounter utilization and complexity strata."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    y_pos = np.arange(len(df_strata))

    # Panel A: Error Rate and Readmission Prevalence
    width = 0.35
    axes[0].barh(y_pos - width / 2, df_strata["prevalence"] * 100, width, label="Observed Prevalence (%)", color="#e41a1c", alpha=0.8)
    axes[0].barh(y_pos + width / 2, df_strata["error_rate"] * 100, width, label="Model Error Rate (%)", color="#377eb8", alpha=0.8)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(df_strata["stratum_name"], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title("A. Observed Prevalence vs. Error Rate by Stratum", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Percentage (%)", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.5, axis="x")
    axes[0].legend(loc="lower right", fontsize=8)

    # Panel B: Mean Uncertainty & Error Detection AUROC
    axes[1].barh(y_pos - width / 2, df_strata["mean_uncertainty"] * 100, width, label="Mean Uncertainty (σ_p x100)", color="#984ea3", alpha=0.8)
    axes[1].barh(y_pos + width / 2, df_strata["error_detection_auroc"], width, label="Error Detection AUROC", color="#4daf4a", alpha=0.8)
    axes[1].axvline(0.7116, color="#ff7f00", linestyle="--", linewidth=1.2, label="Nominal Error AUROC (0.712)")

    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(df_strata["stratum_name"], fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_title("B. Epistemic Uncertainty & Error Detection by Stratum", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Uncertainty (x100) / Error AUROC", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.5, axis="x")
    axes[1].legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_temporal_shift(
    df_temporal: pd.DataFrame,
    save_path: Path,
):
    """Plot longitudinal metric stability across chronological encounter progression."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Filter Quartiles for chronological trend
    df_q = df_temporal[df_temporal["category"] == "Chronological Quartile"].copy()
    labels = ["Q1: Earliest\n(1999-2001)", "Q2: Early-Mid\n(2002-2003)", "Q3: Late-Mid\n(2004-2006)", "Q4: Latest\n(2007-2008)"]
    x = np.arange(len(df_q))

    # Panel A: Discrimination & Calibration Stability
    axes[0].plot(x, df_q["roc_auc"], marker="o", color="#1b9e77", linewidth=2.2, label="ROC-AUC")
    axes[0].plot(x, df_q["pr_auc"] * 3.0, marker="s", color="#d95f02", linewidth=2.0, linestyle="--", label="PR-AUC (scaled x3)")
    axes[0].plot(x, df_q["calibration_slope"], marker="^", color="#7570b3", linewidth=2.0, linestyle=":", label="Calibration Slope")

    axes[0].set_title("A. Longitudinal Discrimination & Calibration Stability", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("Metric Value", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower left", fontsize=9)

    # Panel B: Mean Uncertainty & Error Detection
    axes[1].plot(x, df_q["mean_uncertainty"] * 100, marker="o", color="#e7298a", linewidth=2.2, label="Mean Uncertainty (σ_p x100)")
    axes[1].plot(x, df_q["error_detection_auroc"], marker="^", color="#66a61e", linewidth=2.0, linestyle="--", label="Error Detection AUROC")
    axes[1].plot(x, df_q["error_rate"] * 100, marker="d", color="#e6ab02", linewidth=2.0, linestyle=":", label="Error Rate (%)")

    axes[1].set_title("B. Epistemic Dispersion & Failure Warning Over Time", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("Uncertainty (x100) / Metric Value", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_uncertainty_shift(
    df_missingness: pd.DataFrame,
    df_composition: pd.DataFrame,
    df_encounter_comp: pd.DataFrame,
    save_path: Path,
):
    """Plot uncertainty inflation across all key shift dimensions."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Compile representative shift scenarios
    scenarios = [
        ("Nominal Baseline", df_missingness.loc[df_missingness["scenario"] == "M0_Nominal_Reference", "mean_uncertainty"].values[0]),
        ("Missingness +10%", df_missingness.loc[df_missingness["scenario"] == "M1_Random_MCAR_10pct", "mean_uncertainty"].values[0]),
        ("Missingness +25%", df_missingness.loc[df_missingness["scenario"] == "M2_Random_MCAR_25pct", "mean_uncertainty"].values[0]),
        ("Missingness +50%", df_missingness.loc[df_missingness["scenario"] == "M3_Random_MCAR_50pct", "mean_uncertainty"].values[0]),
        ("Targeted Glycemic Mask", df_missingness.loc[df_missingness["scenario"] == "M4a_Targeted_Glycemic_Lab", "mean_uncertainty"].values[0]),
        ("Targeted Meds Mask", df_missingness.loc[df_missingness["scenario"] == "M4b_Targeted_Medications", "mean_uncertainty"].values[0]),
        ("Shift: Geriatric Skew", df_composition.loc[df_composition["shift_scenario"] == "Shift_Geriatric_Enriched", "mean_uncertainty"].values[0]),
        ("Shift: High Inpatient", df_encounter_comp.loc[df_encounter_comp["shift_scenario"] == "Shift_High_Utilization_Heavy", "mean_uncertainty"].values[0]),
        ("Shift: Multimorbid", df_encounter_comp.loc[df_encounter_comp["shift_scenario"] == "Shift_Multimorbidity_Heavy", "mean_uncertainty"].values[0]),
        ("Shift: Low Complexity", df_encounter_comp.loc[df_encounter_comp["shift_scenario"] == "Shift_Low_Complexity_Heavy", "mean_uncertainty"].values[0]),
    ]

    names = [s[0] for s in scenarios]
    vals = np.array([s[1] for s in scenarios])
    base_val = vals[0]

    colors = ["#2b83ba" if i == 0 else "#fdae61" if vals[i] < base_val * 1.2 else "#d7191c" for i in range(len(vals))]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axvline(base_val, color="#2b83ba", linestyle="--", linewidth=1.5, label=f"Nominal Uncertainty (σ_p = {base_val:.4f})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Predictive Uncertainty (σ_p)", fontsize=10)
    ax.set_title("Uncertainty Inflation Across Controlled Distribution Shift Scenarios", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_risk_coverage_shift(
    nominal_y: np.ndarray,
    nominal_prob: np.ndarray,
    nominal_u: np.ndarray,
    shifted_scenarios: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    save_path: Path,
    threshold: float = 0.20,
):
    """Plot selective prediction risk-coverage curves under nominal vs shifted distributions."""
    plt.figure(figsize=(9, 6), dpi=300)

    # Compute risk-coverage curve for nominal
    def get_curve(y_t, p_t, u_t):
        err = ((p_t >= threshold).astype(int) != y_t).astype(int)
        order = np.argsort(u_t)
        err_sorted = err[order]
        covs = np.linspace(0.10, 1.00, 50)
        risks = [float(np.mean(err_sorted[:max(1, int(np.ceil(c * len(y_t))))])) * 100 for c in covs]
        return covs * 100, np.array(risks)

    cov_nom, risk_nom = get_curve(nominal_y, nominal_prob, nominal_u)
    plt.plot(cov_nom, risk_nom, color="#2b83ba", linewidth=2.5, label=f"Nominal Reference (Base Error = {risk_nom[-1]:.2f}%)")

    styles = [
        ("Missingness +25%", "#fdae61", "-."),
        ("Missingness +50%", "#d7191c", ":"),
        ("High-Utilization Heavy", "#984ea3", "--"),
        ("Multimorbidity Heavy", "#4daf4a", "-"),
    ]

    for (name, color, ls) in styles:
        if name in shifted_scenarios:
            y_s, p_s, u_s = shifted_scenarios[name]
            cov_s, risk_s = get_curve(y_s, p_s, u_s)
            plt.plot(cov_s, risk_s, color=color, linestyle=ls, linewidth=2.0, label=f"{name} (Base Error = {risk_s[-1]:.2f}%)")

    plt.title("Selective Classification: Risk-Coverage Dynamics Under Distribution Shift", fontsize=11, fontweight="bold")
    plt.xlabel("Coverage (% of Cohort Retained by Uncertainty Filter)", fontsize=10)
    plt.ylabel("Residual Classification Error Rate (%)", fontsize=10)
    plt.xlim([10, 102])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=8.5)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_robustness_summary(
    df_summary_matrix: pd.DataFrame,
    save_path: Path,
):
    """Plot master robustness degradation scoreboard across all evaluation dimensions."""
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    # Plot delta ROC-AUC vs delta Error Rate
    categories = df_summary_matrix["category"].unique()
    cat_colors = {
        "Nominal Reference": "#377eb8",
        "Missingness Perturbation": "#e41a1c",
        "Demographic Cohort": "#4daf4a",
        "Encounter Complexity": "#984ea3",
        "Temporal Progression": "#ff7f00",
    }

    for cat in categories:
        sub = df_summary_matrix[df_summary_matrix["category"] == cat]
        color = cat_colors.get(cat, "gray")
        ax.scatter(
            sub["delta_roc_auc"],
            sub["delta_error_rate"] * 100.0,
            s=sub["n_samples"] / 100.0 + 30,
            color=color,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.6,
            label=cat,
        )

        for _, r in sub.iterrows():
            ax.annotate(
                r["shift_name"],
                (r["delta_roc_auc"], r["delta_error_rate"] * 100.0),
                fontsize=7.5,
                alpha=0.85,
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)

    ax.set_title("Master Robustness Landscape: Discrimination vs Error Rate Shift", fontsize=11, fontweight="bold")
    ax.set_xlabel("Δ ROC-AUC (Relative to Nominal Reference 0.6504)", fontsize=10)
    ax.set_ylabel("Δ Error Rate (Percentage Points)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
