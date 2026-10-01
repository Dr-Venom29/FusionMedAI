"""
Generate visual example figures for local patient cases and clinical input/output flow.
"""

import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "clinical" / "uncertainty"
RESEARCH_DIR = REPO_ROOT / "research" / "clinical" / "Volume_08_Uncertainty"


def plot_local_cases_chart():
    csv_path = EXPERIMENT_DIR / "tables" / "local_uncertainty_cases.csv"
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    y_pos = np.arange(len(df))
    # Case short labels
    labels = [
        "Case 1: High Risk / Low Unc (Inpatient=12)",
        "Case 2: High Risk / High Unc (Age 90-100)",
        "Case 3: Low Risk / Low Unc (Inpatient=0)",
        "Case 4: Low Risk / High Unc (Complex Diag)",
        "Case 5: Near-Threshold Ambiguity (Inpatient=2)",
        "Case 6: False Positive / High Unc (Inpatient=5)",
        "Case 7: False Negative / High Unc (Inpatient=0)",
    ]

    means = df["raw_mean_prob"].values
    cals = df["calibrated_prob"].values
    lowers = df["pi_95_lower"].values
    uppers = df["pi_95_upper"].values
    actuals = df["true_outcome"].values

    # Error bars (lower to upper)
    xerr_lower = means - lowers
    xerr_upper = uppers - means

    # Plot 95% Interval
    ax.errorbar(
        means,
        y_pos,
        xerr=[xerr_lower, xerr_upper],
        fmt="none",
        ecolor="#2b83ba",
        elinewidth=2.5,
        capsize=6,
        capthick=2,
        label="95% Bootstrap Prediction Interval [q_2.5, q_97.5]",
    )

    # Plot Raw Mean
    ax.scatter(
        means,
        y_pos,
        color="#2b83ba",
        s=80,
        zorder=5,
        label="Raw Ensemble Mean (p̄)",
    )

    # Plot Calibrated Risk
    ax.scatter(
        cals,
        y_pos,
        color="#d7191c",
        marker="D",
        s=70,
        zorder=6,
        label="Isotonic Calibrated Risk (p_cal)",
    )

    # Actual outcome markers on right
    for i, act in enumerate(actuals):
        color = "#d7191c" if act == 1 else "#2ca25f"
        txt = "Readmitted (y=1)" if act == 1 else "Not Readmitted (y=0)"
        ax.text(
            1.02,
            i,
            txt,
            color=color,
            fontweight="bold",
            va="center",
            fontsize=8.5,
        )

    # Decision threshold line
    ax.axvline(0.20, color="#313695", linestyle="--", linewidth=1.8, label="Operating Threshold θ = 0.20")
    ax.axvspan(0.17, 0.23, color="orange", alpha=0.15, label="Decision Ambiguity Zone [0.17, 0.23]")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold")
    ax.invert_yaxis()  # top-down
    ax.set_xlabel("Predicted Readmission Probability", fontsize=11)
    ax.set_title("Representative Patient Case Studies: Prediction Intervals & Decision Ambiguity", fontsize=12, fontweight="bold")
    ax.set_xlim([-0.02, 1.25])
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")
    ax.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    out_path = EXPERIMENT_DIR / "figures" / "local_case_profiles.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_path}")


def plot_clinical_io_pipeline():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.axis("off")

    # Draw boxes for I/O pipeline
    # Box 1: Input EHR Data
    bbox_in = dict(boxstyle="round,pad=0.6", fc="#e0f3f8", ec="#2b83ba", lw=2)
    ax.text(0.10, 0.50, "INPUT\nPatient Encounter\n(119 Clinical Features)\n\n• Prior Inpatient Visits\n• Time in Hospital\n• Diagnoses & Meds\n• Age & Glycemic Flags", ha="center", va="center", fontsize=9.5, bbox=bbox_in)

    # Arrow 1
    ax.annotate("", xy=(0.24, 0.50), xytext=(0.19, 0.50), arrowprops=dict(arrowstyle="->", lw=2.5, color="#2b83ba"))

    # Box 2: Bootstrap Ensemble (50 Models)
    bbox_ens = dict(boxstyle="round,pad=0.6", fc="#fee0d2", ec="#de2d26", lw=2)
    ax.text(0.38, 0.50, "INFERENCE\nBootstrap CatBoost\n(50 Ensemble Models)\n\n• Model 1 → p₁\n• Model 2 → p₂\n• ...\n• Model 50 → p₅₀", ha="center", va="center", fontsize=9.5, bbox=bbox_ens)

    # Arrow 2
    ax.annotate("", xy=(0.52, 0.50), xytext=(0.47, 0.50), arrowprops=dict(arrowstyle="->", lw=2.5, color="#de2d26"))

    # Box 3: Calibration & Uncertainty
    bbox_proc = dict(boxstyle="round,pad=0.6", fc="#e5f5e0", ec="#31a354", lw=2)
    ax.text(0.66, 0.50, "PROCESSING\nCalibration & Variance\n\n• Raw Mean: p̄(x)\n• Isotonic Calibrated: p_cal\n• Bootstrap Std: σ_p(x)\n• 95% Interval: [q_2.5, q_97.5]\n• Decision Tier Assignment", ha="center", va="center", fontsize=9.5, bbox=bbox_proc)

    # Arrow 3
    ax.annotate("", xy=(0.80, 0.50), xytext=(0.75, 0.50), arrowprops=dict(arrowstyle="->", lw=2.5, color="#31a354"))

    # Box 4: ClinicalOutput Schema
    bbox_out = dict(boxstyle="round,pad=0.6", fc="#f7f7f7", ec="#525252", lw=2)
    sample_json = (
        "OUTPUT: ClinicalOutput JSON\n\n"
        "{\n"
        '  "prediction": 1,\n'
        '  "probability": 0.219,\n'
        '  "calibrated_prob": 0.216,\n'
        '  "confidence": "moderate",\n'
        '  "uncertainty": {\n'
        '    "std_probability": 0.039,\n'
        '    "interval_95": [0.138, 0.283]\n'
        "  },\n"
        '  "decision_tier": "Near Threshold Ambiguity"\n'
        "}"
    )
    ax.text(0.92, 0.50, sample_json, ha="center", va="center", fontsize=8, family="monospace", bbox=bbox_out)

    plt.title("Clinical Tabular Inference Pipeline: Input Features → Bootstrap Uncertainty → ClinicalOutput Contract", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    out_path = EXPERIMENT_DIR / "figures" / "clinical_output_pipeline.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_path}")


def mirror_to_research():
    res_fig_dir = RESEARCH_DIR / "figures"
    res_fig_dir.mkdir(parents=True, exist_ok=True)
    for fig_name in ["local_case_profiles.png", "clinical_output_pipeline.png"]:
        src = EXPERIMENT_DIR / "figures" / fig_name
        dst = res_fig_dir / fig_name
        shutil.copy2(src, dst)
        print(f"Mirrored: {dst}")


if __name__ == "__main__":
    plot_local_cases_chart()
    plot_clinical_io_pipeline()
    mirror_to_research()
