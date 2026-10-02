"""
FusionMedAI - Phase C11.5 Scientific Figure Generator
Generates publication-quality figures for Phase C11.5 multimodal baseline evaluation.
Saves figures to `research/fusion/figures/c11_5/`.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "experiments" / "fusion" / "baseline_comparison"
OUT_DIR = REPO_ROOT / "research" / "fusion" / "Volume_05_Baseline_Fusion" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Color Palette
COLOR_RETINA = "#2b5c8f"    # Clinical Blue
COLOR_FOOT = "#d95f02"      # Wound Orange
COLOR_CLINICAL = "#7570b3"  # EHR Purple
COLOR_ENTROPY = "#1b9e77"   # Sage Green
COLOR_GRID = "#e0e0e0"

# -----------------------------------------------------------------------------
# 1. Baseline Weight Distribution
# -----------------------------------------------------------------------------
def plot_baseline_weight_distribution():
    with open(EXP_DIR / "summary.json", "r") as f:
        summary_data = json.load(f)["tri_modal_summary"]

    baselines = ["B1", "B2", "B3", "B4", "B5", "B6"]
    w_retina = [summary_data[b]["mean_weights"]["retina"] for b in baselines]
    w_foot = [summary_data[b]["mean_weights"]["foot"] for b in baselines]
    w_clinical = [summary_data[b]["mean_weights"]["clinical"] for b in baselines]

    x = np.arange(len(baselines))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    r1 = ax.bar(x - width, w_retina, width, label="Retina ($w_R$)", color=COLOR_RETINA, edgecolor="black", linewidth=0.8, alpha=0.9)
    r2 = ax.bar(x, w_foot, width, label="Foot Ulcer ($w_F$)", color=COLOR_FOOT, edgecolor="black", linewidth=0.8, alpha=0.9)
    r3 = ax.bar(x + width, w_clinical, width, label="Clinical Tabular ($w_C$)", color=COLOR_CLINICAL, edgecolor="black", linewidth=0.8, alpha=0.9)

    ax.set_ylabel("Mean Decision Authority Weight ($w_i$)")
    ax.set_title("Modality Weight Allocation Across Baseline Ladder (N=500)")
    ax.set_xticks(x)
    ax.set_xticklabels([
        "B1\n(Reliability)",
        "B2\n(Uniform)",
        "B3\n(Confidence)",
        "B4\n(Conf+Rel)",
        "B5\n(Conf+Rel+Unc)",
        "B6\n(Full ACARA-U)"
    ])
    ax.set_ylim(0, 1.08)
    ax.axhline(1.0/3.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, label="Equal Split (1/3)")
    ax.legend(frameon=True, facecolor="white", edgecolor=COLOR_GRID, loc="upper right")

    for rects in [r1, r2, r3]:
        for rect in rects:
            height = rect.get_height()
            if height > 0.03:
                ax.annotate(f"{height:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT_DIR / "baseline_weight_distribution.png", dpi=300)
    plt.close()
    print("  -> Saved baseline_weight_distribution.png")


# -----------------------------------------------------------------------------
# 2. Routing Entropy Comparison
# -----------------------------------------------------------------------------
def plot_routing_entropy():
    with open(EXP_DIR / "summary.json", "r") as f:
        summary_data = json.load(f)["tri_modal_summary"]

    baselines = ["B1", "B2", "B3", "B4", "B5", "B6"]
    entropies = [summary_data[b]["mean_routing_entropy"] for b in baselines]
    labels = ["B1\nUnimodal", "B2\nUniform", "B3\nConf", "B4\nConf+Rel", "B5\n+Uncert", "B6\nACARA-U"]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    bars = ax.bar(labels, entropies, color=COLOR_ENTROPY, width=0.5, edgecolor="black", linewidth=0.8, alpha=0.85)

    max_entropy = np.log(3)
    ax.axhline(max_entropy, color="#d95f02", linestyle="--", linewidth=1.5, label=f"Theoretical Max $H(w) = \\ln(3) \\approx {max_entropy:.4f}$")

    ax.set_ylabel("Mean Routing Entropy $H(w) = -\\sum w_i \\ln(w_i)$")
    ax.set_title("Routing Entropy Across Baseline Ladder (N=500)")
    ax.set_ylim(0, 1.25)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.legend(frameon=True, facecolor="white", edgecolor=COLOR_GRID, loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "routing_entropy.png", dpi=300)
    plt.close()
    print("  -> Saved routing_entropy.png")


# -----------------------------------------------------------------------------
# 3. Modality Dominance Rate
# -----------------------------------------------------------------------------
def plot_modality_dominance():
    with open(EXP_DIR / "summary.json", "r") as f:
        summary_data = json.load(f)["tri_modal_summary"]

    baselines = ["B1", "B2", "B3", "B4", "B5", "B6"]
    dom_retina = [summary_data[b]["dominant_modality_rates"].get("retina", 0) * 100.0 for b in baselines]
    dom_foot = [summary_data[b]["dominant_modality_rates"].get("foot", 0) * 100.0 for b in baselines]
    dom_clinical = [summary_data[b]["dominant_modality_rates"].get("clinical", 0) * 100.0 for b in baselines]

    x = np.arange(len(baselines))
    width = 0.55

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

    p1 = ax.bar(x, dom_retina, width, label="Retina Plurality", color=COLOR_RETINA, edgecolor="black", linewidth=0.8)
    p2 = ax.bar(x, dom_foot, width, bottom=dom_retina, label="Foot Ulcer Plurality", color=COLOR_FOOT, edgecolor="black", linewidth=0.8)
    bottom_c = [r + f for r, f in zip(dom_retina, dom_foot)]
    p3 = ax.bar(x, dom_clinical, width, bottom=bottom_c, label="Clinical Plurality", color=COLOR_CLINICAL, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Dominant Modality Allocation Rate (%)")
    ax.set_title("Modality Authority Plurality Rate Across Baselines (N=500)")
    ax.set_xticks(x)
    ax.set_xticklabels([
        "B1\n(Reliability)",
        "B2\n(Uniform)",
        "B3\n(Confidence)",
        "B4\n(Conf+Rel)",
        "B5\n(Conf+Rel+Unc)",
        "B6\n(Full ACARA-U)"
    ])
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, facecolor="white", edgecolor=COLOR_GRID, loc="upper right")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "modality_dominance.png", dpi=300)
    plt.close()
    print("  -> Saved modality_dominance.png")


# -----------------------------------------------------------------------------
# 4. Missing Modality Behavior (7 Operational Configs)
# -----------------------------------------------------------------------------
def plot_missing_modality_behavior():
    with open(EXP_DIR / "missing_modality_matrix.json", "r") as f:
        data = json.load(f)

    # Plot B6 (Full ACARA-U) behavior across all 7 configurations
    configs = [
        "Config_1_Tri_Modal",
        "Config_2_Bi_Modal_RF",
        "Config_3_Bi_Modal_RC",
        "Config_4_Bi_Modal_FC",
        "Config_5_Uni_Modal_R",
        "Config_6_Uni_Modal_F",
        "Config_7_Uni_Modal_C"
    ]
    config_labels = [
        "C1: R+F+C\n(Tri-Modal)",
        "C2: R+F\n(No Clin)",
        "C3: R+C\n(No Foot)",
        "C4: F+C\n(No Ret)",
        "C5: R Only\n(Retina)",
        "C6: F Only\n(Foot)",
        "C7: C Only\n(Clinical)"
    ]

    w_r = [data[c]["B6"]["mean_weights"]["retina"] for c in configs]
    w_f = [data[c]["B6"]["mean_weights"]["foot"] for c in configs]
    w_c = [data[c]["B6"]["mean_weights"]["clinical"] for c in configs]

    x = np.arange(len(configs))
    width = 0.55

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    p1 = ax.bar(x, w_r, width, label="Retina ($w_R$)", color=COLOR_RETINA, edgecolor="black", linewidth=0.8)
    p2 = ax.bar(x, w_f, width, bottom=w_r, label="Foot Ulcer ($w_F$)", color=COLOR_FOOT, edgecolor="black", linewidth=0.8)
    bottom_c = [r + f for r, f in zip(w_r, w_f)]
    p3 = ax.bar(x, w_c, width, bottom=bottom_c, label="Clinical Tabular ($w_C$)", color=COLOR_CLINICAL, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Normalized Routing Weight ($\\sum w_i = 1.0$)")
    ax.set_title("ACARA-U (B6) Authority Redistribution Under Missing Modality Scenarios")
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, facecolor="white", edgecolor=COLOR_GRID, loc="upper right")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "missing_modality_behavior.png", dpi=300)
    plt.close()
    print("  -> Saved missing_modality_behavior.png")


# -----------------------------------------------------------------------------
# 5. Perturbation Response Curves (C ^, U ^, Q v)
# -----------------------------------------------------------------------------
def plot_perturbation_response():
    with open(EXP_DIR / "perturbation_benchmark.json", "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

    # Panel A: Confidence Sweep
    x_c = [0.50, 0.75, 1.00]
    conf_data = data["confidence_responsiveness"]
    axes[0].plot(x_c, conf_data["B2"], label="B2 (Uniform)", color="gray", linestyle="--", linewidth=1.8, marker="o")
    axes[0].plot(x_c, conf_data["B3"], label="B3 (Confidence)", color=COLOR_CLINICAL, linestyle="-.", linewidth=2.0, marker="s")
    axes[0].plot(x_c, conf_data["B4"], label="B4 (Conf+Rel)", color="#e7298a", linestyle=":", linewidth=2.0, marker="^")
    axes[0].plot(x_c, conf_data["B6"], label="B6 (ACARA-U)", color=COLOR_RETINA, linestyle="-", linewidth=2.2, marker="D")
    axes[0].set_xlabel("Instance Confidence ($C_i$)")
    axes[0].set_ylabel("Modality Routing Weight ($w_i$)")
    axes[0].set_title("A: Confidence Responsiveness ($C_i \\uparrow$)")
    axes[0].set_ylim(0.2, 0.5)
    axes[0].legend(frameon=True, facecolor="white", fontsize=9)

    # Panel B: Uncertainty Surge
    x_u = [0.00, 0.50, 1.00]
    unc_data = data["uncertainty_responsiveness"]
    axes[1].plot(x_u, unc_data["B2"], label="B2 (Uniform)", color="gray", linestyle="--", linewidth=1.8, marker="o")
    axes[1].plot(x_u, unc_data["B4"], label="B4 (No Uncert)", color="black", linestyle=":", linewidth=2.0, marker="^")
    axes[1].plot(x_u, unc_data["B5"], label="B5 (+Uncert)", color="#e7298a", linestyle="-.", linewidth=2.0, marker="s")
    axes[1].plot(x_u, unc_data["B6"], label="B6 (ACARA-U)", color=COLOR_RETINA, linestyle="-", linewidth=2.2, marker="D")
    axes[1].set_xlabel("Predictive Uncertainty ($U_i$)")
    axes[1].set_ylabel("Modality Routing Weight ($w_i$)")
    axes[1].set_title("B: Uncertainty Penalty ($U_i \\uparrow$)")
    axes[1].set_ylim(0.2, 0.5)
    axes[1].legend(frameon=True, facecolor="white", fontsize=9)

    # Panel C: Quality Responsiveness
    x_q = [0.00, 0.50, 1.00]
    qual_data = data["quality_responsiveness"]
    axes[2].plot(x_q, qual_data["B2"], label="B2 (Uniform)", color="gray", linestyle="--", linewidth=1.8, marker="o")
    axes[2].plot(x_q, qual_data["B4"], label="B4 (No Quality)", color="black", linestyle=":", linewidth=2.0, marker="^")
    axes[2].plot(x_q, qual_data["B5"], label="B5 (No Quality)", color="#e7298a", linestyle="-.", linewidth=2.0, marker="s")
    axes[2].plot(x_q, qual_data["B6"], label="B6 (ACARA-U)", color=COLOR_FOOT, linestyle="-", linewidth=2.2, marker="D")
    axes[2].set_xlabel("Input Quality ($Q_i$)")
    axes[2].set_ylabel("Modality Routing Weight ($w_i$)")
    axes[2].set_title("C: Quality Responsiveness ($Q_i \\uparrow$)")
    axes[2].set_ylim(0.2, 0.5)
    axes[2].legend(frameon=True, facecolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "perturbation_response.png", dpi=300)
    plt.close()
    print("  -> Saved perturbation_response.png")


# -----------------------------------------------------------------------------
# 6. Cross-Modality Risk Conflict / Disagreement
# -----------------------------------------------------------------------------
def plot_conflict_disagreement():
    with open(EXP_DIR / "disagreement_analysis.json", "r") as f:
        data = json.load(f)

    metrics = [
        "$X_{RF}$\n(|Retina - Foot|)",
        "$X_{RC}$\n(|Retina - Clin|)",
        "$X_{FC}$\n(|Foot - Clin|)",
        "$X_{\\max}$\n(Max Conflict)"
    ]
    means = [
        data["mean_abs_disagreement_RF"],
        data["mean_abs_disagreement_RC"],
        data["mean_abs_disagreement_FC"],
        data["mean_X_max"]
    ]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    bars = ax.bar(metrics, means, color=["#386cb0", "#7570b3", "#d95f02", "#e7298a"],
                  edgecolor="black", width=0.5, linewidth=0.8, alpha=0.85)

    ax.set_ylabel("Mean Pairwise Absolute Risk Divergence")
    ax.set_title("Cross-Modality Pairwise Risk Divergence Profile (N=500)")
    ax.set_ylim(0, 0.65)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT_DIR / "conflict_disagreement.png", dpi=300)
    plt.close()
    print("  -> Saved conflict_disagreement.png")


if __name__ == "__main__":
    print("Generating Phase C11.5 Publication Figures...")
    plot_baseline_weight_distribution()
    plot_routing_entropy()
    plot_modality_dominance()
    plot_missing_modality_behavior()
    plot_perturbation_response()
    plot_conflict_disagreement()
    print("All figures successfully created in research/fusion/figures/c11_5/!")
