"""
FusionMedAI - Phase C11.14: Plot Publication Figures
Generates 4 high-resolution (300 DPI) publication figures for Volume 14:
- fig1_policy_action_distributions.png
- fig2_threshold_sensitivity_heatmap.png
- fig3_regime_action_stratification.png
- fig4_action_transition_flow.png
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "experiments" / "fusion" / "dcri_policy_analysis" / "results"
FIG_DIR = REPO_ROOT / "research" / "fusion" / "Volume_14_DCRI_Decision_Policy_Analysis" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_fig1_action_distributions(threshold_data: dict):
    """Figure 1: Policy A vs Policy B Action Distributions at Standard Thresholds."""
    p_op = threshold_data["primary_operating_point"]
    p_a = p_op["policy_a_dcri"]["percentages"]
    p_b = p_op["policy_b_fused_risk"]["percentages"]

    categories = ["Routine Review\n(Tier 0: < 0.20)", "Additional Assessment\n(Tier 1: [0.20, 0.40))", "Escalation\n(Tier 2: >= 0.40)"]
    vals_b = [p_b["routine_review"], p_b["additional_assessment"], p_b["escalation"]]
    vals_a = [p_a["routine_review"], p_a["additional_assessment"], p_a["escalation"]]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    rects1 = ax.bar(x - width/2, vals_b, width, label="Policy B: Fused Risk $R_{\\mathrm{fusion}}$ (Unpenalized)", color="#4A90E2", edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, vals_a, width, label="Policy A: $\\mathrm{DCRI}_{0.10}$ (Uncertainty-Discounted)", color="#50E3C2", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Proportion of Cohort (%)")
    ax.set_title("Hypothetical Action Distribution: Fused Risk vs. $\\mathrm{DCRI}_{0.10}$ ($\\tau_1=0.20, \\tau_2=0.40$)", pad=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 60)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95)

    # Bar labels
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    out_path = FIG_DIR / "fig1_policy_action_distributions.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig2_threshold_sensitivity_heatmap(threshold_data: dict):
    """Figure 2: Reclassification Rate Across the 2-D Threshold Grid."""
    sweep = threshold_data["threshold_sweep_results"]
    tau_1_vals = sorted(list(set(r["tau_1"] for r in sweep)))
    tau_2_vals = sorted(list(set(r["tau_2"] for r in sweep)))

    reclass_grid = np.full((len(tau_1_vals), len(tau_2_vals)), np.nan)

    for r in sweep:
        i = tau_1_vals.index(r["tau_1"])
        j = tau_2_vals.index(r["tau_2"])
        reclass_grid[i, j] = r["reclassifications"]["rate"] * 100.0

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=300)
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad("white")

    im = ax.imshow(reclass_grid, cmap=cmap, origin="lower", aspect="auto")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(tau_2_vals)))
    ax.set_yticks(np.arange(len(tau_1_vals)))
    ax.set_xticklabels([f"{v:.2f}" for v in tau_2_vals])
    ax.set_yticklabels([f"{v:.2f}" for v in tau_1_vals])

    ax.set_xlabel("High Threshold $\\tau_2$ (Escalation Boundary)")
    ax.set_ylabel("Low Threshold $\\tau_1$ (Routine Boundary)")
    ax.set_title("Policy Reclassification Rate (%) Across Threshold Grid", pad=12, fontweight="bold")

    # Loop over data dimensions and create text annotations.
    for i in range(len(tau_1_vals)):
        for j in range(len(tau_2_vals)):
            val = reclass_grid[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 20.0 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=text_color, fontweight="bold", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Reclassification Rate (%)")

    plt.tight_layout()
    out_path = FIG_DIR / "fig2_threshold_sensitivity_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig3_regime_action_stratification(regime_data: dict):
    """Figure 3: Action Distribution Stratified Across Modality Availability Regimes."""
    reg_dict = regime_data["regime_results"]
    active_regs = ["R", "F", "C", "RF", "RC", "FC", "RFC"]

    routine_pcts = [reg_dict[r]["policy_a_dcri"]["percentages"]["routine_review"] for r in active_regs]
    additional_pcts = [reg_dict[r]["policy_a_dcri"]["percentages"]["additional_assessment"] for r in active_regs]
    escalation_pcts = [reg_dict[r]["policy_a_dcri"]["percentages"]["escalation"] for r in active_regs]

    x = np.arange(len(active_regs))
    width = 0.55

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

    p1 = ax.bar(x, routine_pcts, width, label="Routine Review (Tier 0)", color="#2ECC71", edgecolor="black", linewidth=0.6)
    p2 = ax.bar(x, additional_pcts, width, bottom=routine_pcts, label="Additional Assessment (Tier 1)", color="#F39C12", edgecolor="black", linewidth=0.6)
    bottom_3 = [routine_pcts[i] + additional_pcts[i] for i in range(len(active_regs))]
    p3 = ax.bar(x, escalation_pcts, width, bottom=bottom_3, label="Escalation (Tier 2)", color="#E74C3C", edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Proportion of Regime Cohort (%)")
    ax.set_title("$\\mathrm{DCRI}_{0.10}$ Decision Action Allocation by Availability Regime", pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(M={len(r)})" for r in active_regs])
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    out_path = FIG_DIR / "fig3_regime_action_stratification.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig4_transition_matrix(threshold_data: dict):
    """Figure 4: Policy Transition / Migration Flow Matrix."""
    matrix = np.array(threshold_data["primary_operating_point"]["transition_matrix_fused_to_dcri"])

    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)
    im = ax.imshow(matrix, cmap="Blues", origin="upper")

    labels = ["Routine (0)", "Additional (1)", "Escalation (2)"]
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Policy A Action ($\\mathrm{DCRI}_{0.10}$)", fontweight="bold")
    ax.set_ylabel("Policy B Action ($R_{\\mathrm{fusion}}$ Reference)", fontweight="bold")
    ax.set_title("Action Migration Matrix ($R_{\\mathrm{fusion}} \\to \\mathrm{DCRI}_{0.10}$)", pad=14, fontweight="bold")

    for i in range(3):
        for j in range(3):
            count = matrix[i, j]
            pct = count / 500.0 * 100.0
            color = "white" if count > 100 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Packet Count")

    plt.tight_layout()
    out_path = FIG_DIR / "fig4_action_transition_flow.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    with open(RESULTS_DIR / "threshold_results.json", "r", encoding="utf-8") as f:
        threshold_data = json.load(f)
    with open(RESULTS_DIR / "regime_results.json", "r", encoding="utf-8") as f:
        regime_data = json.load(f)

    plot_fig1_action_distributions(threshold_data)
    plot_fig2_threshold_sensitivity_heatmap(threshold_data)
    plot_fig3_regime_action_stratification(regime_data)
    plot_fig4_transition_matrix(threshold_data)
    print("All 4 Phase C11.14 figures generated successfully.")


if __name__ == "__main__":
    main()
