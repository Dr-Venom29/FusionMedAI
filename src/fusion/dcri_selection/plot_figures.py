"""
src/fusion/dcri_selection/plot_figures.py
Generates publication-quality 300 DPI figures for Research Volume 13: DCRI Global Uncertainty Penalty Selection.
Dynamically reads parameters and metrics from experimental result artifacts.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# Ensure matplotlib uses a non-interactive backend
plt.switch_backend("Agg")


def generate_volume_13_figures(repo_root: Path):
    exp_dir = repo_root / "experiments" / "fusion" / "dcri_selection" / "results"
    fig_dir = repo_root / "research" / "fusion" / "Volume_13_DCRI_Delta_Selection" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "delta_selection_config.json", "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
    with open(exp_dir / "selection_summary.json", "r", encoding="utf-8") as f:
        sel_data = json.load(f)
    with open(exp_dir / "delta_results.json", "r", encoding="utf-8") as f:
        delta_data = json.load(f)["candidates"]
    with open(exp_dir / "distribution_results.json", "r", encoding="utf-8") as f:
        dist_data = json.load(f)["distribution_profiles"]
    with open(exp_dir / "regime_results.json", "r", encoding="utf-8") as f:
        regime_data = json.load(f)
    with open(exp_dir / "rank_stability.json", "r", encoding="utf-8") as f:
        rank_data = json.load(f)["rank_stabilities"]

    deltas = [d["delta"] for d in delta_data]
    cids = [d["candidate_id"] for d in delta_data]
    
    selected_delta = float(sel_data["selected_delta"])
    prov_delta = float(cfg_data["historical_provisional_delta"])
    
    sel_rho = float(sel_data["selected_metrics"].get("spearman_rho", 0.0))
    prov_rho = float(next((r["spearman_rho"] for r in rank_data if abs(r["delta"] - prov_delta) < 1e-6), 0.0))

    # -------------------------------------------------------------
    # FIGURE 1: DCRI Distribution & Negative Transition Across Delta
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    means = [d["mean"] for d in dist_data]
    medians = [d["median"] for d in dist_data]
    q1s = [d["quantiles"]["q1"] for d in dist_data]
    q3s = [d["quantiles"]["q3"] for d in dist_data]
    p05s = [d["quantiles"]["p05"] for d in dist_data]
    p95s = [d["quantiles"]["p95"] for d in dist_data]

    ax.plot(deltas, means, "o-", color="#1f77b4", label="Mean DCRI", linewidth=2, markersize=6)
    ax.plot(deltas, medians, "s--", color="#2ca02c", label="Median DCRI", linewidth=1.8, markersize=5)
    ax.fill_between(deltas, q1s, q3s, color="#1f77b4", alpha=0.2, label="IQR (Q1 - Q3)")
    ax.fill_between(deltas, p05s, p95s, color="#1f77b4", alpha=0.08, label="90% Interval (P5 - P95)")

    ax.axhline(0.0, color="#d62728", linestyle=":", linewidth=1.5, label="Zero-DCRI Boundary")
    ax.axvline(selected_delta, color="#ff7f0e", linestyle="-.", linewidth=1.5, label=rf"Selected $\delta^*={selected_delta:.2f}$")
    ax.axvline(prov_delta, color="#7f7f7f", linestyle="--", linewidth=1.2, label=rf"Provisional $\delta={prov_delta:.2f}$")

    ax.set_title(r"DCRI Distribution Profiles Across Candidate Penalty Grid $\delta \in [0.0, 1.0]$", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Uncertainty Penalty Multiplier $\delta$", fontsize=10)
    ax.set_ylabel("Decision-Level Risk Index (DCRI)", fontsize=10)
    ax.set_xticks(deltas)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig1_dcri_distributions.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURE 2: Uncertainty Penalty Scaling & Exceedance Rates
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    
    mean_pens = [d["mean_penalty"] for d in delta_data]
    max_pens = [d["max_penalty"] for d in delta_data]
    
    ax1.plot(deltas, mean_pens, "o-", color="#9467bd", label=r"Mean Penalty $\delta \bar{U}_{sum}$", linewidth=2)
    ax1.plot(deltas, max_pens, "v--", color="#8c564b", label=r"Max Penalty $\delta \max(U_{sum})$", linewidth=1.5)
    ax1.axvline(selected_delta, color="#ff7f0e", linestyle="-.", label=rf"Selected $\delta^*={selected_delta:.2f}$")
    ax1.set_title("Mean & Max Uncertainty Penalty Magnitude", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"$\delta$", fontsize=10)
    ax1.set_ylabel("Penalty Magnitude", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    exc_10 = [d["penalty_exceedance_rates"]["gt_0_10"] * 100 for d in delta_data]
    exc_20 = [d["penalty_exceedance_rates"]["gt_0_20"] * 100 for d in delta_data]
    exc_30 = [d["penalty_exceedance_rates"]["gt_0_30"] * 100 for d in delta_data]

    ax2.plot(deltas, exc_10, "s-", color="#17becf", label=r"Penalty $> 0.10$", linewidth=1.8)
    ax2.plot(deltas, exc_20, "^-", color="#bcbd22", label=r"Penalty $> 0.20$", linewidth=1.8)
    ax2.plot(deltas, exc_30, "d-", color="#e377c2", label=r"Penalty $> 0.30$", linewidth=1.8)
    ax2.axvline(selected_delta, color="#ff7f0e", linestyle="-.", label=rf"Selected $\delta^*={selected_delta:.2f}$")
    ax2.set_title("Penalty Threshold Exceedance Proportions (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"$\delta$", fontsize=10)
    ax2.set_ylabel("Packets Exceeding Threshold (%)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(fig_dir / "fig2_uncertainty_penalty_scaling.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURE 3: Negative DCRI Rate Stratified by Modality Cardinality
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    neg_rate_all = [d["negative_rate"] * 100 for d in delta_data]
    neg_mat = regime_data["regime_matrices"]["negative_rate"]
    
    r_rates = [neg_mat["R"][cid] * 100 for cid in cids]
    f_rates = [neg_mat["F"][cid] * 100 for cid in cids]
    c_rates = [neg_mat["C"][cid] * 100 for cid in cids]
    rfc_rates = [neg_mat["RFC"][cid] * 100 for cid in cids]
    
    ax.plot(deltas, neg_rate_all, "k-o", linewidth=2.2, label="Full Cohort (N=500)", zorder=5)
    ax.plot(deltas, rfc_rates, "r--s", linewidth=1.8, label="RFC (Triple Modality, M=3)")
    ax.plot(deltas, r_rates, "b:^", linewidth=1.5, label="Retina Only (M=1)")
    ax.plot(deltas, f_rates, "g:v", linewidth=1.5, label="Foot Only (M=1)")
    ax.plot(deltas, c_rates, "m:d", linewidth=1.5, label="Clinical Only (M=1)")
    
    ax.axhline(35.0, color="#7f7f7f", linestyle=":", label="Feasibility Bound (35%)")
    ax.axvline(selected_delta, color="#ff7f0e", linestyle="-.", linewidth=1.5, label=rf"Selected $\delta^*={selected_delta:.2f}$")
    ax.axvline(prov_delta, color="#7f7f7f", linestyle="--", linewidth=1.2, label=rf"Provisional $\delta={prov_delta:.2f}$")
    
    ax.set_title(r"Negative-DCRI Rate $P(\mathrm{DCRI} < 0)$ Stratified by Availability Regime", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Uncertainty Penalty Multiplier $\delta$", fontsize=10)
    ax.set_ylabel("Negative DCRI Packets (%)", fontsize=10)
    ax.set_xticks(deltas)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig3_negative_dcri_rate_by_regime.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURE 4: Rank Stability with Base Fused Risk R_fusion
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    spearmans = [r["spearman_rho"] for r in rank_data]
    kendalls = [r["kendall_tau"] for r in rank_data]

    ax.plot(deltas, spearmans, "o-", color="#2ca02c", label=r"Spearman Correlation $\rho_s(\mathrm{DCRI}_\delta, R_{\mathrm{fusion}})$", linewidth=2)
    ax.plot(deltas, kendalls, "s--", color="#1f77b4", label=r"Kendall's $\tau(\mathrm{DCRI}_\delta, R_{\mathrm{fusion}})$", linewidth=1.8)

    ax.axhline(0.90, color="#d62728", linestyle=":", label="Rank Stability Floor (0.90)")
    ax.axvline(selected_delta, color="#ff7f0e", linestyle="-.", linewidth=1.5, label=rf"Selected $\delta^*={selected_delta:.2f}$ ($\rho_s = {sel_rho:.4f}$)")
    ax.axvline(prov_delta, color="#7f7f7f", linestyle="--", linewidth=1.2, label=rf"Provisional $\delta={prov_delta:.2f}$ ($\rho_s = {prov_rho:.4f}$)")

    ax.set_title("Decision-Level Rank Stability Relative to Base Fusion Risk", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$\delta$", fontsize=10)
    ax.set_ylabel("Rank Correlation Coefficient", fontsize=10)
    ax.set_xticks(deltas)
    ax.set_ylim(0.75, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig4_rank_stability_spearman.png")
    plt.close(fig)

    print(f"Generated 4 publication-quality figures in {fig_dir}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    generate_volume_13_figures(repo_root)
