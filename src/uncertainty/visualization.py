import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import src.config as config
from src.uncertainty.risk_coverage import compute_risk_coverage_curve

logger = logging.getLogger("Uncertainty_Visualization")

# Set aesthetic theme
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

# Custom Color Palette
PALETTE_CORRECT = "#2ecc71"   # Emerald Green
PALETTE_INCORRECT = "#e74c3c" # Coral Red
PALETTE_NEUT = "#34495e"      # Slate Blue
PALETTE_MC = "#9b59b6"        # Amethyst Purple
PALETTE_DETERM = "#3498db"    # Steel Blue

def generate_distribution_plot(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    xlabel: str,
    save_path: Path,
    is_log: bool = False
):
    """Utility for plotting distribution density curves for Correct vs Incorrect."""
    plt.figure(figsize=(7, 4.5))
    
    # Split
    correct = df[df["is_correct"]][metric_col]
    incorrect = df[~df["is_correct"]][metric_col]
    
    # Plot KDEs
    if len(correct) > 1:
        sns.kdeplot(correct, fill=True, color=PALETTE_CORRECT, label="Correct Predictions", alpha=0.4, linewidth=2)
    else:
        plt.hist(correct, bins=10, alpha=0.4, color=PALETTE_CORRECT, label="Correct Predictions")
        
    if len(incorrect) > 1:
        sns.kdeplot(incorrect, fill=True, color=PALETTE_INCORRECT, label="Incorrect Predictions", alpha=0.4, linewidth=2)
    else:
        plt.hist(incorrect, bins=10, alpha=0.4, color=PALETTE_INCORRECT, label="Incorrect Predictions")
        
    if is_log:
        plt.yscale("log")
        
    plt.title(title, fontweight="bold", pad=15)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    
    # Save in multiple formats
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path.with_suffix(".png"), dpi=300)
    plt.savefig(save_path.with_suffix(".svg"), format="svg")
    plt.close()

def generate_all_uncertainty_plots(
    df: pd.DataFrame,
    convergence_data: dict,
    output_dir: Path
):
    """Generates all 10 requested figures for Volume 8 report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating uncertainty analysis figures in: {output_dir}")
    
    # ----------------------------------------------------
    # Figure 1: Confidence Distribution (Raw vs Calibrated)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw Confidence
    correct_raw = df[df["is_correct"]]["raw_confidence"]
    incorrect_raw = df[~df["is_correct"]]["raw_confidence"]
    if len(correct_raw) > 1:
        sns.kdeplot(correct_raw, fill=True, ax=axes[0], color=PALETTE_CORRECT, label="Correct", alpha=0.4)
    if len(incorrect_raw) > 1:
        sns.kdeplot(incorrect_raw, fill=True, ax=axes[0], color=PALETTE_INCORRECT, label="Incorrect", alpha=0.4)
    axes[0].set_title("Raw Confidence", fontweight="bold")
    axes[0].set_xlabel("Confidence (Max Probability)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    
    # Calibrated Confidence
    correct_calib = df[df["is_correct"]]["calib_confidence"]
    incorrect_calib = df[~df["is_correct"]]["calib_confidence"]
    if len(correct_calib) > 1:
        sns.kdeplot(correct_calib, fill=True, ax=axes[1], color=PALETTE_CORRECT, label="Correct", alpha=0.4)
    if len(incorrect_calib) > 1:
        sns.kdeplot(incorrect_calib, fill=True, ax=axes[1], color=PALETTE_INCORRECT, label="Incorrect", alpha=0.4)
    axes[1].set_title("Calibrated Confidence (T-scaled)", fontweight="bold")
    axes[1].set_xlabel("Calibrated Confidence")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    
    plt.suptitle("Confidence Distributions (Correct vs Incorrect)", fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_confidence_distribution.png", dpi=300)
    plt.savefig(output_dir / "fig1_confidence_distribution.svg", format="svg")
    plt.close()
    
    # ----------------------------------------------------
    # Figure 2: Entropy Distribution (Raw vs Calibrated)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw Entropy
    correct_raw_ent = df[df["is_correct"]]["raw_entropy_norm"]
    incorrect_raw_ent = df[~df["is_correct"]]["raw_entropy_norm"]
    if len(correct_raw_ent) > 1:
        sns.kdeplot(correct_raw_ent, fill=True, ax=axes[0], color=PALETTE_CORRECT, label="Correct", alpha=0.4)
    if len(incorrect_raw_ent) > 1:
        sns.kdeplot(incorrect_raw_ent, fill=True, ax=axes[0], color=PALETTE_INCORRECT, label="Incorrect", alpha=0.4)
    axes[0].set_title("Raw Normalized Entropy", fontweight="bold")
    axes[0].set_xlabel("Normalized Shannon Entropy")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    
    # Calibrated Entropy
    correct_calib_ent = df[df["is_correct"]]["calib_entropy_norm"]
    incorrect_calib_ent = df[~df["is_correct"]]["calib_entropy_norm"]
    if len(correct_calib_ent) > 1:
        sns.kdeplot(correct_calib_ent, fill=True, ax=axes[1], color=PALETTE_CORRECT, label="Correct", alpha=0.4)
    if len(incorrect_calib_ent) > 1:
        sns.kdeplot(incorrect_calib_ent, fill=True, ax=axes[1], color=PALETTE_INCORRECT, label="Incorrect", alpha=0.4)
    axes[1].set_title("Calibrated Normalized Entropy", fontweight="bold")
    axes[1].set_xlabel("Normalized Shannon Entropy (Calibrated)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    
    plt.suptitle("Shannon Entropy Distributions (Correct vs Incorrect)", fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "fig2_entropy_distribution.png", dpi=300)
    plt.savefig(output_dir / "fig2_entropy_distribution.svg", format="svg")
    plt.close()

    # ----------------------------------------------------
    # Figure 3: MC Variance Distribution
    # ----------------------------------------------------
    generate_distribution_plot(
        df=df,
        metric_col="predictive_variance",
        title="Predictive Variance Distribution",
        xlabel="Mean Class Probability Variance across Stochastic Passes",
        save_path=output_dir / "fig3_mc_variance_distribution"
    )

    # ----------------------------------------------------
    # Figure 4: Mutual Information Distribution
    # ----------------------------------------------------
    generate_distribution_plot(
        df=df,
        metric_col="mutual_information",
        title="Mutual Information (Epistemic Uncertainty) Distribution",
        xlabel="Mutual Information",
        save_path=output_dir / "fig4_mutual_information_distribution"
    )

    # ----------------------------------------------------
    # Figure 5: Uncertainty by DR Grade (Multi-Panel)
    # ----------------------------------------------------
    grade_metrics = df.groupby("ground_truth_label")[["calib_entropy_norm", "predictive_variance", "mutual_information"]].mean()
    # Reorder index to align with class sequence
    seq = [c for c in config.CLASS_NAMES if c in grade_metrics.index]
    grade_metrics = grade_metrics.reindex(seq)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Calibrated Entropy (Norm)
    sns.barplot(x=grade_metrics.index, y=grade_metrics["calib_entropy_norm"], ax=axes[0], hue=grade_metrics.index, palette="viridis", legend=False)
    axes[0].set_title("Mean Calib. Entropy (Norm.)", fontweight="bold", pad=10)
    axes[0].set_xlabel("Ground Truth DR Grade")
    axes[0].set_ylabel("Entropy Value")
    axes[0].tick_params(axis='x', rotation=30)
    
    # 2. Predictive Variance
    sns.barplot(x=grade_metrics.index, y=grade_metrics["predictive_variance"], ax=axes[1], hue=grade_metrics.index, palette="viridis", legend=False)
    axes[1].set_title("Mean Predictive Variance", fontweight="bold", pad=10)
    axes[1].set_xlabel("Ground Truth DR Grade")
    axes[1].set_ylabel("Variance Value")
    axes[1].tick_params(axis='x', rotation=30)
    
    # 3. Mutual Information
    sns.barplot(x=grade_metrics.index, y=grade_metrics["mutual_information"], ax=axes[2], hue=grade_metrics.index, palette="viridis", legend=False)
    axes[2].set_title("Mean Mutual Information", fontweight="bold", pad=10)
    axes[2].set_xlabel("Ground Truth DR Grade")
    axes[2].set_ylabel("Mutual Information")
    axes[2].tick_params(axis='x', rotation=30)
    
    plt.suptitle("Uncertainty Metrics by Diabetic Retinopathy Grade", fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "fig5_uncertainty_by_dr_grade.png", dpi=300)
    plt.savefig(output_dir / "fig5_uncertainty_by_dr_grade.svg", format="svg")
    plt.close()

    # ----------------------------------------------------
    # Figure 6: Risk-Coverage Curves
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5.5))
    is_correct = df["is_correct"].values
    
    # Different uncertainty signals
    signals = [
        ("1 - Calib. Confidence", df["calib_confidence"].values, False, PALETTE_DETERM, "-"),
        ("Calib. Entropy (Norm.)", df["calib_entropy_norm"].values, True, PALETTE_NEUT, "-"),
        ("1 - Calib. Margin", df["calib_margin"].values, False, "gray", "--"),
        ("Predictive Entropy", df["predictive_entropy"].values, True, PALETTE_MC, "-"),
        ("Predictive Variance", df["predictive_variance"].values, True, "orange", "-"),
        ("Mutual Information", df["mutual_information"].values, True, "green", "-")
    ]
    
    for label, score, higher_is_more_uncertain, color, ls in signals:
        cov, risk, _, _ = compute_risk_coverage_curve(is_correct, score, higher_is_more_uncertain)
        plt.plot(cov * 100, risk * 100, label=label, color=color, linestyle=ls, linewidth=2)
        
    # Baseline: Dataset Error Rate
    dataset_error = (1.0 - np.mean(is_correct)) * 100
    plt.axhline(dataset_error, color="red", linestyle=":", label=f"Baseline Error ({dataset_error:.2f}%)", linewidth=1.5)
    
    plt.title("Risk-Coverage Curves (Selective Prediction)", fontweight="bold", pad=15)
    plt.xlabel("Coverage (%)")
    plt.ylabel("Risk / Error Rate (%)")
    plt.xlim(10, 100)
    plt.ylim(0, max(dataset_error * 1.5, 30))
    plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "fig6_risk_coverage_curves.png", dpi=300)
    plt.savefig(output_dir / "fig6_risk_coverage_curves.svg", format="svg")
    plt.close()

    # ----------------------------------------------------
    # Figure 7: Uncertainty vs. Confidence Scatter Plot
    # ----------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="calib_confidence",
        y="mutual_information",
        hue="is_correct",
        palette={True: PALETTE_CORRECT, False: PALETTE_INCORRECT},
        alpha=0.6,
        edgecolor="w",
        s=40
    )
    plt.title("Calibrated Confidence vs. Epistemic Uncertainty (MI)", fontweight="bold", pad=15)
    plt.xlabel("Calibrated Confidence (Max Probability)")
    plt.ylabel("Mutual Information (Stochastic Disagreement)")
    
    # Custom Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = ["Correct Prediction" if l == "True" else "Incorrect Prediction" for l in labels]
    plt.legend(handles, new_labels, frameon=True, facecolor="white", edgecolor="none")
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig7_uncertainty_vs_confidence.png", dpi=300)
    plt.savefig(output_dir / "fig7_uncertainty_vs_confidence.svg", format="svg")
    plt.close()

    # ----------------------------------------------------
    # Figure 8: Uncertainty vs. Prediction Error (ROCs)
    # ----------------------------------------------------
    plt.figure(figsize=(7.5, 6))
    is_incorrect = (1 - is_correct).astype(int)
    
    # We plot ROC curves for our error detection methods
    # Helper to plot ROC curve
    def plot_err_roc(score, label, color, higher_is_more_uncertain=True):
        s = np.copy(score)
        if not higher_is_more_uncertain:
            s = 1.0 - s if np.max(s) <= 1.0 else -s
        fpr, tpr, _ = roc_curve(is_incorrect, s)
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, label=f"{label} (AUC = {roc_auc:.4f})", linewidth=1.8)
        
    plot_err_roc(df["calib_confidence"].values, "1 - Calib. Confidence", PALETTE_DETERM, False)
    plot_err_roc(df["calib_entropy_norm"].values, "Calib. Entropy (Norm.)", PALETTE_NEUT, True)
    plot_err_roc(df["predictive_entropy"].values, "MC Predictive Entropy", PALETTE_MC, True)
    plot_err_roc(df["predictive_variance"].values, "MC Predictive Variance", "orange", True)
    plot_err_roc(df["mutual_information"].values, "MC Mutual Information", "green", True)
    
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random (AUC = 0.5000)", linewidth=1)
    
    plt.title("Error Detection ROC Curves", fontweight="bold", pad=15)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Error Detection Sensitivity)")
    plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "fig8_error_detection_roc.png", dpi=300)
    plt.savefig(output_dir / "fig8_error_detection_roc.svg", format="svg")
    plt.close()

    # ----------------------------------------------------
    # Figure 9: MC Convergence Analysis Plot
    # ----------------------------------------------------
    if convergence_data:
        # convergence_data contains lists of predictive_entropy, mean_variance, mutual_information per N
        n_vals = sorted([int(k) for k in convergence_data.keys()])
        
        # Let's compute average metric value across all subset samples for each N
        avg_entropies = []
        avg_variances = []
        avg_mis = []
        
        for n in n_vals:
            avg_entropies.append(np.mean(convergence_data[str(n)]["predictive_entropy"]))
            avg_variances.append(np.mean(convergence_data[str(n)]["mean_variance"]))
            avg_mis.append(np.mean(convergence_data[str(n)]["mutual_info"]))
            
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # 1. Predictive Entropy
        axes[0].plot(n_vals, avg_entropies, marker="o", color=PALETTE_MC, linewidth=2.5, markersize=8)
        axes[0].axvline(x=25, color="red", linestyle="--", alpha=0.8, label="Selected N=25")
        axes[0].set_title("Predictive Entropy Convergence", fontweight="bold", pad=12)
        axes[0].set_xlabel("Number of MC Passes (N)", fontsize=10)
        axes[0].set_ylabel("Mean Predictive Entropy", fontsize=10)
        axes[0].set_xticks(n_vals)
        axes[0].legend(frameon=True, facecolor="white")
        
        # 2. Predictive Variance
        axes[1].plot(n_vals, avg_variances, marker="s", color="orange", linewidth=2.5, markersize=8)
        axes[1].axvline(x=25, color="red", linestyle="--", alpha=0.8, label="Selected N=25")
        axes[1].set_title("Predictive Variance Convergence", fontweight="bold", pad=12)
        axes[1].set_xlabel("Number of MC Passes (N)", fontsize=10)
        axes[1].set_ylabel("Mean Predictive Variance", fontsize=10)
        axes[1].set_xticks(n_vals)
        axes[1].legend(frameon=True, facecolor="white")
        
        # 3. Mutual Information
        axes[2].plot(n_vals, avg_mis, marker="^", color="green", linewidth=2.5, markersize=8)
        axes[2].axvline(x=25, color="red", linestyle="--", alpha=0.8, label="Selected N=25")
        axes[2].set_title("Mutual Information Convergence", fontweight="bold", pad=12)
        axes[2].set_xlabel("Number of MC Passes (N)", fontsize=10)
        axes[2].set_ylabel("Mean Mutual Information", fontsize=10)
        axes[2].set_xticks(n_vals)
        axes[2].legend(frameon=True, facecolor="white")
        
        plt.suptitle("Stochastic Metric Convergence vs. Number of MC Passes", fontweight="bold", y=0.98, fontsize=13)
        plt.tight_layout()
        plt.savefig(output_dir / "fig9_mc_convergence.png", dpi=300)
        plt.savefig(output_dir / "fig9_mc_convergence.svg", format="svg")
        plt.close()
    else:
        logger.warning("No convergence data available. Skipping convergence plot.")

    # ----------------------------------------------------
    # Figure 10: Representative Cases Overview Grid
    # ----------------------------------------------------
    # Generate a graphical table layout using matplotlib
    plt.figure(figsize=(12, 6))
    plt.axis("off")
    
    # We display a summary of key cases (e.g. 5 cases)
    # Filter cases from df
    # Let's check if the 'selected_reason' column is present
    if "selected_reason" in df.columns:
        show_cases = df.dropna(subset=["selected_reason"]).head(6)
        
        cell_text = []
        columns = ["Image ID", "Reason", "GT", "Pred", "Correct?", "Calib. Conf.", "Calib. Entropy", "MC Variance", "Mutual Info"]
        
        for _, row in show_cases.iterrows():
            correct_str = "Yes" if row["is_correct"] else "No"
            cell_text.append([
                row["image_id"],
                row["selected_reason"][:30],
                row["ground_truth_label"],
                row["prediction_label"],
                correct_str,
                f"{row['calib_confidence']:.2%}",
                f"{row['calib_entropy_norm']:.4f}",
                f"{row['predictive_variance']:.4f}",
                f"{row['mutual_information']:.4f}"
            ])
            
        if cell_text:
            table = plt.table(
                cellText=cell_text,
                colLabels=columns,
                loc="center",
                cellLoc="center",
                colColours=["#2c3e50"] * len(columns)
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 2.0)
            
            # Format text colors for headers
            for key, cell in table.get_celld().items():
                if key[0] == 0:
                    cell.get_text().set_color("white")
                    cell.get_text().set_weight("bold")
                    
            plt.title("Representative Uncertainty Cases Summary Grid", fontweight="bold", pad=20)
            plt.tight_layout()
            plt.savefig(output_dir / "fig10_representative_cases.png", dpi=300)
            plt.savefig(output_dir / "fig10_representative_cases.svg", format="svg")
            
    plt.close()
    
    logger.info("All 10 figures successfully generated.")
