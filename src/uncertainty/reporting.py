import sys
import os
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import src.config as config
from src.uncertainty.metrics import evaluate_error_detection
from src.uncertainty.risk_coverage import compute_risk_coverage_curve, evaluate_selective_prediction
from src.uncertainty.utils import get_git_commit_hash

logger = logging.getLogger("Uncertainty_Reporting")

def df_to_markdown_file(df: pd.DataFrame, filepath: Path):
    """Converts a pandas DataFrame to markdown and writes it to a file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(df.to_markdown(index=False))

def generate_all_tables_and_reports(
    df: pd.DataFrame,
    mc_probs: np.ndarray,
    validation_report: dict,
    verification_info: dict,
    output_dir: Path,
    exp_dir: Path
):
    """
    Compiles all results, generates 6 CSV and Markdown tables, saves final metrics.json,
    manifest.json, and copies them to the experiments output directory.
    """
    output_tables_dir = output_dir / "tables"
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating tables and reports in: {output_tables_dir}")
    
    is_correct = df["is_correct"].values
    is_incorrect = 1 - is_correct.astype(int)
    
    # ----------------------------------------------------
    # Table 1: uncertainty_summary
    # ----------------------------------------------------
    summary_metrics = ["raw_confidence", "calib_confidence", "raw_entropy_norm", "calib_entropy_norm", 
                       "predictive_entropy", "predictive_variance", "mutual_information"]
    summary_data = []
    for metric in summary_metrics:
        if metric in df.columns:
            summary_data.append({
                "Metric": metric,
                "Mean": float(df[metric].mean()),
                "Std": float(df[metric].std()),
                "Min": float(df[metric].min()),
                "Median": float(df[metric].median()),
                "Max": float(df[metric].max())
            })
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_tables_dir / "uncertainty_summary.csv", index=False)
    df_to_markdown_file(df_summary, output_tables_dir / "uncertainty_summary.md")
    
    # ----------------------------------------------------
    # Table 2: uncertainty_by_class
    # ----------------------------------------------------
    # Group by class label
    class_groups = sorted(df["ground_truth"].unique())
    class_data = []
    for c in class_groups:
        sub = df[df["ground_truth"] == c]
        class_data.append({
            "DR Grade": config.CLASS_NAMES[c],
            "Grade Idx": c,
            "Sample Count": len(sub),
            "Error Rate": float(1.0 - sub["is_correct"].mean()),
            "Mean Calib. Confidence": float(sub["calib_confidence"].mean()),
            "Mean Calib. Entropy (Norm)": float(sub["calib_entropy_norm"].mean()),
            "Mean MC Entropy": float(sub["predictive_entropy"].mean()),
            "Mean MC Variance": float(sub["predictive_variance"].mean()),
            "Mean Mutual Info": float(sub["mutual_information"].mean())
        })
    df_class = pd.DataFrame(class_data)
    df_class.to_csv(output_tables_dir / "uncertainty_by_class.csv", index=False)
    df_to_markdown_file(df_class, output_tables_dir / "uncertainty_by_class.md")

    # ----------------------------------------------------
    # Table 3: correct_vs_incorrect
    # ----------------------------------------------------
    correct_sub = df[df["is_correct"]]
    incorrect_sub = df[~df["is_correct"]]
    
    ci_data = [
        {
            "Subset": "Correct Predictions",
            "Count": len(correct_sub),
            "Mean Calib. Conf.": float(correct_sub["calib_confidence"].mean()) if len(correct_sub) > 0 else 0.0,
            "Std Calib. Conf.": float(correct_sub["calib_confidence"].std()) if len(correct_sub) > 1 else 0.0,
            "Mean Calib. Entropy": float(correct_sub["calib_entropy_norm"].mean()) if len(correct_sub) > 0 else 0.0,
            "Mean MC Variance": float(correct_sub["predictive_variance"].mean()) if len(correct_sub) > 0 else 0.0,
            "Mean Mutual Info": float(correct_sub["mutual_information"].mean()) if len(correct_sub) > 0 else 0.0
        },
        {
            "Subset": "Incorrect Predictions",
            "Count": len(incorrect_sub),
            "Mean Calib. Conf.": float(incorrect_sub["calib_confidence"].mean()) if len(incorrect_sub) > 0 else 0.0,
            "Std Calib. Conf.": float(incorrect_sub["calib_confidence"].std()) if len(incorrect_sub) > 1 else 0.0,
            "Mean Calib. Entropy": float(incorrect_sub["calib_entropy_norm"].mean()) if len(incorrect_sub) > 0 else 0.0,
            "Mean MC Variance": float(incorrect_sub["predictive_variance"].mean()) if len(incorrect_sub) > 0 else 0.0,
            "Mean Mutual Info": float(incorrect_sub["mutual_information"].mean()) if len(incorrect_sub) > 0 else 0.0
        }
    ]
    df_ci = pd.DataFrame(ci_data)
    df_ci.to_csv(output_tables_dir / "correct_vs_incorrect.csv", index=False)
    df_to_markdown_file(df_ci, output_tables_dir / "correct_vs_incorrect.md")

    # ----------------------------------------------------
    # Table 4: mc_dropout_summary
    # ----------------------------------------------------
    # Summary of MC Dropout run
    mcd_summary_data = [{
        "MC Passes Count (N)": mc_probs.shape[1],
        "Discovered Dropout Layers": ", ".join([f"{l[0]} (p={l[1]})" for l in validation_report["dropout_layers"]]),
        "Stochasticity Validation Check": "PASSED" if validation_report["probability_variance_nonzero"] else "FAILED",
        "Sanity Validation Variance": float(validation_report["aggregate_variance"]),
        "Sanity Validation Prob. Std": float(validation_report["mean_probability_std"]),
        "Test Set Average Predictive Var": float(df["predictive_variance"].mean()),
        "Test Set Average Mutual Info": float(df["mutual_information"].mean())
    }]
    df_mcd = pd.DataFrame(mcd_summary_data)
    df_mcd.to_csv(output_tables_dir / "mc_dropout_summary.csv", index=False)
    df_to_markdown_file(df_mcd, output_tables_dir / "mc_dropout_summary.md")

    # ----------------------------------------------------
    # Table 5: uncertainty_error_detection
    # ----------------------------------------------------
    methods = [
        ("1 - Calibrated Confidence", df["calib_confidence"].values, False),
        ("Calibrated Entropy (Norm)", df["calib_entropy_norm"].values, True),
        ("1 - Calibrated Margin", df["calib_margin"].values, False),
        ("MC Predictive Entropy", df["predictive_entropy"].values, True),
        ("MC Predictive Variance", df["predictive_variance"].values, True),
        ("MC Mutual Information", df["mutual_information"].values, True)
    ]
    
    detection_data = []
    for name, score, higher_is_more_uncertain in methods:
        auroc, auprc = evaluate_error_detection(is_incorrect, score, higher_is_more_uncertain)
        _, _, aurc, e_aurc = compute_risk_coverage_curve(is_correct, score, higher_is_more_uncertain)
        detection_data.append({
            "Method": name,
            "AUROC": auroc,
            "AUPRC": auprc,
            "AURC": aurc,
            "E-AURC": e_aurc
        })
    df_detection = pd.DataFrame(detection_data)
    df_detection.to_csv(output_tables_dir / "uncertainty_error_detection.csv", index=False)
    df_to_markdown_file(df_detection, output_tables_dir / "uncertainty_error_detection.md")

    # ----------------------------------------------------
    # Table 6: risk_coverage
    # ----------------------------------------------------
    # Risk (error rate) at coverages: 100%, 90%, 80%, 70%, 50%
    coverages_to_eval = [100, 90, 80, 70, 50]
    rc_rows = []
    
    for name, score, higher_is_more_uncertain in methods:
        sel_results = evaluate_selective_prediction(is_correct, score, higher_is_more_uncertain)
        row = {"Method": name}
        for cov in coverages_to_eval:
            row[f"Risk @ {cov}% Coverage"] = float(sel_results["milestones"][f"risk_at_{cov}"])
        rc_rows.append(row)
        
    df_rc = pd.DataFrame(rc_rows)
    df_rc.to_csv(output_tables_dir / "risk_coverage.csv", index=False)
    df_to_markdown_file(df_rc, output_tables_dir / "risk_coverage.md")

    # ----------------------------------------------------
    # Save uncertainty_metrics.json
    # ----------------------------------------------------
    # Write JSON summaries
    best_method_idx = df_detection["AUROC"].idxmax()
    best_method = df_detection.iloc[best_method_idx]
    
    metrics = {
        "Total images evaluated": len(df),
        "Correct predictions": int(is_correct.sum()),
        "Incorrect predictions": int(is_incorrect.sum()),
        "Overall Error Rate": float(np.mean(is_incorrect)),
        "Mean Calibrated Confidence": float(df["calib_confidence"].mean()),
        "Mean Mutual Information": float(df["mutual_information"].mean()),
        "Best Error Detection Method": str(best_method["Method"]),
        "Best Error Detection AUROC": float(best_method["AUROC"]),
        "Best Error Detection AUPRC": float(best_method["AUPRC"]),
        "Best Error Detection AURC": float(best_method["AURC"]),
        "Risk at 50% Coverage (Best Method)": float(df_rc.loc[best_method_idx, "Risk @ 50% Coverage"]),
        "Generation timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_dir / "uncertainty_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # ----------------------------------------------------
    # Save manifest.json
    # ----------------------------------------------------
    import torchvision
    import torch
    manifest = {
        "Dataset version": "APTOS 2019 Blindness Detection",
        "Dataset test split checksum": verification_info.get("calibration_manifest_sha256", "N/A"),
        "Image size": config.IMAGE_SIZE,
        "Calibration source": verification_info.get("temperature_source", "N/A"),
        "Temperature used": verification_info.get("temperature", 1.0),
        "Checkpoint SHA256": verification_info.get("checkpoint_sha256", "N/A"),
        "Git commit hash": get_git_commit_hash(),
        "Uncertainty pipeline version": config.UNCERTAINTY_VERSION,
        "MC stochastic passes count": mc_probs.shape[1],
        "Python version": sys.version,
        "Torch version": torch.__version__,
        "Torchvision version": torchvision.__version__,
        "CUDA version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "OS": os.name if hasattr(os, "name") else "N/A"
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
        
    # Copy to experiment output directory (e.g. experiments/uncertainty/v001_mc_dropout/)
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_logs_dir = exp_dir / "logs"
    exp_logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config and copy key jsons
    with open(exp_dir / "config.json", "w") as f:
        # Save key parameters
        json.dump({
            "model": "efficientnet_b3",
            "checkpoint": verification_info.get("checkpoint_path"),
            "temperature": verification_info.get("temperature"),
            "mc_passes": mc_probs.shape[1],
            "timestamp": time.time()
        }, f, indent=4)
        
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    with open(exp_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
        
    logger.info(f"Summaries and JSON metadata copied to experiments directory: {exp_dir}")
