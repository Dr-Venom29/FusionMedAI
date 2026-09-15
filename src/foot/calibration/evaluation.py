import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from src.foot.calibration.metrics import compute_calibration_metrics, compute_per_class_calibration

def evaluate_calibrated_model(
    logits: torch.Tensor,
    labels: torch.Tensor,
    method_name: str,
    num_bins: int = 10,
    num_classes: int = 4
) -> Dict[str, Any]:
    """
    Evaluates a set of logits (raw or scaled) against ground-truth labels.
    """
    metrics = compute_calibration_metrics(logits, labels, num_bins=num_bins, num_classes=num_classes)
    metrics["method_name"] = method_name
    return metrics

def generate_calibration_results_dataframe(
    raw_metrics: Dict[str, Any],
    temp_metrics: Dict[str, Any],
    vec_metrics: Dict[str, Any]
) -> pd.DataFrame:
    """
    Generates a structured pandas DataFrame summarizing evaluation across calibration methods.
    """
    records = []
    for m in [raw_metrics, temp_metrics, vec_metrics]:
        records.append({
            "Method": m["method_name"],
            "Accuracy": m["Accuracy"],
            "Macro_F1": m["Macro F1"],
            "Balanced_Accuracy": m["Balanced Accuracy"],
            "Macro_ROC_AUC": m["Macro ROC-AUC"],
            "NLL": m["NLL"],
            "Brier_Score": m["Brier"],
            "ECE": m["ECE"],
            "MCE": m["MCE"],
            "Mean_Confidence": m["Mean Confidence"],
            "Overconfidence_Gap": m["Overconfidence Gap"]
        })
    return pd.DataFrame(records)

def generate_classwise_calibration_dataframe(
    probs: torch.Tensor,
    labels: torch.Tensor,
    method_name: str,
    num_classes: int = 4
) -> pd.DataFrame:
    """
    Generates per-class calibration metrics CSV dataframe.
    """
    per_class = compute_per_class_calibration(probs, labels, num_classes=num_classes)
    records = []
    for grade_name, stats in per_class.items():
        records.append({
            "Method": method_name,
            "Class": grade_name,
            "Sample_Count": stats["sample_count"],
            "Accuracy": stats["accuracy"],
            "Mean_Confidence": stats["mean_confidence"],
            "Overconfidence_Gap": stats["overconfidence_gap"],
            "ECE": stats["ece"]
        })
    return pd.DataFrame(records)
