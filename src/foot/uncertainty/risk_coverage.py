import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Any, List

def compute_risk_coverage_curve(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    num_thresholds: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Risk-Coverage curve (rejection rate vs error rate / risk).
    
    Args:
        uncertainties: Array of shape [S] (higher means higher uncertainty)
        errors: Binary array of shape [S] (1 if prediction incorrect, 0 if correct)
        
    Returns:
        Tuple of (coverages, risk_rates)
    """
    sorted_indices = np.argsort(uncertainties) # Ascending order (lowest uncertainty retained first)
    sorted_errors = errors[sorted_indices]
    
    total_samples = len(uncertainties)
    coverages = np.linspace(1.0 / total_samples, 1.0, num_thresholds)
    risk_rates = []
    
    for cov in coverages:
        k = int(np.ceil(cov * total_samples))
        selected_errors = sorted_errors[:k]
        risk = np.mean(selected_errors)
        risk_rates.append(risk)
        
    return np.array(coverages), np.array(risk_rates)


def evaluate_risk_coverage(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    coverage_levels: List[float] = [1.0, 0.90, 0.80, 0.70, 0.60, 0.50]
) -> pd.DataFrame:
    """
    Evaluates error rate (risk) and accuracy at key discrete coverage percentages.
    
    Args:
        uncertainties: Uncertainty scores per sample [S]
        errors: Ground truth error indicator per sample [S]
        coverage_levels: Fraction of retained data points
        
    Returns:
        DataFrame with columns ['coverage_pct', 'retained_samples', 'error_rate', 'accuracy']
    """
    sorted_indices = np.argsort(uncertainties) # Ascending: retain lowest uncertainty first
    sorted_errors = errors[sorted_indices]
    total_samples = len(uncertainties)
    
    records = []
    for cov in coverage_levels:
        k = max(1, int(np.round(cov * total_samples)))
        retained_errors = sorted_errors[:k]
        err_rate = float(np.mean(retained_errors))
        acc = float(1.0 - err_rate)
        records.append({
            "coverage_pct": round(cov * 100.0, 1),
            "retained_samples": k,
            "error_rate": round(err_rate, 4),
            "accuracy": round(acc, 4)
        })
        
    return pd.DataFrame(records)
