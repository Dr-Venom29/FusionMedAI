"""
Standardized Expected Calibration Error (ECE) Computation (Phase C11.3).
Computes 10-bin Equal-Frequency (Quantile) ECE on calibrated validation predictions.
Strictly validation-only computation.
"""

from typing import Union, List, Dict, Any, Tuple
import numpy as np


def compute_equal_frequency_ece(
    probs: Union[np.ndarray, List[Any]],
    labels: Union[np.ndarray, List[int]],
    n_bins: int = 10,
    is_multiclass: bool = False
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE) using equal-frequency (quantile) binning.
    
    Args:
        probs: Calibrated probabilities.
               If is_multiclass is True: shape (N, K) posterior class probabilities.
               If is_multiclass is False: shape (N,) positive-class probabilities.
        labels: Ground truth class integer indices.
        n_bins: Number of equal-frequency bins (fixed to 10 by protocol freeze).
        is_multiclass: Whether task is multi-class or binary.
        
    Returns:
        (ece: float, calibration_score: float, bin_details: List[Dict])
    """
    probs_arr = np.asarray(probs, dtype=float)
    labels_arr = np.asarray(labels, dtype=int).ravel()
    
    if is_multiclass:
        if probs_arr.ndim != 2:
            raise ValueError(f"Multiclass probabilities must be 2D, got shape {probs_arr.shape}")
        confidences = np.max(probs_arr, axis=1)
        predictions = np.argmax(probs_arr, axis=1)
        accuracies = (predictions == labels_arr).astype(float)
    else:
        confidences = probs_arr.ravel()
        accuracies = labels_arr.astype(float)
        
    n_samples = len(confidences)
    if n_samples == 0:
        return 0.0, 1.0, []
        
    # Quantile bin edges
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(confidences, quantiles)
    bin_edges[0] -= 1e-7
    bin_edges[-1] += 1e-7
    
    ece = 0.0
    bin_details = []
    
    for i in range(n_bins):
        lower = float(bin_edges[i])
        upper = float(bin_edges[i + 1])
        mask = (confidences > lower) & (confidences <= upper)
        bin_count = int(np.sum(mask))
        
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            gap = abs(bin_acc - bin_conf)
            weight = bin_count / n_samples
            ece += weight * gap
        else:
            bin_acc = 0.0
            bin_conf = (lower + upper) / 2.0
            gap = 0.0
            weight = 0.0
            
        bin_details.append({
            "bin_index": i,
            "range": [round(lower, 6), round(upper, 6)],
            "count": bin_count,
            "accuracy": round(bin_acc, 6),
            "confidence": round(bin_conf, 6),
            "gap": round(gap, 6),
            "weight": round(weight, 6)
        })
        
    ece = float(np.clip(ece, 0.0, 1.0))
    cal_score = float(np.clip(1.0 - ece, 0.0, 1.0))
    return ece, cal_score, bin_details
