import numpy as np
import torch
from typing import Tuple, Dict

def compute_calibration_bins(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15
) -> Dict[str, torch.Tensor]:
    """
    Computes reliability bins for a set of predictions and true labels.

    Args:
        probs: (N, C) tensor of predicted probabilities or (N,) for top-1 probs.
        labels: (N,) tensor of ground truth class indices.
        num_bins: Number of confidence bins.

    Returns:
        Dictionary containing expected accuracies, confidences, and bin counts.
    """
    if probs.ndim == 2:
        confidences, predictions = torch.max(probs, dim=1)
    else:
        confidences = probs
        # We need predictions... if it's 1D, we assume the user already gave top-1 confidences
        # but for accuracy we also need correct predictions. This function assumes 2D probs.
        raise ValueError("Expected probs to be 2-dimensional (N, C)")

    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0.0, 1.0, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accuracies = torch.zeros(num_bins)
    bin_confidences = torch.zeros(num_bins)
    bin_counts = torch.zeros(num_bins)

    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            bin_accuracies[i] = accuracies[in_bin].float().mean()
            bin_confidences[i] = confidences[in_bin].mean()
        bin_counts[i] = in_bin.sum()

    return {
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "bin_edges": bin_boundaries
    }
