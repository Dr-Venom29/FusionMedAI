import torch
import torch.nn.functional as F
from typing import Dict
from .reliability import compute_calibration_bins

def compute_accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    _, preds = torch.max(probs, dim=1)
    return (preds == labels).float().mean().item()

def compute_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return F.cross_entropy(logits, labels).item()

def compute_brier_score(probs: torch.Tensor, labels: torch.Tensor, num_classes: int = 5) -> float:
    labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
    return F.mse_loss(probs, labels_one_hot).item()

def compute_ece_mce(probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 15) -> tuple[float, float]:
    bins = compute_calibration_bins(probs, labels, num_bins=num_bins)
    
    bin_accuracies = bins["bin_accuracies"]
    bin_confidences = bins["bin_confidences"]
    bin_counts = bins["bin_counts"]
    
    total_samples = len(labels)
    
    # ECE
    ece = torch.sum(torch.abs(bin_accuracies - bin_confidences) * (bin_counts / total_samples))
    
    # MCE (only for bins with samples)
    mask = bin_counts > 0
    if mask.sum() > 0:
        mce = torch.max(torch.abs(bin_accuracies[mask] - bin_confidences[mask]))
    else:
        mce = torch.tensor(0.0)
        
    return ece.item(), mce.item()

def compute_all_metrics(logits: torch.Tensor, labels: torch.Tensor, num_bins: int = 15, num_classes: int = 5) -> Dict[str, float]:
    probs = F.softmax(logits, dim=1)
    
    accuracy = compute_accuracy(probs, labels)
    nll = compute_nll(logits, labels)
    brier = compute_brier_score(probs, labels, num_classes=num_classes)
    ece, mce = compute_ece_mce(probs, labels, num_bins=num_bins)
    
    confidences, _ = torch.max(probs, dim=1)
    mean_confidence = confidences.mean().item()
    
    overconfidence_gap = mean_confidence - accuracy
    
    return {
        "Accuracy": accuracy,
        "NLL": nll,
        "Brier": brier,
        "ECE": ece,
        "MCE": mce,
        "Mean Confidence": mean_confidence,
        "Overconfidence Gap": overconfidence_gap
    }
