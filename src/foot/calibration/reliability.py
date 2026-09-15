import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

WAGNER_CLASSES = {
    0: "Grade 1",
    1: "Grade 2",
    2: "Grade 3",
    3: "Grade 4"
}

def plot_reliability_diagram(
    probs: torch.Tensor,
    labels: torch.Tensor,
    title: str = "Reliability Diagram",
    num_bins: int = 10,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plots a single reliability diagram showing average confidence vs actual accuracy.
    Empty bins (count == 0) are plotted as NaN (omitted from bars) to prevent misrepresenting zero data as 0% accuracy.
    """
    probs = probs.detach()
    labels = labels.detach()
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = (predictions == labels).float()
    
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_gaps = []
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_size = in_bin.sum().item()
        
        if bin_size > 0:
            acc = accuracies[in_bin].mean().item()
            conf = confidences[in_bin].mean().item()
            bin_accs.append(acc)
            bin_confs.append(conf)
            bin_gaps.append(abs(conf - acc))
        else:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)
            bin_gaps.append(np.nan)
            
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    
    # Bar plot of bin accuracies
    bin_centers = ((bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0).detach().cpu().numpy()
    widths = 1.0 / num_bins
    
    ax.bar(bin_centers, bin_accs, width=widths, alpha=0.6, color="royalblue", edgecolor="black", label="Outputs")
    ax.bar(bin_centers, bin_gaps, bottom=bin_accs, width=widths, alpha=0.3, color="crimson", edgecolor="red", label="Calibration Gap")
    
    ax.set_xlabel("Confidence", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

def plot_reliability_comparison(
    method_probs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    num_bins: int = 10,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plots a combined comparison grid of reliability diagrams.
    Empty bins are omitted (NaN) to avoid visual artifacts.
    """
    fig, axes = plt.subplots(1, len(method_probs), figsize=(5 * len(method_probs), 5))
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_centers = ((bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0).detach().cpu().numpy()
    widths = 1.0 / num_bins
    labels = labels.detach()
    
    for idx, (m_name, probs) in enumerate(method_probs.items()):
        probs = probs.detach()
        ax = axes[idx]
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = (predictions == labels).float()
        
        bin_accs = []
        bin_gaps = []
        for i in range(num_bins):
            b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences >= b_low) & (confidences <= b_high) if i == num_bins - 1 else (confidences >= b_low) & (confidences < b_high)
            bin_size = in_bin.sum().item()
            if bin_size > 0:
                acc = accuracies[in_bin].mean().item()
                conf = confidences[in_bin].mean().item()
                bin_accs.append(acc)
                bin_gaps.append(abs(conf - acc))
            else:
                bin_accs.append(np.nan)
                bin_gaps.append(np.nan)
                
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.bar(bin_centers, bin_accs, width=widths, alpha=0.6, color="royalblue", edgecolor="black")
        ax.bar(bin_centers, bin_gaps, bottom=bin_accs, width=widths, alpha=0.3, color="crimson", edgecolor="red")
        
        ax.set_xlabel("Confidence", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=10, fontweight="bold")
        ax.set_title(m_name, fontsize=11, fontweight="bold")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, linestyle=":", alpha=0.5)
        
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

def plot_confidence_distribution(
    method_probs: Dict[str, torch.Tensor],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plots overlaid confidence histograms comparing raw vs calibrated predictions.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["crimson", "royalblue", "forestgreen"]
    
    for idx, (m_name, probs) in enumerate(method_probs.items()):
        probs = probs.detach()
        confidences, _ = torch.max(probs, dim=1)
        conf_np = confidences.detach().cpu().numpy()
        color = colors[idx % len(colors)]
        ax.hist(conf_np, bins=20, range=(0, 1), alpha=0.4, label=m_name, color=color, edgecolor="black")
        
    ax.set_xlabel("Prediction Confidence", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sample Count", fontsize=11, fontweight="bold")
    ax.set_title("Confidence Distribution Comparison", fontsize=12, fontweight="bold")
    ax.set_xlim([0, 1])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

def plot_classwise_reliability(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 4,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plots per-grade confidence diagnostic curves for Wagner grades 1 through 4.
    Empty bins are omitted (NaN) to prevent visual artifacts.
    """
    probs = probs.detach()
    labels = labels.detach()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    bin_boundaries = torch.linspace(0, 1, 11)
    bin_centers = ((bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0).detach().cpu().numpy()
    
    for c in range(num_classes):
        ax = axes[c]
        mask = (labels == c)
        if mask.sum() > 0:
            c_probs = probs[mask]
            c_labels = labels[mask]
            confidences, predictions = torch.max(c_probs, dim=1)
            accuracies = (predictions == c_labels).float()
            
            bin_accs = []
            for i in range(10):
                b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
                in_bin = (confidences >= b_low) & (confidences <= b_high) if i == 9 else (confidences >= b_low) & (confidences < b_high)
                if in_bin.sum() > 0:
                    bin_accs.append(accuracies[in_bin].mean().item())
                else:
                    bin_accs.append(np.nan)
            ax.plot([0, 1], [0, 1], "k--", label="Perfect")
            ax.bar(bin_centers, bin_accs, width=0.1, alpha=0.6, color="teal", edgecolor="black")
            ax.set_title(f"Wagner {WAGNER_CLASSES[c]} (N={mask.sum().item()})", fontsize=11, fontweight="bold")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.grid(True, linestyle=":", alpha=0.5)
            
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
