import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from .reliability import compute_calibration_bins

def plot_reliability_diagram(
    probs: torch.Tensor,
    labels: torch.Tensor,
    save_path: Path,
    num_bins: int = 15,
    title: str = "Reliability Diagram"
):
    """Plots a reliability diagram."""
    bins = compute_calibration_bins(probs, labels, num_bins=num_bins)
    
    bin_accuracies = bins["bin_accuracies"].numpy()
    bin_confidences = bins["bin_confidences"].numpy()
    bin_counts = bins["bin_counts"].numpy()
    
    # Filter out empty bins
    mask = bin_counts > 0
    bin_accuracies = bin_accuracies[mask]
    bin_confidences = bin_confidences[mask]
    bin_counts = bin_counts[mask]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), dpi=300, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    
    # Empirical calibration (bars)
    widths = bins["bin_edges"][1:].numpy() - bins["bin_edges"][:-1].numpy()
    centers = bins["bin_edges"][:-1].numpy() + widths / 2
    
    # We plot all bins, but set height to 0 where count is 0
    full_accuracies = bins["bin_accuracies"].numpy()
    full_counts = bins["bin_counts"].numpy()
    full_accuracies[full_counts == 0] = 0
    
    ax1.bar(centers, full_accuracies, width=widths, edgecolor="black", color="blue", alpha=0.6, align="center", label="Model")
    
    ax1.set_ylabel("Accuracy")
    ax1.set_title(title)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Bin counts subplot
    ax2.bar(centers, full_counts, width=widths, edgecolor="black", color="green", alpha=0.6, align="center")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.set_xlim([0.0, 1.0])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confidence_histogram(
    probs: torch.Tensor,
    save_path: Path,
    num_bins: int = 15,
    title: str = "Confidence Histogram"
):
    """Plots the distribution of confidences."""
    if probs.ndim == 2:
        confidences, _ = torch.max(probs, dim=1)
    else:
        confidences = probs
        
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    
    ax.hist(confidences.numpy(), bins=num_bins, range=(0, 1), edgecolor="black", color="skyblue")
    
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.set_xlim([0.0, 1.0])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confidence_distribution(
    probs_before: torch.Tensor,
    probs_after: torch.Tensor,
    save_path: Path,
    title: str = "Confidence Distribution (Before vs After)"
):
    """Plots overlapping KDEs or histograms for before/after comparison."""
    if probs_before.ndim == 2:
        conf_before, _ = torch.max(probs_before, dim=1)
        conf_after, _ = torch.max(probs_after, dim=1)
    else:
        conf_before = probs_before
        conf_after = probs_after
        
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    
    ax.hist(conf_before.numpy(), bins=20, range=(0, 1), alpha=0.5, label="Before", color="red")
    ax.hist(conf_after.numpy(), bins=20, range=(0, 1), alpha=0.5, label="After (Calibrated)", color="blue")
    
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim([0.0, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_calibration_curve(
    probs: torch.Tensor,
    labels: torch.Tensor,
    save_path: Path,
    num_bins: int = 15,
    title: str = "Calibration Curve"
):
    """Plots Expected vs Actual Confidence bar charts."""
    bins = compute_calibration_bins(probs, labels, num_bins=num_bins)
    
    bin_accuracies = bins["bin_accuracies"].numpy()
    bin_edges = bins["bin_edges"].numpy()
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    
    # Bar plot for accuracies
    widths = bin_edges[1:] - bin_edges[:-1]
    centers = bin_edges[:-1] + widths / 2
    
    ax.bar(centers, bin_accuracies, width=widths, edgecolor="black", color="blue", alpha=0.6, align="center")
    
    ax.set_xlabel("Expected Confidence")
    ax.set_ylabel("Actual Accuracy")
    ax.set_title(title)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_all_plots(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    labels: torch.Tensor,
    figures_dir: Path,
    num_bins: int = 15
):
    probs_before = F.softmax(logits_before, dim=1)
    probs_after = F.softmax(logits_after, dim=1)
    
    plot_reliability_diagram(probs_before, labels, figures_dir / "reliability_before.png", num_bins, "Reliability Diagram (Before)")
    plot_reliability_diagram(probs_after, labels, figures_dir / "reliability_after.png", num_bins, "Reliability Diagram (After)")
    
    plot_confidence_histogram(probs_before, figures_dir / "confidence_histogram_before.png", num_bins, "Confidence Histogram (Before)")
    plot_confidence_histogram(probs_after, figures_dir / "confidence_histogram_after.png", num_bins, "Confidence Histogram (After)")
    
    plot_confidence_distribution(probs_before, probs_after, figures_dir / "confidence_distribution.png")
    
    plot_calibration_curve(probs_before, labels, figures_dir / "calibration_curve_before.png", num_bins, "Calibration Curve (Before)")
    plot_calibration_curve(probs_after, labels, figures_dir / "calibration_curve_after.png", num_bins, "Calibration Curve (After)")
