import sys
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import src.config as config

logger = logging.getLogger("Uncertainty_MCDropout")

def discover_dropout_layers(model: nn.Module) -> List[Tuple[str, float]]:
    """Programmatically inspects the model and returns all nn.Dropout layers and their probabilities."""
    dropout_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            dropout_layers.append((name, float(module.p)))
    return dropout_layers

def enable_only_dropout(model: nn.Module) -> List[Tuple[str, float]]:
    """
    Sets only nn.Dropout layers to train() mode, keeping everything else (including BatchNorm)
    in eval() mode.
    """
    model.eval() # First freeze everything
    dropout_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.train() # Enable dropout stochasticity
            dropout_layers.append((name, float(module.p)))
    return dropout_layers

def run_stochasticity_validation(
    model: nn.Module,
    test_loader: DataLoader,
    temperature: float,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Runs a 5-pass stochastic validation check on a small sample of test images.
    Verifies that logits and probabilities differ across passes and that variance is non-zero.
    Aborts immediately if variance is effectively zero.
    Saves a report to results/uncertainty/mc_dropout_validation.json.
    """
    logger.info("Running mandatory stochasticity validation check...")
    
    # 1. Programmatically discover dropout layers
    dropout_layers = discover_dropout_layers(model)
    logger.info(f"Discovered dropout layers: {dropout_layers}")
    
    if not dropout_layers:
        raise RuntimeError("No nn.Dropout layers found in the model! MC Dropout is invalid for this architecture.")
        
    # Get a small batch of images (e.g. 5 images)
    dataset = test_loader.dataset
    # Let's take the first 5 indices
    sample_indices = list(range(min(5, len(dataset))))
    subset_dataset = Subset(dataset, sample_indices)
    subset_loader = DataLoader(subset_dataset, batch_size=len(sample_indices), shuffle=False)
    
    # Load first batch
    for inputs, _ in subset_loader:
        sample_inputs = inputs.to(device)
        break
        
    # Enable only dropout
    enable_only_dropout(model)
    
    # Run 5 passes
    num_passes = 5
    all_logits_passes = []
    all_probs_passes = []
    
    with torch.no_grad():
        for t in range(num_passes):
            # Forward pass
            logits = model(sample_inputs)
            # Calibrate logits before softmax
            calibrated_logits = logits / temperature
            probs = F.softmax(calibrated_logits, dim=1)
            
            all_logits_passes.append(logits.cpu().numpy())
            all_probs_passes.append(probs.cpu().numpy())
            
    # Convert list of arrays of shape (5, C) to array of shape (num_passes, 5, C)
    all_logits_passes = np.array(all_logits_passes) # (5, 5, C)
    all_probs_passes = np.array(all_probs_passes)   # (5, 5, C)
    
    # Calculate differences and variance across passes
    # Variance across the passes dimension (axis 0)
    variance_logits = np.var(all_logits_passes, axis=0) # shape (5, C)
    variance_probs = np.var(all_probs_passes, axis=0)   # shape (5, C)
    
    mean_logits_std = np.mean(np.std(all_logits_passes, axis=0))
    mean_probs_std = np.mean(np.std(all_probs_passes, axis=0))
    
    logits_differ = bool(mean_logits_std > 1e-6)
    probs_differ = bool(mean_probs_std > 1e-6)
    variance_nonzero = bool(np.mean(variance_probs) > 1e-7)
    
    validation_report = {
        "dropout_detected": True,
        "dropout_layers": dropout_layers,
        "passes_tested": num_passes,
        "mean_logits_std": float(mean_logits_std),
        "mean_probability_std": float(mean_probs_std),
        "logits_differ": logits_differ,
        "probabilities_differ": probs_differ,
        "probability_variance_nonzero": variance_nonzero,
        "aggregate_variance": float(np.mean(variance_probs)),
        "timestamp": time.time()
    }
    
    # Save the report
    output_path = config.UNCERTAINTY_RESULTS_DIR / "mc_dropout_validation.json"
    with open(output_path, "w") as f:
        json.dump(validation_report, f, indent=4)
        
    logger.info(f"Stochasticity validation report saved to {output_path}")
    logger.info(f"Stochasticity check results: Logits differ = {logits_differ}, Probs differ = {probs_differ}, Var > 0 = {variance_nonzero}")
    
    if not (logits_differ and probs_differ and variance_nonzero):
        msg = "STOCHASTICITY FAILURE: Stochastic passes produce effectively identical logits/probabilities. The dropout layer may not be working in training mode or is bypassed. Aborting!"
        logger.error(msg)
        raise RuntimeError(msg)
        
    return validation_report

def run_mc_dropout_inference(
    model: nn.Module,
    test_loader: DataLoader,
    temperature: float,
    num_passes: int = 25,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs stochastic inference using MC Dropout on the complete test set.
    For every MC pass: logits -> logits / T -> softmax
    
    Returns:
      - all_mc_probs: np.ndarray of shape (num_samples, num_passes, num_classes)
      - ground_truth: np.ndarray of shape (num_samples,)
    """
    logger.info(f"Starting MC Dropout inference with N = {num_passes} passes...")
    enable_only_dropout(model)
    
    num_samples = len(test_loader.dataset)
    num_classes = config.NUM_CLASSES
    
    # Initialize array to store all passes' probabilities
    # Shape: (num_samples, num_passes, num_classes)
    all_mc_probs = np.zeros((num_samples, num_passes, num_classes))
    all_targets = np.zeros(num_samples, dtype=int)
    
    start_idx = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="MC Inference")):
            inputs = inputs.to(device)
            batch_size = len(inputs)
            end_idx = start_idx + batch_size
            
            # Run N forward passes on this batch in memory
            for pass_idx in range(num_passes):
                logits = model(inputs)
                
                # Apply temperature scaling before softmax
                calibrated_logits = logits / temperature
                probs = F.softmax(calibrated_logits, dim=1).cpu().numpy()
                
                all_mc_probs[start_idx:end_idx, pass_idx, :] = probs
                
            all_targets[start_idx:end_idx] = targets.numpy()
            start_idx = end_idx
                
    return all_mc_probs, all_targets

def run_convergence_analysis(
    all_mc_probs: np.ndarray,
    subset_size: int = 50
) -> Dict[str, Any]:
    """
    Conducts convergence analysis on a subset of the test split for N in {5, 10, 25, 50}.
    Computes mean probabilities and standard metrics at each N to evaluate stability.
    
    all_mc_probs: shape (num_samples, max_passes, num_classes) where max_passes should be >= 50.
    """
    logger.info("Running MC convergence analysis...")
    num_samples, max_passes, num_classes = all_mc_probs.shape
    
    # Select subset for convergence (first subset_size samples)
    n_subset = min(subset_size, num_samples)
    probs_subset = all_mc_probs[:n_subset] # (n_subset, max_passes, num_classes)
    
    n_values = [5, 10, 25]
    if max_passes >= 50:
        n_values.append(50)
        
    convergence_data = {}
    
    # Compute metrics for each N
    for n in n_values:
        # Slice the first n passes
        probs_n = probs_subset[:, :n, :] # (n_subset, n, num_classes)
        
        # Mean probability
        mean_prob = np.mean(probs_n, axis=1) # (n_subset, num_classes)
        
        # Shannon Entropy of mean probability
        eps = 1e-9
        clipped_mean_prob = np.clip(mean_prob, eps, 1.0 - eps)
        predictive_entropy = -np.sum(clipped_mean_prob * np.log(clipped_mean_prob), axis=-1)
        
        # Expected entropy
        clipped_probs_n = np.clip(probs_n, eps, 1.0 - eps)
        entropies_n = -np.sum(clipped_probs_n * np.log(clipped_probs_n), axis=-1) # (n_subset, n)
        expected_entropy = np.mean(entropies_n, axis=1) # (n_subset,)
        
        # Mutual Information
        mutual_info = predictive_entropy - expected_entropy
        
        # Predictive Variance (Mean Probability Variance)
        # Variance across passes (axis 1), then mean across classes (axis 1)
        variance_n = np.var(probs_n, axis=1) # (n_subset, num_classes)
        mean_variance = np.mean(variance_n, axis=1) # (n_subset,)
        
        convergence_data[str(n)] = {
            "mean_prob": mean_prob.tolist(),
            "predictive_entropy": predictive_entropy.tolist(),
            "expected_entropy": expected_entropy.tolist(),
            "mutual_info": mutual_info.tolist(),
            "mean_variance": mean_variance.tolist()
        }
        
    # Analyze difference between N=25 and the maximum N
    n_max = n_values[-1]
    if len(n_values) > 1:
        n_prev = n_values[-2] # usually 25 if n_max is 50
        
        entropy_diff = np.abs(np.array(convergence_data[str(n_prev)]["predictive_entropy"]) - np.array(convergence_data[str(n_max)]["predictive_entropy"]))
        var_diff = np.abs(np.array(convergence_data[str(n_prev)]["mean_variance"]) - np.array(convergence_data[str(n_max)]["mean_variance"]))
        mi_diff = np.abs(np.array(convergence_data[str(n_prev)]["mutual_info"]) - np.array(convergence_data[str(n_max)]["mutual_info"]))
        
        logger.info(f"Stability between N={n_prev} and N={n_max}:")
        logger.info(f"  Mean Absolute Diff in Predictive Entropy: {np.mean(entropy_diff):.6f}")
        logger.info(f"  Mean Absolute Diff in Predictive Variance: {np.mean(var_diff):.6f}")
        logger.info(f"  Mean Absolute Diff in Mutual Info: {np.mean(mi_diff):.6f}")
        
    return convergence_data
