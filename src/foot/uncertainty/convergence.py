import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple
from src.foot.uncertainty.mc_dropout import enable_foot_mc_dropout
from src.foot.uncertainty.metrics import compute_mc_uncertainty_metrics

def run_mc_convergence_study(
    model: torch.nn.Module,
    vector_scaler: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    pass_counts: List[int] = [5, 10, 15, 20, 25, 30],
    entropy_threshold: float = 1e-3,
    variance_threshold: float = 1e-4,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Evaluates MC Dropout convergence across increasing pass counts N on the validation set.
    
    Convergence Selection Criteria:
        Primary convergence is evaluated on Predictive Entropy (stabilization threshold <= entropy_threshold)
        and Predictive Variance (stabilization threshold <= variance_threshold). The selected pass count N*
        is the smallest evaluated N where subsequent increments satisfy both stabilization criteria.
    
    Args:
        model: PyTorch model wrapper
        vector_scaler: Frozen vector scaling calibrator
        val_loader: Validation DataLoader
        pass_counts: List of pass counts to evaluate
        entropy_threshold: Maximum acceptable change in mean predictive entropy (default: 0.001 nats)
        variance_threshold: Maximum acceptable change in mean predictive variance (default: 0.0001)
        device: PyTorch device
        
    Returns:
        Dict containing evaluated records, convergence metrics, and empirically selected_pass_count.
    """
    enable_foot_mc_dropout(model)
    max_passes = max(pass_counts)
    total_batches = len(val_loader)
    
    print(f"       Maximum passes: {max_passes}\n", flush=True)
    all_stochastic_probs = []
    
    # Extract max_passes stochastic probability passes for all validation samples
    conv_start = time.perf_counter()
    with torch.no_grad():
        for pass_idx in range(max_passes):
            pass_start = time.perf_counter()
            pass_probs = []
            
            for batch_idx, (images, _) in enumerate(val_loader, start=1):
                images = images.to(device)
                logits = model(images)
                logits_tensor = logits["logits"] if isinstance(logits, dict) else logits
                
                scaled_logits = vector_scaler(logits_tensor)
                probs = F.softmax(scaled_logits, dim=1)
                pass_probs.append(probs.cpu())
                
                if batch_idx % 10 == 0 or batch_idx == total_batches:
                    elapsed = time.perf_counter() - pass_start
                    progress = batch_idx / total_batches * 100
                    print(
                        f"\r       Pass {pass_idx + 1:02d}/{max_passes} | "
                        f"Batch {batch_idx:03d}/{total_batches:03d} ({progress:5.1f}%) | "
                        f"Elapsed: {elapsed:6.1f}s",
                        end="",
                        flush=True
                    )
                    
            all_stochastic_probs.append(torch.cat(pass_probs, dim=0))
            pass_time = time.perf_counter() - pass_start
            print(f"\n       Pass {pass_idx + 1:02d}/{max_passes} completed in {pass_time:.1f}s", flush=True)
            
    total_conv_time = time.perf_counter() - conv_start
    print(f"\n       Validation stochastic passes completed in {total_conv_time / 60:.2f} minutes.\n", flush=True)
            
    # Stack into [max_passes, S, K]
    full_mc_probs = torch.stack(all_stochastic_probs, dim=0)
    
    results = []
    prev_metrics = None
    
    for n in pass_counts:
        sub_probs = full_mc_probs[:n] # [n, S, K]
        metrics = compute_mc_uncertainty_metrics(sub_probs)
        
        mean_entropy = float(torch.mean(metrics["predictive_entropy"]).item())
        mean_variance = float(torch.mean(metrics["predictive_variance"]).item())
        mean_mi = float(torch.mean(metrics["mutual_information"]).item())
        
        if prev_metrics is not None:
            entropy_delta = abs(mean_entropy - prev_metrics["mean_entropy"])
            var_delta = abs(mean_variance - prev_metrics["mean_variance"])
        else:
            entropy_delta = 0.0
            var_delta = 0.0
            
        record = {
            "pass_count": n,
            "mean_predictive_entropy": round(mean_entropy, 6),
            "mean_predictive_variance": round(mean_variance, 6),
            "mean_mutual_information": round(mean_mi, 6),
            "entropy_delta": round(entropy_delta, 6),
            "variance_delta": round(var_delta, 6)
        }
        results.append(record)
        prev_metrics = {"mean_entropy": mean_entropy, "mean_variance": mean_variance}
        print(f"       N={n:02d} -> Entropy: {mean_entropy:.4f} nats | Variance: {mean_variance:.6f} | Delta H: {entropy_delta:.6f}", flush=True)
        
    # Rule-based convergence selection: Smallest N where subsequent deltas remain below thresholds
    selected_n = max_passes
    for i in range(1, len(results)):
        subsequent_stable = all(
            r["entropy_delta"] <= entropy_threshold and r["variance_delta"] <= variance_threshold
            for r in results[i:]
        )
        if subsequent_stable:
            selected_n = results[i]["pass_count"]
            break
            
    print(f"\n       Selected N* = {selected_n}\n", flush=True)
    
    return {
        "pass_counts_evaluated": pass_counts,
        "entropy_threshold": entropy_threshold,
        "variance_threshold": variance_threshold,
        "selected_pass_count": selected_n,
        "convergence_records": results
    }
