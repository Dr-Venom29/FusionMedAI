import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

EPS = 1e-10

def compute_entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Computes Shannon entropy across specified probability dimension."""
    probs_clamped = torch.clamp(probs, min=EPS, max=1.0)
    return -torch.sum(probs_clamped * torch.log(probs_clamped), dim=dim)

def compute_mc_uncertainty_metrics(
    mc_probs: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Computes MC Dropout uncertainty statistics from a tensor of stochastic pass probabilities.
    
    Statistical Note:
        Sample variance per class uses `torch.var(..., unbiased=True)` (Bessel's correction dividing
        by N - 1 when N > 1) to provide an unbiased estimate of the variance across stochastic passes.
    
    Args:
        mc_probs: Tensor of shape [N_passes, S_samples, K_classes]
        
    Returns:
        Dict containing:
            - predictive_mean: [S, K]
            - predictive_variance: [S] (mean unbiased sample variance across K classes)
            - predictive_entropy: [S] (Total Uncertainty H(p_bar))
            - expected_entropy: [S] (Aleatoric Uncertainty E[H(p_t)])
            - mutual_information: [S] (Epistemic Uncertainty MI)
    """
    # 1. Predictive Mean [S, K]
    predictive_mean = torch.mean(mc_probs, dim=0)
    
    # 2. Predictive Variance per class [S, K] (unbiased sample variance when N > 1) and scalar per sample [S]
    var_per_class = torch.var(mc_probs, dim=0, unbiased=True if mc_probs.shape[0] > 1 else False)
    predictive_variance = torch.mean(var_per_class, dim=-1)
    
    # 3. Total Uncertainty: Entropy of Predictive Mean [S]
    predictive_entropy = compute_entropy(predictive_mean, dim=-1)
    
    # 4. Aleatoric Uncertainty: Expected Entropy across passes [S]
    pass_entropies = compute_entropy(mc_probs, dim=-1) # [N, S]
    expected_entropy = torch.mean(pass_entropies, dim=0) # [S]
    
    # 5. Epistemic Uncertainty: Mutual Information [S]
    mutual_information = torch.clamp(predictive_entropy - expected_entropy, min=0.0)
    
    return {
        "predictive_mean": predictive_mean,
        "predictive_variance": predictive_variance,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information
    }

def compute_error_detection_metrics(
    uncertainties: np.ndarray,
    errors: np.ndarray
) -> Dict[str, Any]:
    """
    Computes AUROC and AUPRC (Average Precision) for misclassification error detection.
    
    Args:
        uncertainties: Array of uncertainty scores per sample [S]
        errors: Binary ground truth error array [S] (1 = misclassified, 0 = correct)
        
    Returns:
        Dict with keys 'auroc', 'auprc', and 'status'
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    # Verify both classes (0 and 1) exist in target array
    if len(np.unique(errors)) < 2:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "status": "undefined"
        }
        
    try:
        auroc = float(roc_auc_score(errors, uncertainties))
        auprc = float(average_precision_score(errors, uncertainties))
        return {
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "status": "evaluated"
        }
    except ValueError as e:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "status": f"error: {str(e)}"
        }
