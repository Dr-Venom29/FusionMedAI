import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import src.config as config

logger = logging.getLogger("Uncertainty_Metrics")

def compute_entropy_np(probs: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy using base e."""
    eps = 1e-9
    clipped_probs = np.clip(probs, eps, 1.0 - eps)
    return -np.sum(clipped_probs * np.log(clipped_probs), axis=-1)

def compute_stochastic_metrics(all_mc_probs: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes MC stochastic uncertainty metrics from the probability tensor of shape (num_samples, N, num_classes).
    
    Returns a dictionary of numpy arrays of shape (num_samples,):
      - mean_probs: mean probability across passes (shape num_samples x num_classes)
      - predictive_entropy: Shannon entropy of mean probabilities H(\bar{p})
      - predictive_entropy_norm: Normalized predictive entropy H(\bar{p}) / log(5)
      - expected_entropy: Mean of entropies of stochastic passes E[H(p)]
      - expected_entropy_norm: E[H(p)] / log(5)
      - predictive_variance: Mean probability variance across passes (mean Var(p_c))
      - mutual_information: H(\bar{p}) - E[H(p)]
    """
    logger.info("Computing stochastic uncertainty metrics...")
    num_samples, n_passes, num_classes = all_mc_probs.shape
    
    # 1. Compute Mean Probabilities
    mean_probs = np.mean(all_mc_probs, axis=1) # (num_samples, num_classes)
    
    # 2. Predictive Entropy H(\bar{p})
    predictive_entropy = compute_entropy_np(mean_probs)
    predictive_entropy_norm = predictive_entropy / np.log(num_classes)
    
    # 3. Expected Entropy E[H(p_t)]
    # Compute entropy for each pass
    pass_entropies = np.zeros((num_samples, n_passes))
    for t in range(n_passes):
        pass_entropies[:, t] = compute_entropy_np(all_mc_probs[:, t, :])
        
    expected_entropy = np.mean(pass_entropies, axis=1)
    expected_entropy_norm = expected_entropy / np.log(num_classes)
    
    # 4. Predictive Variance (Mean Probability Variance across passes)
    # Variance of probabilities for each class across passes, then average over classes
    class_variances = np.var(all_mc_probs, axis=1) # (num_samples, num_classes)
    predictive_variance = np.mean(class_variances, axis=1) # (num_samples,)
    
    # 5. Mutual Information: MI = H(\bar{p}) - E[H(p_t)]
    mutual_information = predictive_entropy - expected_entropy
    # Clip tiny negative values due to numerical precision
    mutual_information = np.clip(mutual_information, 0.0, None)
    
    return {
        "mean_probs": mean_probs,
        "predictive_entropy": predictive_entropy,
        "predictive_entropy_norm": predictive_entropy_norm,
        "expected_entropy": expected_entropy,
        "expected_entropy_norm": expected_entropy_norm,
        "predictive_variance": predictive_variance,
        "mutual_information": mutual_information
    }

def evaluate_error_detection(
    is_incorrect: np.ndarray,
    uncertainty_score: np.ndarray,
    higher_is_more_uncertain: bool = True
) -> Tuple[float, float]:
    """
    Computes AUROC and AUPRC for error detection where:
      - Positive class is is_incorrect = 1
      - Score represents prediction uncertainty (higher = more likely incorrect)
      
    If higher_is_more_uncertain is False (e.g. for raw confidence or margin),
    the score is inverted inside the function (score = 1.0 - score or -score)
    to maintain consistent positive orientation.
    """
    # Clone and orient score
    score = np.copy(uncertainty_score)
    if not higher_is_more_uncertain:
        # Invert the score (works for bounded metrics like confidence and margin)
        # For general scale we can do negative
        if np.max(score) <= 1.0 and np.min(score) >= 0.0:
            score = 1.0 - score
        else:
            score = -score
            
    # Compute metrics
    # If all targets are 1 or 0, AUROC/AUPRC is undefined
    if len(np.unique(is_incorrect)) < 2:
        logger.warning("Only one unique class label present in error detection evaluation. AUROC/AUPRC undefined.")
        return 0.5, 0.0
        
    auroc = roc_auc_score(is_incorrect, score)
    
    precisions, recalls, _ = precision_recall_curve(is_incorrect, score)
    auprc = auc(recalls, precisions)
    
    return auroc, auprc
