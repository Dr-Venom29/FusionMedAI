"""
Standardized Area Under the ROC Curve (AUC) Computation (Phase C11.3).
Computes Macro One-vs-Rest (OvR) AUC for multiclass and standard ROC-AUC for binary tasks.
Strictly validation-only computation.
"""

from typing import Union, List, Optional
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def compute_binary_auc(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]]
) -> float:
    """
    Computes standard ROC-AUC for binary classification.
    
    Args:
        y_true: Ground truth binary labels {0, 1}.
        y_prob: Predicted probabilities for the positive class (1).
        
    Returns:
        float AUC in [0.0, 1.0].
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()
    
    if len(np.unique(y_t)) < 2:
        raise ValueError("Binary AUC requires at least two distinct classes in ground truth.")
        
    auc = float(roc_auc_score(y_t, y_p))
    return float(np.clip(auc, 0.0, 1.0))


def compute_macro_ovr_auc(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[List[float]]],
    num_classes: int
) -> float:
    """
    Computes Macro One-vs-Rest (OvR) ROC-AUC across K classes.
    
    Args:
        y_true: Ground truth class integer indices in {0, ..., K-1}.
        y_prob: Posterior class probability matrix of shape (N, K).
        num_classes: Expected number of classes K.
        
    Returns:
        float Macro-averaged OvR AUC in [0.0, 1.0].
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_prob, dtype=float)
    
    if y_p.ndim != 2 or y_p.shape[1] != num_classes:
        raise ValueError(
            f"Probability matrix must have shape (N, {num_classes}), got {y_p.shape}"
        )
        
    classes = list(range(num_classes))
    y_bin = label_binarize(y_t, classes=classes)
    
    # Compute macro-averaged OvR ROC-AUC
    auc = float(roc_auc_score(y_bin, y_p, multi_class="ovr", average="macro"))
    return float(np.clip(auc, 0.0, 1.0))
