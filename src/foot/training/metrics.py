import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

def compute_baseline_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics for Wagner 4-Class Classification (Phase 10.4.11).
    
    Returns:
        Dict containing accuracy, balanced_accuracy, macro_f1, weighted_f1, macro_precision, macro_recall,
        per-class metrics, confusion matrix, and optional ROC-AUC.
    """
    if class_names is None:
        class_names = ["Grade 1", "Grade 2", "Grade 3", "Grade 4"]
        
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Class-wise Metrics
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    class_metrics = {}
    for i, c_name in enumerate(class_names):
        support = int(np.sum(y_true == i))
        class_metrics[c_name] = {
            "precision": float(round(prec_per_class[i], 4)),
            "recall": float(round(rec_per_class[i], 4)),
            "f1_score": float(round(f1_per_class[i], 4)),
            "support": support
        }
        
    metrics = {
        "accuracy": float(round(acc, 4)),
        "balanced_accuracy": float(round(bal_acc, 4)),
        "macro_precision": float(round(macro_prec, 4)),
        "macro_recall": float(round(macro_rec, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "weighted_f1": float(round(weighted_f1, 4)),
        "class_metrics": class_metrics,
        "confusion_matrix": cm.tolist()
    }
    
    # ROC-AUC computation
    if y_probs is not None and y_probs.ndim == 2 and y_probs.shape[1] == len(class_names):
        try:
            macro_roc_auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
            metrics["macro_roc_auc"] = float(round(macro_roc_auc, 4))
        except Exception:
            metrics["macro_roc_auc"] = None
            
    return metrics
