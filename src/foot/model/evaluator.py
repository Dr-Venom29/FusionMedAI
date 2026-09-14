import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)
from typing import Dict, Any, List

def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray = None,
    class_names: List[str] = None
) -> Dict[str, Any]:
    """
    Computes baseline evaluation metrics for Wagner 4-class classification.
    
    Primary Metric: Macro F1-Score.
    Secondary Metrics: Accuracy, Weighted F1, Class-wise Precision/Recall/F1, Confusion Matrix, ROC-AUC.
    """
    if class_names is None:
        class_names = ["Grade 1", "Grade 2", "Grade 3", "Grade 4"]
        
    num_classes = len(class_names)
    
    # 1. Overall Classification Metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    
    # 2. Class-Wise Precision, Recall, F1
    precision_per_class = precision_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    
    class_metrics = {}
    for c_idx, c_name in enumerate(class_names):
        class_metrics[c_name] = {
            "class_index": c_idx,
            "precision": round(float(precision_per_class[c_idx]), 4),
            "recall": round(float(recall_per_class[c_idx]), 4),
            "f1_score": round(float(f1_per_class[c_idx]), 4)
        }
        
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes)).tolist()
    
    # 4. Multi-class One-vs-Rest ROC-AUC (if probabilities provided)
    macro_auc = None
    if y_probs is not None and y_probs.shape[1] == num_classes:
        try:
            macro_auc = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
        except Exception:
            macro_auc = None
            
    metrics = {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_roc_auc": round(macro_auc, 4) if macro_auc is not None else None,
        "class_metrics": class_metrics,
        "confusion_matrix": cm
    }
    
    return metrics

if __name__ == "__main__":
    # Sanity check
    y_t = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_p = np.array([0, 1, 2, 3, 0, 2, 2, 3])
    probs = np.eye(4)[y_p]
    m = compute_evaluation_metrics(y_t, y_p, probs)
    print("Evaluator module sanity check results:")
    print(f" Macro F1: {m['macro_f1']}, Accuracy: {m['accuracy']}")
    print(" Evaluator sanity check passed!")
