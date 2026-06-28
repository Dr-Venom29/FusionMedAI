import numpy as np
from typing import Dict, Union, Optional
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score
)
from src.config import NUM_CLASSES

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Computes classification evaluation metrics for Diabetic Retinopathy staging.
    
    Metrics computed:
      - Accuracy
      - Balanced Accuracy
      - Precision (Macro)
      - Recall (Macro)
      - F1 Score (Macro)
      - Cohen's Kappa (Linear)
      - Quadratic Weighted Kappa (QWK)
      - Sensitivity (Macro Recall)
      - Specificity (Macro)
      - ROC AUC (One-vs-Rest Macro, if y_probs is provided)
      - Confusion Matrix
      
    Args:
        y_true: True integer class labels, shape [num_samples]
        y_pred: Predicted class labels, shape [num_samples]
        y_probs: Predicted class probabilities, shape [num_samples, num_classes] (optional)
        
    Returns:
        Dict[str, Union[float, np.ndarray]]: Computed metrics.
    """
    # 1. Basic metrics using scikit-learn
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Precision, Recall (Sensitivity), F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # Cohen's Kappa & Quadratic Weighted Kappa (QWK)
    linear_kappa = cohen_kappa_score(y_true, y_pred, weights=None)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    
    # 2. Specificity (Macro) calculation
    # For each class i, specificity = TN_i / (TN_i + FP_i)
    num_classes = NUM_CLASSES
    specificities = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = sum(cm[i, j] for j in range(num_classes) if j != i)
        fp = sum(cm[j, i] for j in range(num_classes) if j != i)
        
        # TN is the sum of everything except row i and col i
        tn = sum(
            cm[j, k] 
            for j in range(num_classes) 
            for k in range(num_classes) 
            if j != i and k != i
        )
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
        
    macro_specificity = float(np.mean(specificities))
    
    # 3. Assemble results dict
    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall), # Identical to Sensitivity macro
        "f1": float(f1),
        "linear_kappa": float(linear_kappa),
        "qwk": float(qwk),
        "sensitivity": float(recall),
        "specificity": macro_specificity,
        "confusion_matrix": cm
    }
    
    # 4. Optional ROC AUC (One-vs-Rest)
    if y_probs is not None:
        try:
            # Handle edge cases where not all classes are represented in y_true during mini batch dry-runs
            auc_score = roc_auc_score(
                y_true, 
                y_probs, 
                multi_class="ovr", 
                average="macro",
                labels=np.arange(num_classes)
            )
            metrics["auc"] = float(auc_score)
        except Exception:
            # Fallback if AUC cannot be calculated (e.g. single class representation in split)
            metrics["auc"] = 0.5
            
    return metrics
