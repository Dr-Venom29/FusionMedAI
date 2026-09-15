import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from typing import Dict, Tuple, Any

def compute_classification_scores(probs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    preds_np = probs.argmax(dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()
    
    acc = float(np.mean(preds_np == labels_np))
    macro_f1 = float(f1_score(labels_np, preds_np, average="macro"))
    bal_acc = float(balanced_accuracy_score(labels_np, preds_np))
    
    try:
        roc_auc = float(roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro"))
    except ValueError:
        roc_auc = 0.0
        
    return {
        "Accuracy": round(acc, 4),
        "Macro F1": round(macro_f1, 4),
        "Balanced Accuracy": round(bal_acc, 4),
        "Macro ROC-AUC": round(roc_auc, 4)
    }

def compute_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, labels).item())

def compute_brier_score(probs: torch.Tensor, labels: torch.Tensor, num_classes: int = 4) -> float:
    labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
    return float(F.mse_loss(probs, labels_one_hot).item())

def compute_ece_mce(probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 10) -> Tuple[float, float, Dict[str, Any]]:
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = (predictions == labels).float()
    
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    ece = torch.tensor(0.0)
    mce = torch.tensor(0.0)
    
    total_samples = len(labels)
    bin_details = []
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_size = in_bin.sum().item()
        
        if bin_size > 0:
            acc_in_bin = float(accuracies[in_bin].mean().item())
            conf_in_bin = float(confidences[in_bin].mean().item())
            gap = abs(acc_in_bin - conf_in_bin)
            
            ece += (bin_size / total_samples) * gap
            mce = max(mce, torch.tensor(gap))
            
            bin_details.append({
                "bin_index": i,
                "range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "count": bin_size,
                "accuracy": round(acc_in_bin, 4),
                "confidence": round(conf_in_bin, 4),
                "gap": round(gap, 4)
            })
        else:
            bin_details.append({
                "bin_index": i,
                "range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "count": 0,
                "accuracy": 0.0,
                "confidence": round(float((bin_lower + bin_upper) / 2.0), 4),
                "gap": 0.0
            })
            
    return float(ece.item()), float(mce.item()), {"bin_protocol": f"{num_bins} equal-width bins", "bins": bin_details}

def compute_per_class_calibration(probs: torch.Tensor, labels: torch.Tensor, num_classes: int = 4) -> Dict[str, Dict[str, float]]:
    """Computes ECE, NLL, and accuracy per Wagner grade."""
    per_class = {}
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            c_probs = probs[mask]
            c_labels = labels[mask]
            c_conf, c_pred = torch.max(c_probs, dim=1)
            c_acc = float((c_pred == c_labels).float().mean().item())
            c_mean_conf = float(c_conf.mean().item())
            c_ece, _ = compute_ece_mce(c_probs, c_labels, num_bins=10)[:2]
            
            per_class[f"Grade_{c+1}"] = {
                "sample_count": int(mask.sum().item()),
                "accuracy": round(c_acc, 4),
                "mean_confidence": round(c_mean_conf, 4),
                "overconfidence_gap": round(c_mean_conf - c_acc, 4),
                "ece": round(c_ece, 4)
            }
    return per_class

def compute_calibration_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10,
    num_classes: int = 4
) -> Dict[str, Any]:
    """
    Computes complete calibration diagnostic suite across fixed 10-bin protocol.
    """
    probs = F.softmax(logits, dim=1)
    
    clf_scores = compute_classification_scores(probs, labels)
    nll = compute_nll(logits, labels)
    brier = compute_brier_score(probs, labels, num_classes=num_classes)
    ece, mce, bin_info = compute_ece_mce(probs, labels, num_bins=num_bins)
    
    confidences, _ = torch.max(probs, dim=1)
    mean_conf = float(confidences.mean().item())
    overconfidence_gap = float(mean_conf - clf_scores["Accuracy"])
    per_class_stats = compute_per_class_calibration(probs, labels, num_classes=num_classes)
    
    return {
        "Accuracy": clf_scores["Accuracy"],
        "Macro F1": clf_scores["Macro F1"],
        "Balanced Accuracy": clf_scores["Balanced Accuracy"],
        "Macro ROC-AUC": clf_scores["Macro ROC-AUC"],
        "NLL": round(nll, 4),
        "Brier": round(brier, 4),
        "ECE": round(ece, 4),
        "MCE": round(mce, 4),
        "Mean Confidence": round(mean_conf, 4),
        "Overconfidence Gap": round(overconfidence_gap, 4),
        "bin_details": bin_info,
        "per_class_calibration": per_class_stats
    }
