import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import List, Union
from sklearn.metrics import roc_curve, auc

def plot_training_history(
    history: dict,
    save_dir: Union[str, Path]
) -> None:
    """
    Plots and saves loss and accuracy curves (train vs validation) from history dictionary.
    
    Args:
        history: Dictionary containing epoch-wise metrics (train_loss, val_loss, train_acc, val_acc, lr)
        save_dir: Path to directory where curves will be saved.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # 1. Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", color="orange", lw=2, linestyle="--")
    plt.title("Epoch vs Loss (Baseline)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png", dpi=300)
    plt.savefig(save_dir / "loss_curve.svg")
    plt.close()
    
    # 2. Plot Accuracy (and other metrics if present like QWK)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy", color="royalblue", lw=2)
    plt.plot(epochs, history["val_acc"], label="Val Accuracy", color="orange", lw=2, linestyle="--")
    if "val_qwk" in history:
        plt.plot(epochs, history["val_qwk"], label="Val QWK", color="forestgreen", lw=2, linestyle=":")
    plt.title("Epoch vs Accuracy / QWK (Baseline)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "metrics_curve.png", dpi=300)
    plt.savefig(save_dir / "metrics_curve.svg")
    plt.close()

    # 3. Plot Learning Rate
    if "lr" in history:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["lr"], label="Learning Rate", color="purple", lw=2)
        plt.title("Epoch vs Learning Rate Profile", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "lr_curve.png", dpi=300)
        plt.savefig(save_dir / "lr_curve.svg")
        plt.close()

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Union[str, Path]
) -> None:
    """
    Plots and saves the confusion matrix with absolute values and row-normalized percentages.
    
    Args:
        cm: Confusion matrix array.
        class_names: List of class label names.
        save_path: Path to save the figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate row-normalized matrix (percentages)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # handle division by zero
    
    # Generate labels containing both count and percentage
    labels = []
    for i in range(cm.shape[0]):
        row_labels = []
        for j in range(cm.shape[1]):
            row_labels.append(f"{cm[i, j]}\n({cm_norm[i, j]:.1%})")
        labels.append(row_labels)
    labels = np.array(labels)
    
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm_norm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5
    )
    plt.title("Confusion Matrix (Baseline EfficientNet-B0)", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=10, labelpad=10)
    plt.ylabel("True Label", fontsize=10, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path.with_suffix(".svg"))
    plt.close()

def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    save_path: Union[str, Path]
) -> None:
    """
    Plots and saves one-vs-rest ROC curves for multi-class classification.
    
    Args:
        y_true: True integer class labels (shape: [num_samples])
        y_probs: Predicted probabilities (shape: [num_samples, num_classes])
        class_names: List of class label names.
        save_path: Path to save the figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_classes = len(class_names)
    
    # Check if inputs are empty or invalid
    if len(y_true) == 0 or len(y_probs) == 0 or y_probs.shape[1] != num_classes:
        return
        
    plt.figure(figsize=(8, 6.5))
    
    # Plot ROC curve for each class
    for i in range(num_classes):
        # Convert multiclass true labels to binary (One-vs-Rest)
        y_true_binary = (y_true == i).astype(int)
        
        # Calculate ROC
        fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.4f})")
        
    # Plot random guess line
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate", fontsize=10, labelpad=10)
    plt.ylabel("True Positive Rate", fontsize=10, labelpad=10)
    plt.title("One-vs-Rest ROC Curves (Baseline EfficientNet-B0)", fontsize=12, fontweight="bold", pad=15)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path.with_suffix(".svg"))
    plt.close()
