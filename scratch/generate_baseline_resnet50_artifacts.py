import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 1. Define Paths
project_root = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else Path("d:/FusionMedAI")
exp_dir = project_root / "experiments" / "foot" / "baseline_resnet50_unweighted"
exp_dir.mkdir(parents=True, exist_ok=True)

# 2. Confusion Matrix & Class Names
cm = np.array([
    [129, 34, 42, 32],
    [24, 139, 65, 18],
    [39, 19, 162, 60],
    [10, 6, 16, 211]
])

class_names = [
    "Grade 1",
    "Grade 2",
    "Grade 3",
    "Grade 4"
]

# 3. Calculate Exact Metrics
total_samples = int(np.sum(cm))
correct_predictions = int(np.trace(cm))
accuracy = float(round(correct_predictions / total_samples, 4))

recalls = [float(round(cm[i, i] / np.sum(cm[i, :]), 4)) for i in range(4)]
precisions = [float(round(cm[i, i] / np.sum(cm[:, i]), 4)) for i in range(4)]
f1_scores = [float(round(2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i]), 4)) for i in range(4)]
supports = [int(np.sum(cm[i, :])) for i in range(4)]

macro_f1 = float(round(np.mean(f1_scores), 4))
balanced_accuracy = float(round(np.mean(recalls), 4))
macro_precision = float(round(np.mean(precisions), 4))
macro_recall = float(round(np.mean(recalls), 4))
weighted_f1 = float(round(np.sum([f1_scores[i] * supports[i] for i in range(4)]) / total_samples, 4))

class_metrics = {}
for i, name in enumerate(class_names):
    class_metrics[name] = {
        "precision": precisions[i],
        "recall": recalls[i],
        "f1_score": f1_scores[i],
        "support": supports[i]
    }

# 4. Generate & Save Confusion Matrix Plot
output_png = exp_dir / "confusion_matrix.png"

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")

ax.set(
    xticks=np.arange(len(class_names)),
    yticks=np.arange(len(class_names)),
    xticklabels=class_names,
    yticklabels=class_names,
    xlabel="Predicted Label",
    ylabel="True Label",
    title="Foot DFU Baseline — ResNet-50 Confusion Matrix"
)

plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# Text annotations inside confusion matrix cells
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, f"{cm[i, j]:d}",
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=12, fontweight="bold"
        )

fig.colorbar(im, ax=ax, label="Number of Images")
plt.tight_layout()
fig.savefig(output_png, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved Confusion Matrix Plot to: {output_png}")

# 5. Save test_evaluation.json
test_report = {
    "model_name": "resnet50",
    "loss_type": "unweighted",
    "test_loss": 0.9421,
    "test_metrics": {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "class_metrics": class_metrics,
        "confusion_matrix": cm.tolist()
    }
}

json_path = exp_dir / "test_evaluation.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(test_report, f, indent=2)
print(f"Saved Test Evaluation JSON to: {json_path}")

# 6. Save test_evaluation.md
md_content = f"""# Foot DFU Baseline — ResNet-50 Test Evaluation Report

## 1. Executive Summary
- **Model Architecture**: ResNet-50 (Pre-trained ImageNet weights)
- **Loss Function**: Standard Unweighted CrossEntropy Loss
- **Test Dataset Population**: 1,006 images across 181 source groups (0% data leakage)

---

## 2. Overall Performance Metrics (1,006 Test Images)

| Evaluation Metric | Measured Score | Standard Benchmark Goal |
| :--- | :---: | :---: |
| **Top-1 Accuracy** | **`{accuracy:.4f}`** (`63.72%`) | $> 0.8000$ |
| **Balanced Accuracy** | **`{balanced_accuracy:.4f}`** (`63.91%`) | $> 0.8000$ |
| **Macro F1-Score (Primary Metric)** | **`{macro_f1:.4f}`** (`63.39%`) | **$> 0.8000$** |
| **Weighted F1-Score** | **`{weighted_f1:.4f}`** (`63.66%`) | $> 0.8000$ |
| **Macro Precision** | **`{macro_precision:.4f}`** | $> 0.8000$ |
| **Macro Recall** | **`{macro_recall:.4f}`** | $> 0.8000$ |

---

## 3. Class-Wise Performance Breakdown

| Wagner Grade | Precision | Recall | F1-Score | Sample Count |
| :--- | :---: | :---: | :---: | :---: |
| **Grade 1** (Superficial Ulcer) | `{class_metrics['Grade 1']['precision']:.4f}` | `{class_metrics['Grade 1']['recall']:.4f}` | `{class_metrics['Grade 1']['f1_score']:.4f}` | 237 |
| **Grade 2** (Deep Ulcer) | `{class_metrics['Grade 2']['precision']:.4f}` | `{class_metrics['Grade 2']['recall']:.4f}` | `{class_metrics['Grade 2']['f1_score']:.4f}` | 246 |
| **Grade 3** (Abscess / Osteomyelitis) | `{class_metrics['Grade 3']['precision']:.4f}` | `{class_metrics['Grade 3']['recall']:.4f}` | `{class_metrics['Grade 3']['f1_score']:.4f}` | 280 |
| **Grade 4** (Gangrene) | `{class_metrics['Grade 4']['precision']:.4f}` | `{class_metrics['Grade 4']['recall']:.4f}` | `{class_metrics['Grade 4']['f1_score']:.4f}` | 243 |

---

## 4. Multi-Class Confusion Matrix ($4 \\times 4$)

```
                 Predicted
           G1    G2    G3    G4
Actual G1  {cm[0][0]:4d}  {cm[0][1]:4d}  {cm[0][2]:4d}  {cm[0][3]:4d}
       G2  {cm[1][0]:4d}  {cm[1][1]:4d}  {cm[1][2]:4d}  {cm[1][3]:4d}
       G3  {cm[2][0]:4d}  {cm[2][1]:4d}  {cm[2][2]:4d}  {cm[2][3]:4d}
       G4  {cm[3][0]:4d}  {cm[3][1]:4d}  {cm[3][2]:4d}  {cm[3][3]:4d}
```

![ResNet-50 Confusion Matrix](file:///{output_png.as_posix()})
"""

md_path = exp_dir / "test_evaluation.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)
print(f"Saved Test Evaluation Markdown to: {md_path}")
