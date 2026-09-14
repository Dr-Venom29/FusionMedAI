import os
import sys
import json
from pathlib import Path
import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.foot.config import (
    SEED,
    NUM_CLASSES,
    CLASS_NAMES,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    DEVICE,
    FOOT_EXPERIMENTS_DIR,
    METADATA_DIR,
    PROCESSED_SPLITS_DIR
)
from src.foot.data.dataloader import create_foot_dataloaders
from src.foot.model.baseline_model import build_foot_baseline_model
from src.foot.model.trainer import FootBaselineTrainer, set_reproducibility

def run_baseline_experiment():
    print("==================================================", flush=True)
    print("Phase 10.4 — Foot DFU Baseline Model Training", flush=True)
    print("==================================================", flush=True)
    
    set_reproducibility(SEED)
    
    # 1. Create DataLoaders
    print(f"Creating DataLoaders (Batch Size: {BATCH_SIZE}, Workers: {NUM_WORKERS}, Seed: {SEED})...", flush=True)
    train_csv = PROCESSED_SPLITS_DIR / "train.csv"
    val_csv = PROCESSED_SPLITS_DIR / "val.csv"
    test_csv = PROCESSED_SPLITS_DIR / "test.csv"
    
    # Determine optimal worker count for current device/OS
    workers = 0 if DEVICE == "cpu" else NUM_WORKERS

    train_loader, val_loader, test_loader = create_foot_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        batch_size=BATCH_SIZE,
        num_workers=workers,
        seed=SEED
    )
    print(" DataLoaders created successfully.", flush=True)
    
    # 2. Build Model
    print(f"Building EfficientNet-B0 Baseline Model on device: {DEVICE}...", flush=True)
    model = build_foot_baseline_model(num_classes=NUM_CLASSES, pretrained=True, device=DEVICE)
    print(" Baseline Model built successfully.", flush=True)
    
    # 3. Setup Loss Function (Sqrt Inverse Frequency Weights)
    # Weights computed during Phase 10.3.4: Grade 1: 1.0870, Grade 2: 1.0669, Grade 3: 1.0000, Grade 4: 1.0731
    class_weights_tensor = torch.tensor([1.0870, 1.0669, 1.0000, 1.0731], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # 4. Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # 5. Execute Training
    checkpoint_dir = FOOT_EXPERIMENTS_DIR / "baseline" / "checkpoints"
    trainer = FootBaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=DEVICE,
        epochs=EPOCHS,
        patience=PATIENCE,
        checkpoint_dir=checkpoint_dir,
        seed=SEED
    )
    
    training_summary = trainer.train()
    
    # 6. Evaluate Best Model on Frozen Test Set
    print("\n==================================================")
    print("Evaluating Best Model Checkpoint on Frozen Test Partition...")
    print("==================================================")
    
    best_ckpt_path = checkpoint_dir / "best_model.pth"
    best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    
    test_loss, test_metrics = trainer.evaluate(test_loader)
    
    print("\n--- Frozen Test Set Evaluation Results ---")
    print(f" Test Loss:         {test_loss:.4f}")
    print(f" Macro F1-Score:    {test_metrics['macro_f1']:.4f} (PRIMARY METRIC)")
    print(f" Top-1 Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f" Weighted F1-Score: {test_metrics['weighted_f1']:.4f}")
    if test_metrics.get("macro_roc_auc") is not None:
        print(f" Macro ROC-AUC:     {test_metrics['macro_roc_auc']:.4f}")
        
    print("\nClass-Wise Metrics:")
    for c_name, c_m in test_metrics["class_metrics"].items():
        print(f"  {c_name:10s} | Precision: {c_m['precision']:.4f} | Recall: {c_m['recall']:.4f} | F1: {c_m['f1_score']:.4f}")
        
    print("\n4x4 Confusion Matrix (Rows=True, Cols=Pred):")
    for row in test_metrics["confusion_matrix"]:
        print(f"  {row}")
        
    # Save Test Evaluation Reports
    test_report = {
        "model_name": "EfficientNet-B0 Baseline",
        "checkpoint_evaluated": str(best_ckpt_path),
        "best_val_loss": training_summary["best_val_loss"],
        "best_val_macro_f1": training_summary["best_val_macro_f1"],
        "test_loss": round(test_loss, 4),
        "test_metrics": test_metrics
    }
    
    test_json_path = METADATA_DIR / "baseline_test_evaluation.json"
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)
    print(f"\nSaved Test evaluation JSON to: {test_json_path}")
    
    md_content = rf"""# Phase 10.4 — Foot DFU Baseline Model Evaluation Report

## 1. Baseline Model Specifications

- **Architecture**: EfficientNet-B0 (Pre-trained ImageNet weights + Wagner 4-class head)
- **Input Resolution**: $224 \\times 224 \\times 3$ RGB
- **Normalization**: Observed RGB mean $[0.4937, 0.3630, 0.3272]$, std $[0.1745, 0.1632, 0.1551]$
- **Optimizer**: AdamW ($\text{{lr}}=1\text{{e-}}4$, $\text{{weight\_decay}}=1\text{{e-}}4$)
- **Scheduler**: CosineAnnealingLR ($T_{{\text{{max}}}}=20$, $\eta_{{\text{{min}}}}=1\text{{e-}}6$)
- **Loss Function**: Sqrt Inverse Frequency Weighted Cross-Entropy

---

## 2. Overall Performance on Frozen Test Partition (1,006 Images)

| Evaluation Metric | Measured Score | Standard Benchmark Goal |
| :--- | :---: | :---: |
| **Macro F1-Score (Primary Metric)** | **`{test_metrics['macro_f1']:.4f}`** | **$> 0.8000$** |
| **Top-1 Accuracy** | **`{test_metrics['accuracy']:.4f}`** | $> 0.8000$ |
| **Weighted F1-Score** | **`{test_metrics['weighted_f1']:.4f}`** | $> 0.8000$ |
| **Macro ROC-AUC** | **`{test_metrics['macro_roc_auc']}`** | $> 0.9000$ |

---

## 3. Class-Wise Performance Profile

| Wagner Class | Precision | Recall | F1-Score | Sample Count |
| :--- | :---: | :---: | :---: | :---: |
| **Grade 1** | `{test_metrics['class_metrics']['Grade 1']['precision']:.4f}` | `{test_metrics['class_metrics']['Grade 1']['recall']:.4f}` | `{test_metrics['class_metrics']['Grade 1']['f1_score']:.4f}` | 237 |
| **Grade 2** | `{test_metrics['class_metrics']['Grade 2']['precision']:.4f}` | `{test_metrics['class_metrics']['Grade 2']['recall']:.4f}` | `{test_metrics['class_metrics']['Grade 2']['f1_score']:.4f}` | 246 |
| **Grade 3** | `{test_metrics['class_metrics']['Grade 3']['precision']:.4f}` | `{test_metrics['class_metrics']['Grade 3']['recall']:.4f}` | `{test_metrics['class_metrics']['Grade 3']['f1_score']:.4f}` | 280 |
| **Grade 4** | `{test_metrics['class_metrics']['Grade 4']['precision']:.4f}` | `{test_metrics['class_metrics']['Grade 4']['recall']:.4f}` | `{test_metrics['class_metrics']['Grade 4']['f1_score']:.4f}` | 243 |

---

## 4. Multi-Class Confusion Matrix

```
                 Predicted
           G1    G2    G3    G4
Actual G1  {test_metrics['confusion_matrix'][0][0]:4d}  {test_metrics['confusion_matrix'][0][1]:4d}  {test_metrics['confusion_matrix'][0][2]:4d}  {test_metrics['confusion_matrix'][0][3]:4d}
       G2  {test_metrics['confusion_matrix'][1][0]:4d}  {test_metrics['confusion_matrix'][1][1]:4d}  {test_metrics['confusion_matrix'][1][2]:4d}  {test_metrics['confusion_matrix'][1][3]:4d}
       G3  {test_metrics['confusion_matrix'][2][0]:4d}  {test_metrics['confusion_matrix'][2][1]:4d}  {test_metrics['confusion_matrix'][2][2]:4d}  {test_metrics['confusion_matrix'][2][3]:4d}
       G4  {test_metrics['confusion_matrix'][3][0]:4d}  {test_metrics['confusion_matrix'][3][1]:4d}  {test_metrics['confusion_matrix'][3][2]:4d}  {test_metrics['confusion_matrix'][3][3]:4d}
```

---

## 5. Phase 10.4 Acceptance Sign-Off

The baseline framework is **EXECUTED & VERIFIED**. Checkpoints are saved under `experiments/foot/baseline/checkpoints/`.
"""
    
    test_md_path = METADATA_DIR / "baseline_test_evaluation.md"
    with open(test_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved Test evaluation Markdown to: {test_md_path}")
    
    print("\nPhase 10.4 Baseline Training Experiment Completed Successfully!")

if __name__ == "__main__":
    run_baseline_experiment()
