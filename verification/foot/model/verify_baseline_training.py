import os
import sys
import json
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.config import (
    SEED,
    NUM_CLASSES,
    CLASS_NAMES,
    OBSERVED_DATASET_MEAN,
    OBSERVED_DATASET_STD
)
from src.foot.model.baseline_model import build_foot_baseline_model
from src.foot.model.evaluator import compute_evaluation_metrics
from src.foot.model.trainer import FootBaselineTrainer, set_reproducibility
from src.foot.data.dataloader import create_foot_dataloaders

def verify_baseline_training_pipeline():
    print("==================================================", flush=True)
    print("Verifying Phase 10.4 — Baseline Model Training & Pipeline", flush=True)
    print("==================================================", flush=True)
    
    set_reproducibility(SEED)
    
    # 1. Test Model Architecture & Gradient Backprop
    print("1. Testing EfficientNet-B0 baseline model forward pass & backpropagation...", flush=True)
    dummy_input = torch.randn(8, 3, 224, 224)
    dummy_target = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    
    model = build_foot_baseline_model(num_classes=4, pretrained=False, device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_target)
    loss.backward()
    
    # Verify non-null gradient on classifier head
    classifier_grad = model.backbone.classifier[1].weight.grad
    assert classifier_grad is not None and classifier_grad.abs().sum() > 0, "Null gradient detected on model head!"
    print(f" [PASS] Forward pass loss: {loss.item():.4f}, Classifier gradient norm: {classifier_grad.norm().item():.4f}", flush=True)

    # 2. Test Evaluator Metrics Computation
    print("2. Testing evaluation metrics computation...", flush=True)
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 2, 2, 3])
    probs = np.eye(4)[y_pred]
    
    metrics = compute_evaluation_metrics(y_true, y_pred, probs, CLASS_NAMES)
    assert "macro_f1" in metrics and "accuracy" in metrics, "Missing core metrics in evaluator output!"
    assert "Grade 3" in metrics["class_metrics"], "Missing Grade 3 class metrics!"
    print(f" [PASS] Evaluator verified (Macro F1: {metrics['macro_f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}).", flush=True)

    # 3. Test Trainer Integration on Synthetic Loader (1 epoch)
    print("3. Testing FootBaselineTrainer 1-epoch execution & checkpointing...", flush=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_ckpt = Path(tmp_dir) / "checkpoints"
        
        # Create synthetic datasets
        ds_synthetic = torch.utils.data.TensorDataset(dummy_input, dummy_target)
        train_loader = torch.utils.data.DataLoader(ds_synthetic, batch_size=4, shuffle=True)
        val_loader = torch.utils.data.DataLoader(ds_synthetic, batch_size=4, shuffle=False)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        
        trainer = FootBaselineTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device="cpu",
            epochs=1,
            patience=5,
            checkpoint_dir=tmp_ckpt,
            seed=SEED
        )
        
        summary = trainer.train()
        
        best_ckpt = tmp_ckpt / "best_model.pth"
        latest_ckpt = tmp_ckpt / "latest_model.pth"
        hist_csv = tmp_ckpt / "training_history.csv"
        
        assert best_ckpt.exists(), "best_model.pth checkpoint was not created!"
        assert latest_ckpt.exists(), "latest_model.pth checkpoint was not created!"
        assert hist_csv.exists(), "training_history.csv was not created!"
        
        # Test loading checkpoint
        ckpt = torch.load(best_ckpt, weights_only=False)
        assert "model_state_dict" in ckpt and "metrics" in ckpt, "Corrupted checkpoint state dict!"
        print(" [PASS] 1-Epoch Trainer execution, checkpoint saving/loading verified.")

    print("\n==================================================")
    print("FOOT BASELINE MODEL TRAINING & PIPELINE VERIFICATION: PASSED")
    print("==================================================")

if __name__ == "__main__":
    verify_baseline_training_pipeline()
