import os
import sys
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.config import SEED, NUM_CLASSES, CLASS_NAMES
from src.foot.models import build_foot_baseline_model
from src.foot.training.config import BaselineConfig
from src.foot.training.metrics import compute_evaluation_metrics
from src.foot.training import FootBaselineTrainer
from src.foot.training.trainer import set_reproducibility

def verify_baseline_training_pipeline():
    print("==================================================", flush=True)
    print("Verifying Baseline Model Training & Pipeline", flush=True)
    print("==================================================", flush=True)
    
    set_reproducibility(SEED)
    
    # 1. Test ResNet-50 Model Architecture & Gradient Backpropagation
    print("1. Testing ResNet-50 baseline model forward pass & backpropagation...", flush=True)
    dummy_input = torch.randn(8, 3, 224, 224)
    dummy_target = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    
    model = build_foot_baseline_model(num_classes=NUM_CLASSES, pretrained=False, device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    out = model(dummy_input)
    logits = out["logits"] if isinstance(out, dict) else out
    loss = criterion(logits, dummy_target)
    loss.backward()
    
    # Verify device consistency and non-null gradient on classifier head
    assert next(model.parameters()).device.type == "cpu", "Model parameters must be on CPU for synthetic test!"
    head_param_grad = list(model.parameters())[-1].grad
    assert head_param_grad is not None and head_param_grad.abs().sum() > 0, "Null gradient detected on classifier head!"
    print(f" [PASS] Forward pass loss: {loss.item():.4f}, Head gradient norm: {head_param_grad.norm().item():.4f}", flush=True)

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
        tmp_path = Path(tmp_dir)
        config = BaselineConfig(
            model_name="resnet50",
            loss_type="unweighted",
            epochs=1,
            batch_size=4,
            experiments_dir=tmp_path
        )
        
        # Create synthetic dataset & loaders
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
            config=config
        )
        
        summary = trainer.train()
        
        best_ckpt = config.checkpoint_dir / "best_model.pt"
        last_ckpt = config.checkpoint_dir / "last_model.pt"
        hist_csv = config.experiment_dir / "training_history.csv"
        
        assert best_ckpt.exists(), "best_model.pt checkpoint was not created!"
        assert last_ckpt.exists(), "last_model.pt checkpoint was not created!"
        assert hist_csv.exists(), "training_history.csv was not created!"
        
        # Test loading checkpoint
        checkpoint = trainer.checkpoint_manager.load(model=model, checkpoint_path=best_ckpt, device="cpu")
        assert "model_state_dict" in checkpoint and "seed" in checkpoint, "Corrupted checkpoint state dict!"
        print(" [PASS] 1-Epoch Trainer execution, checkpoint saving/loading verified.", flush=True)

    print("\n==================================================")
    print("FOOT BASELINE MODEL TRAINING & PIPELINE VERIFICATION: PASSED")
    print("==================================================")

if __name__ == "__main__":
    verify_baseline_training_pipeline()
