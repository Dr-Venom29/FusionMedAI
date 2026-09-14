import os
import sys
import tempfile
import gc
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.config import SEED, NUM_CLASSES
from src.foot.models.factory import create_model, MODEL_REGISTRY
from src.foot.training.checkpoint import CheckpointManager

CANDIDATE_MODELS = ["resnet50", "efficientnet_b0", "efficientnet_b3", "convnext_tiny", "swin_tiny", "vit_b16"]

def verify_checkpoint_saving_and_loading():
    print("==================================================", flush=True)
    print("Phase 10.5 — Foot Ulcer Checkpoint Restoration Verification", flush=True)
    print("==================================================", flush=True)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        for m_name in CANDIDATE_MODELS:
            print(f"\nVerifying checkpoint flow for: {m_name.upper()}...", flush=True)
            ckpt_mgr = CheckpointManager(checkpoint_dir=tmp_path / m_name)
            
            # 1. Instantiate Model & Optimizer
            model_orig = create_model(model_name=m_name, num_classes=4, pretrained=False, device="cpu")
            optimizer_orig = torch.optim.AdamW(model_orig.parameters(), lr=1e-4)
            scheduler_orig = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_orig, T_max=10)
            
            # 2. Save Best Checkpoint
            saved_path = ckpt_mgr.save(
                model=model_orig,
                optimizer=optimizer_orig,
                scheduler=scheduler_orig,
                epoch=5,
                best_val_loss=0.421,
                best_val_macro_f1=0.750,
                config_dict={"model_name": m_name, "seed": SEED},
                seed=SEED,
                is_best=True,
                checkpoint_metric="val_loss"
            )
            
            assert saved_path.exists(), f"Failed to save checkpoint at {saved_path}"
            print(f" [PASS] Checkpoint successfully saved to: {saved_path.name}", flush=True)
            
            # 3. Reload into fresh model instance
            model_fresh = create_model(model_name=m_name, num_classes=4, pretrained=False, device="cpu")
            checkpoint_data = ckpt_mgr.load(model=model_fresh, checkpoint_path=saved_path, device="cpu")
            
            assert "model_state_dict" in checkpoint_data, "Checkpoint missing 'model_state_dict'"
            assert checkpoint_data["epoch"] == 5, f"Expected epoch 5, got {checkpoint_data['epoch']}"
            assert checkpoint_data["best_val_loss"] == 0.421, f"Expected best_val_loss 0.421, got {checkpoint_data['best_val_loss']}"
            assert checkpoint_data["best_val_macro_f1"] == 0.750, f"Expected best_val_macro_f1 0.750, got {checkpoint_data['best_val_macro_f1']}"
            assert checkpoint_data["checkpoint_metric"] == "val_loss", f"Expected checkpoint_metric 'val_loss'"
            print(f" [PASS] Self-describing checkpoint state loaded into fresh {m_name} instance.", flush=True)
            
            # 4. Perform Inference Verification on Dummy Input
            dummy_input = torch.randn(1, 3, 224, 224)
            model_fresh.eval()
            with torch.no_grad():
                out = model_fresh(dummy_input)
                logits = out["logits"] if isinstance(out, dict) else out
                
            assert logits.shape == (1, 4), f"Inference output shape mismatch: expected (1, 4), got {logits.shape}"
            assert not torch.isnan(logits).any(), f"Inference returned NaN values for model {m_name}"
            assert not torch.isinf(logits).any(), f"Inference returned Inf values for model {m_name}"
            print(f" [PASS] Inference contract verified (Logits: {logits.detach().numpy().round(4).tolist()})", flush=True)
            
            del model_orig, model_fresh, optimizer_orig, scheduler_orig
            gc.collect()

    print("\n==================================================")
    print("FOOT CHECKPOINT RESTORATION & INFERENCE VERIFICATION: PASSED")
    print("==================================================")

if __name__ == "__main__":
    verify_checkpoint_saving_and_loading()
