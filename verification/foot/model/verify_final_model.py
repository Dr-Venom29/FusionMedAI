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

from src.foot.config import SEED, NUM_CLASSES, CLASS_NAMES
from src.foot.models import (
    FootEfficientNetB3,
    build_foot_final_model,
    load_foot_final_model,
    FINAL_FOOT_MODEL_CHECKPOINT
)
from src.foot.training.checkpoint import CheckpointManager

def verify_final_model_instantiation():
    print("1. Verifying Final Model Architecture Instantiation (EfficientNet-B3)...", flush=True)
    model = build_foot_final_model(num_classes=4, pretrained=False, device="cpu")
    
    assert isinstance(model, FootEfficientNetB3), "build_foot_final_model did not return FootEfficientNetB3 instance!"
    param_count = sum(p.numel() for p in model.parameters())
    assert 10.0e6 <= param_count <= 11.5e6, f"Expected ~10.70M parameters, got {param_count/1e6:.2f}M"
    
    dummy_input = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        out = model(dummy_input)
        
    logits = out["logits"] if isinstance(out, dict) else out
    assert logits.shape == (2, 4), f"Expected logits shape (2, 4), got {logits.shape}"
    print(f" [PASS] Final Model Instantiated: FootEfficientNetB3 | Params: {param_count/1e6:.2f}M | Logits: {tuple(logits.shape)}", flush=True)
    del model
    gc.collect()

def verify_final_checkpoint_contract():
    print("\n2. Verifying Final Checkpoint Loading & Deterministic Inference Contract...", flush=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ckpt_mgr = CheckpointManager(checkpoint_dir=tmp_path)
        
        # Instantiate and save synthetic final model checkpoint
        orig_model = build_foot_final_model(num_classes=4, pretrained=False, device="cpu")
        optimizer = torch.optim.AdamW(orig_model.parameters(), lr=1e-4)
        saved_ckpt = ckpt_mgr.save(
            model=orig_model,
            optimizer=optimizer,
            scheduler=None,
            epoch=1,
            best_val_loss=0.8779,
            best_val_macro_f1=0.6683,
            config_dict={"model_name": "efficientnet_b3", "seed": SEED},
            seed=SEED,
            is_best=True,
            checkpoint_metric="val_loss"
        )
        
        # Restore via load_foot_final_model
        restored_model = load_foot_final_model(checkpoint_path=saved_ckpt, device="cpu")
        assert not restored_model.training, "Loaded final model must be in eval mode (model.training must be False)!"
        
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.inference_mode():
            output = restored_model(dummy_input)
            
        assert isinstance(output, dict), "Output must be a dictionary!"
        assert "logits" in output and "probs" in output and "predicted_class" in output, "Missing key outputs in model response!"
        
        logits = output["logits"]
        probs = output["probs"]
        preds = output["predicted_class"]
        
        assert logits.shape == (1, 4), f"Expected logits shape (1, 4), got {logits.shape}"
        assert probs.shape == (1, 4), f"Expected probs shape (1, 4), got {probs.shape}"
        assert preds.shape == (1,), f"Expected predicted_class shape (1,), got {preds.shape}"
        assert not torch.isnan(logits).any(), "Logits contain NaN values!"
        assert not torch.isinf(logits).any(), "Logits contain Inf values!"
        
        prob_sum = probs.sum(dim=1).item()
        assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {prob_sum}"
        
        print(f" [PASS] Final Model Inference Contract Verified (Probabilities Sum: {prob_sum:.4f}, Class: {preds.item()})", flush=True)
        del orig_model, restored_model
        gc.collect()

def main():
    print("==================================================", flush=True)
    print("Foot Ulcer Final Model Contract Verification", flush=True)
    print("==================================================", flush=True)
    
    verify_final_model_instantiation()
    verify_final_checkpoint_contract()
    
    print("\n==================================================")
    print("FOOT FINAL MODEL CONTRACT VERIFICATION: PASSED")
    print("==================================================")

if __name__ == "__main__":
    main()
