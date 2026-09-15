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
from src.foot.models.factory import create_model, MODEL_REGISTRY
from src.foot.training.config import BaselineConfig
from src.foot.training.trainer import FootBaselineTrainer, set_reproducibility
from src.foot.training.losses import get_loss_function

CANDIDATE_MODELS = ["efficientnet_b0", "efficientnet_b3", "convnext_tiny", "swin_tiny", "vit_b16"]

def verify_candidate_instantiation():
    print("1. Verifying Candidate Model Instantiation & Output Shapes...", flush=True)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    for m_name in CANDIDATE_MODELS:
        model = create_model(model_name=m_name, num_classes=4, pretrained=False, device="cpu")
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
            
        logits = out["logits"] if isinstance(out, dict) else out
        assert logits.shape == (2, 4), f"Model {m_name} output shape mismatch: expected (2, 4), got {logits.shape}"
        
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0, f"Model {m_name} has 0 parameters!"
        print(f" [PASS] {m_name:18s} | Output shape: {tuple(logits.shape)} | Params: {param_count/1e6:6.2f}M", flush=True)
        
        del model
        gc.collect()

def verify_synthetic_benchmark_runner():
    print("\n2. Verifying Synthetic Benchmark Execution & Artifact Structure...", flush=True)
    set_reproducibility(SEED)
    
    dummy_x = torch.randn(12, 3, 224, 224)
    dummy_y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    ds_synthetic = torch.utils.data.TensorDataset(dummy_x, dummy_y)
    
    train_loader = torch.utils.data.DataLoader(ds_synthetic, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(ds_synthetic, batch_size=4, shuffle=False)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        for m_name in CANDIDATE_MODELS:
            m_exp_dir = tmp_path / m_name
            m_ckpt_dir = m_exp_dir / "checkpoints"
            
            m_config = BaselineConfig(
                model_name=m_name,
                loss_type="unweighted",
                dropout_rate=0.2,
                epochs=1,
                batch_size=4,
                seed=SEED,
                custom_experiment_dir=m_exp_dir
            )
            
            model = create_model(model_name=m_name, num_classes=4, pretrained=False, device="cpu")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
            criterion = nn.CrossEntropyLoss()
            
            trainer = FootBaselineTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                config=m_config
            )
            
            summary = trainer.train()
            
            best_ckpt = m_ckpt_dir / "best_model.pt"
            last_ckpt = m_ckpt_dir / "last_model.pt"
            config_json = m_exp_dir / "config.json"
            
            assert best_ckpt.exists(), f"best_model.pt missing for {m_name}"
            assert last_ckpt.exists(), f"last_model.pt missing for {m_name}"
            assert config_json.exists(), f"config.json missing for {m_name}"
            
            # Clean up explicit model references for fresh state verification
            del model, optimizer, scheduler, trainer
            gc.collect()
            print(f" [PASS] Synthetic runner verified for {m_name}.", flush=True)

def main():
    print("==================================================", flush=True)
    print("Foot Ulcer Architecture Benchmark Verification", flush=True)
    print("==================================================", flush=True)
    
    verify_candidate_instantiation()
    verify_synthetic_benchmark_runner()
    
    print("\n==================================================")
    print("FOOT BENCHMARK PIPELINE VERIFICATION: PASSED")
    print("==================================================")

if __name__ == "__main__":
    main()
