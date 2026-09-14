import os
import sys
import tempfile
from pathlib import Path
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.training.config import BaselineConfig
from src.foot.models.factory import create_model
from src.foot.training.losses import get_loss_function
from src.foot.training.metrics import compute_baseline_metrics
from src.foot.training.trainer import FootBaselineTrainer, set_reproducibility

def verify_baseline_framework():
    print("==================================================")
    print("Verifying Modular Baseline Framework (ResNet50 / EfficientNet)")
    print("==================================================")
    
    set_reproducibility(42)
    
    # 1. Verify Model Factory & Output Interface
    print("1. Testing ResNet50 baseline model factory & output formatting...", flush=True)
    model = create_model(name="resnet50", num_classes=4, pretrained=False, device="cpu")
    dummy_x = torch.randn(4, 3, 224, 224)
    out = model(dummy_x)
    
    assert isinstance(out, dict), "Model output must be a dictionary!"
    assert "logits" in out and "probs" in out and "predicted_class" in out, "Missing required keys in model output dict!"
    assert out["logits"].shape == (4, 4), f"Incorrect logits shape: {out['logits'].shape}"
    assert out["probs"].shape == (4, 4), f"Incorrect probs shape: {out['probs'].shape}"
    assert out["predicted_class"].shape == (4,), f"Incorrect predicted_class shape: {out['predicted_class'].shape}"
    print(" [PASS] ResNet50 model output dict interface verified.")
    
    # 2. Verify Loss Function Factory
    print("2. Testing Loss Function Factory (unweighted vs weighted)...", flush=True)
    loss_unweighted = get_loss_function("unweighted", device="cpu")
    loss_weighted = get_loss_function("weighted", class_weights=[1.0870, 1.0669, 1.0000, 1.0731], device="cpu")
    assert loss_unweighted.weight is None, "Unweighted loss should have None weight!"
    assert loss_weighted.weight is not None, "Weighted loss should have non-None weight!"
    print(" [PASS] Loss Function Factory verified.")
    
    # 3. Verify Metrics & Error Analysis
    print("3. Testing Metrics computation & Error Analysis...", flush=True)
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 2])
    y_probs = np.eye(4)[y_pred]
    
    metrics = compute_baseline_metrics(y_true, y_pred, y_probs)
    assert "accuracy" in metrics and "macro_f1" in metrics and "class_metrics" in metrics, "Incomplete metrics dict!"
    assert metrics["confusion_matrix"][3][3] == 1, "Incorrect confusion matrix element!"
    print(f" [PASS] Metrics verified (Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}).")

    # 4. Verify 1-Epoch Synthetic Trainer Run
    print("4. Testing 1-Epoch Trainer execution with Checkpointing & Error Analysis...", flush=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = BaselineConfig(
            model_name="resnet50",
            loss_type="unweighted",
            epochs=1,
            batch_size=4,
            experiments_dir=tmp_path
        )
        
        dummy_y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(dummy_x, dummy_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        criterion = get_loss_function("unweighted")
        
        trainer = FootBaselineTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            config=config
        )
        
        summary = trainer.train()
        df_errors = trainer.perform_error_analysis(loader)
        
        assert (config.checkpoint_dir / "best_model.pt").exists(), "best_model.pt not saved!"
        assert (config.experiment_dir / "training_history.csv").exists(), "training_history.csv not saved!"
        assert (config.experiment_dir / "error_analysis.csv").exists(), "error_analysis.csv not saved!"
        print(" [PASS] 1-Epoch Trainer execution, checkpointing, and error analysis verified.")
        
    print("\n==================================================")
    print("MODULAR BASELINE FRAMEWORK: VERIFIED & PASSED")
    print("==================================================")

if __name__ == "__main__":
    verify_baseline_framework()
