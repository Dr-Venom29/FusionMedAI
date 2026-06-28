import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import src.config as config
from src.models.model_factory import load_model
from src.training.losses import get_loss_fn
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler
from src.training.train import train_epoch
from src.training.validate import validate_epoch
from src.training.early_stopping import EarlyStopping
from src.training.metrics import calculate_metrics

def verify_training():
    """
    Verification script for the training loop, validation metrics calculation,
    optimizers, learning rate schedulers, and early stopping.
    """
    print("\n==================================================")
    print("Verification: Training and Validation Pipeline")
    print("==================================================")
    
    device = config.DEVICE
    print(f"Target device: {device}")
    
    # 1. Mock Datasets and Loaders
    print("Creating mock dataset (4 samples, shape [3, 224, 224])...")
    dummy_images = torch.randn(4, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    dummy_labels = torch.randint(0, config.NUM_CLASSES, (4,))
    
    dataset = TensorDataset(dummy_images, dummy_labels)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # 2. Setup Model, Loss, Optimizer, and Scheduler
    print("Loading model and training components...")
    model = load_model(name=config.MODEL_NAME, num_classes=config.NUM_CLASSES, pretrained=False)
    model = model.to(device)
    
    criterion = get_loss_fn(loss_type="ce")
    optimizer = get_optimizer(model.parameters(), opt_type="adamw", lr=1e-4, weight_decay=1e-4)
    scheduler = get_scheduler(optimizer, scheduler_type="cosine", epochs=5)
    
    # Save initial weights for optimization verification
    initial_params = [p.clone() for p in model.parameters() if p.requires_grad]
    
    # 3. Verify single training epoch step
    print("Verifying training step (1 epoch, 2 batches)...")
    scaler = GradScaler() if config.USE_AMP and device == "cuda" else None
    
    train_loss, train_acc = train_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        use_amp=config.USE_AMP,
        scaler=scaler
    )
    print(f"[OK] Training epoch complete: Loss = {train_loss:.4f} | Acc = {train_acc:.4f}")
    
    # Verify optimizer updated parameters
    updated_params = [p for p in model.parameters() if p.requires_grad]
    params_updated = False
    for p_init, p_upd in zip(initial_params, updated_params):
        if not torch.equal(p_init.to(device), p_upd):
            params_updated = True
            break
    
    assert params_updated, "Optimizer failed to update model parameters"
    print("[OK] Optimizer parameter updates")
    
    # 4. Verify learning rate scheduler
    init_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    new_lr = optimizer.param_groups[0]["lr"]
    print(f"[OK] Scheduler step: LR changed from {init_lr:.6f} to {new_lr:.6f}")
    assert init_lr != new_lr, "Scheduler failed to update learning rate"
    
    # 5. Verify validation epoch
    print("Verifying validation step (no gradients, metrics calculation)...")
    val_loss, val_metrics = validate_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device
    )
    print(f"[OK] Validation complete: Loss = {val_loss:.4f}")
    print(f"[OK] Validation Metrics Calculated: {list(val_metrics.keys())}")
    
    # Assert key metrics are calculated
    for metric_name in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "qwk", "specificity"]:
        assert metric_name in val_metrics, f"Missing metric '{metric_name}' in validation outputs"
    print("[OK] Metrics verification")
    
    # 6. Verify Early stopping monitor
    print("Testing Early Stopping tracker (mode=max, patience=2)...")
    early_stop = EarlyStopping(patience=2, mode="max")
    
    # Simulate scores
    stop_1 = early_stop(0.5) # Initial score
    stop_2 = early_stop(0.6) # Improvement (reset counter)
    stop_3 = early_stop(0.55) # No improvement
    stop_4 = early_stop(0.55) # Trigger stopping
    
    print(f"[OK] Early stopping progression triggers: [{stop_1}, {stop_2}, {stop_3}, {stop_4}]")
    assert stop_4 == True, "Early stopping logic failed to halt training"
    print("[OK] Early stopping assertion")
    
    print("\n=== TRAINING PIPELINE VERIFICATION SUCCESSFUL ===")
    print("==================================================\n")

if __name__ == "__main__":
    verify_training()
