import sys
from pathlib import Path
import torch

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import src.config as config
from src.models.model_factory import load_model
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler

def verify_checkpoint():
    """
    Verification script for saving, restoring, and checking state consistency
    of model checkpoints, optimizer, scheduler, and reproducibility metadata.
    """
    print("\n==================================================")
    print("Verification: Checkpoint Save and Restore")
    print("==================================================")
    
    device = "cpu"  # Keep verification on CPU for quick local check
    temp_dir = config.CHECKPOINT_DIR
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize model, optimizer, scheduler
    print("Initializing components for mock checkpoint save...")
    model = load_model(name=config.MODEL_NAME, num_classes=config.NUM_CLASSES, pretrained=False)
    optimizer = get_optimizer(model.parameters(), opt_type="adamw", lr=1e-4)
    scheduler = get_scheduler(optimizer, scheduler_type="cosine", epochs=10)
    
    # Track initial parameter values
    orig_params = [p.clone() for p in model.parameters()]
    
    # Mock history dictionary
    mock_history = {
        "train_loss": [0.5, 0.4],
        "val_loss": [0.6, 0.5],
        "val_qwk": [0.1, 0.3]
    }
    
    # 2. Package checkpoint state
    checkpoint_state = {
        "epoch": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": mock_history,
        "best_qwk": 0.3,
        "best_f1": 0.25,
        "best_epoch": 2,
        "seed": config.SEED
    }
    
    # 3. Save checkpoint
    print("Saving mock checkpoint weights...")
    save_checkpoint(
        state=checkpoint_state,
        checkpoint_dir=temp_dir,
        is_best=True,  # Saves both last_model.pt and best_model.pt
        epoch=2
    )
    
    best_file = temp_dir / "best_model.pt"
    last_file = temp_dir / "last_model.pt"
    
    assert best_file.exists(), "best_model.pt file was not saved"
    assert last_file.exists(), "last_model.pt file was not saved"
    print("[OK] Checkpoint files saved successfully")
    
    # 4. Mutate model weights to verify load restores them
    print("Mutating model parameters (simulating new session/training progress)...")
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0) # mutate weights
            
    # Verify weights are now mutated
    mutated = False
    for p_orig, p_curr in zip(orig_params, model.parameters()):
        if not torch.equal(p_orig, p_curr):
            mutated = True
            break
    assert mutated, "Weight mutation verification failed (weights remained identical)"
    
    # 5. Restore checkpoint
    print("Restoring model from best_model.pt checkpoint...")
    restored_data = load_checkpoint(
        checkpoint_path=best_file,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    # 6. Assert consistency
    print("Asserting state consistency...")
    # Verify parameters are restored to original values
    for idx, (p_orig, p_rest) in enumerate(zip(orig_params, model.parameters())):
        assert torch.allclose(p_orig, p_rest), f"Param index {idx} was not successfully restored"
    print("[OK] Model weights restored")
    
    # Verify metadata and history trace
    assert restored_data["epoch"] == 2, f"Expected epoch 2, got {restored_data['epoch']}"
    assert restored_data["best_qwk"] == 0.3, f"Expected best_qwk 0.3, got {restored_data['best_qwk']}"
    assert restored_data["best_epoch"] == 2, f"Expected best_epoch 2, got {restored_data['best_epoch']}"
    assert restored_data["seed"] == config.SEED, f"Expected seed {config.SEED}, got {restored_data['seed']}"
    assert "torch_version" in restored_data["metadata"], "Expected PyTorch version to be recorded in metadata"
    print("[OK] Checkpoint metadata consistency checks")
    
    # Clean up files created
    print("Cleaning up verification temp files...")
    best_file.unlink()
    last_file.unlink()
    
    print("\n=== CHECKPOINT SAVE/LOAD VERIFICATION SUCCESSFUL ===")
    print("==================================================\n")

if __name__ == "__main__":
    verify_checkpoint()
