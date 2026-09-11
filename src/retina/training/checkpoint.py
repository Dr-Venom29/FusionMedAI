import sys
import torch
import torchvision
from pathlib import Path
from typing import Dict, Any, Optional

def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: Path,
    is_best: bool = False,
    epoch: Optional[int] = None
) -> None:
    """
    Saves training state checkpoint.
    Saves as 'last_model.pt', and if is_best=True, also saves as 'best_model.pt'.
    If epoch is provided, optionally saves epoch-specific checkpoints (e.g. 'epoch_10.pt').
    
    Args:
        state: Dict containing model/optimizer/scheduler weights and history.
        checkpoint_dir: Directory path to save weights.
        is_best: If True, copies state to best_model.pt
        epoch: Current epoch number.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Enrich state with metadata if not already present
    if "metadata" not in state:
        state["metadata"] = {}
    
    state["metadata"].update({
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "python_version": sys.version,
    })
    
    # Save last checkpoint
    last_path = checkpoint_dir / "last_model.pt"
    torch.save(state, last_path)
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(state, best_path)
        
    # Save epoch checkpoint (every 10 epochs as a standard checkpoint)
    if epoch is not None and epoch % 10 == 0:
        epoch_path = checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(state, epoch_path)

    # Save config.json alongside checkpoints to ensure portability
    config_src = checkpoint_dir.parent / "config.json"
    if config_src.exists():
        import shutil
        shutil.copy2(config_src, checkpoint_dir / "config.json")

def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Loads checkpoint state into the provided model, optimizer, and scheduler.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model: PyTorch model instance to load weights into.
        optimizer: PyTorch optimizer instance to load state into (optional).
        scheduler: PyTorch learning rate scheduler (optional).
        device: Device target to map tensors to.
        
    Returns:
        Dict[str, Any]: Dictionary containing loaded metadata, epoch, history, best_qwk, best_f1, and best_epoch.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_path}'")
        
    # Load state dict map
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(state["model_state_dict"])
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        
    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
        
    # Return training trace parameters
    return {
        "epoch": state.get("epoch", 0),
        "history": state.get("history", {}),
        "best_qwk": state.get("best_qwk", 0.0),
        "best_f1": state.get("best_f1", 0.0),
        "best_epoch": state.get("best_epoch", 0),
        "metadata": state.get("metadata", {}),
        "seed": state.get("seed", 42)
    }
