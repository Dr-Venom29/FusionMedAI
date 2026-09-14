import torch
from pathlib import Path
from typing import Dict, Any, Optional

class CheckpointManager:
    """
    Manages exact restoration and checkpoint saving for Foot baseline experiments (Phase 10.4.9).
    """
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_model_path = self.checkpoint_dir / "best_model.pt"
        self.last_model_path = self.checkpoint_dir / "last_model.pt"

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        best_metric: float,
        config_dict: Dict[str, Any],
        seed: int,
        is_best: bool = False
    ) -> Path:
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric,
            "config": config_dict,
            "seed": seed
        }
        
        # Save last model checkpoint
        torch.save(state, self.last_model_path)
        
        # Save best model checkpoint if requested
        if is_best:
            torch.save(state, self.best_model_path)
            return self.best_model_path
            
        return self.last_model_path

    def load(self, model: torch.nn.Module, checkpoint_path: Optional[Path] = None, device: str = "cpu") -> Dict[str, Any]:
        target_path = checkpoint_path or self.best_model_path
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: '{target_path}'")
            
        checkpoint = torch.load(target_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
