import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.config import (
    SEED,
    CLASS_NAMES,
    FOOT_EXPERIMENTS_DIR,
    FOOT_RESULTS_DIR,
    METADATA_DIR
)
from src.foot.model.evaluator import compute_evaluation_metrics

def set_reproducibility(seed: int = 42):
    """Sets random seeds across Python, NumPy, PyTorch CPU and PyTorch CUDA."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class FootBaselineTrainer:
    """
    Deterministic Baseline Trainer for Foot DFU Wagner 4-Class Classification.
    Manages AdamW optimization, CosineAnnealingLR scheduling, Sqrt Inverse Frequency Weighted Loss,
    Validation evaluation, Early Stopping, Checkpointing, and Metric Logging.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        device: str = "cpu",
        epochs: int = 20,
        patience: int = 10,
        checkpoint_dir: Path = None,
        seed: int = 42
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        
        set_reproducibility(self.seed)
        
        if checkpoint_dir is None:
            self.checkpoint_dir = FOOT_EXPERIMENTS_DIR / "baseline" / "checkpoints"
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
            
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_one_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device, dtype=torch.long)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(self.train_loader):
                print(f"   Batch [{batch_idx+1:03d}/{len(self.train_loader):03d}] - Loss: {loss.item():.4f}", flush=True)
            
        return running_loss / max(1, total_samples)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device, dtype=torch.long)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
        avg_loss = running_loss / max(1, total_samples)
        metrics = compute_evaluation_metrics(
            y_true=np.array(all_targets),
            y_pred=np.array(all_preds),
            y_probs=np.array(all_probs),
            class_names=CLASS_NAMES
        )
        
        return avg_loss, metrics

    def train(self) -> Dict[str, Any]:
        print("==================================================")
        print("Starting Foot DFU Wagner 4-Class Baseline Model Training")
        print("==================================================")
        print(f" Device: {self.device} | Epochs: {self.epochs} | Seed: {self.seed}")
        print(f" Checkpoint Directory: {self.checkpoint_dir}")
        
        best_val_loss = float("inf")
        best_val_macro_f1 = 0.0
        patience_counter = 0
        best_epoch = 0
        
        history = []
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            ep_start = time.time()
            train_loss = self.train_one_epoch()
            val_loss, val_metrics = self.evaluate(self.val_loader)
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            current_lr = self.optimizer.param_groups[0]["lr"]
            val_macro_f1 = val_metrics["macro_f1"]
            val_acc = val_metrics["accuracy"]
            ep_duration = round(time.time() - ep_start, 2)
            
            epoch_log = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_macro_f1": val_macro_f1,
                "val_accuracy": val_acc,
                "val_weighted_f1": val_metrics["weighted_f1"],
                "learning_rate": current_lr,
                "duration_seconds": ep_duration
            }
            history.append(epoch_log)
            
            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Macro F1: {val_macro_f1:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} ({ep_duration}s)",
                flush=True
            )
            
            # Checkpoint saving on lowest val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_macro_f1 = val_macro_f1
                best_epoch = epoch
                patience_counter = 0
                
                best_ckpt_path = self.checkpoint_dir / "best_model.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_macro_f1": val_macro_f1,
                    "metrics": val_metrics
                }, best_ckpt_path)
            else:
                patience_counter += 1
                
            # Save latest checkpoint
            latest_ckpt_path = self.checkpoint_dir / "latest_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "val_macro_f1": val_macro_f1
            }, latest_ckpt_path)
            
            if patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (Patience = {self.patience}).")
                break
                
        total_training_time = round(time.time() - start_time, 2)
        print("\n==================================================")
        print("Baseline Model Training Completed!")
        print(f" Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f} | Best Val Macro F1: {best_val_macro_f1:.4f}")
        print(f" Total Training Time: {total_training_time} seconds")
        print("==================================================")
        
        # Save history logs
        df_hist = pd.DataFrame(history)
        hist_csv = self.checkpoint_dir / "training_history.csv"
        df_hist.to_csv(hist_csv, index=False)
        
        summary_report = {
            "model_name": "EfficientNet-B0 Baseline",
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "best_val_macro_f1": round(best_val_macro_f1, 4),
            "total_epochs": len(history),
            "total_training_seconds": total_training_time,
            "training_history": history
        }
        
        hist_json = self.checkpoint_dir / "training_history.json"
        with open(hist_json, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2)
            
        return summary_report
