import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.foot.training.config import BaselineConfig
from src.foot.training.losses import get_loss_function
from src.foot.training.metrics import compute_baseline_metrics
from src.foot.training.checkpoint import CheckpointManager

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
    Manages AdamW optimization, CosineAnnealingLR scheduling, Loss execution,
    Validation evaluation, Early Stopping, Checkpointing, and Error Analysis.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        criterion: nn.Module,
        config: BaselineConfig
    ):
        self.config = config
        self.device = torch.device(config.device) if isinstance(config.device, str) else config.device
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Enforce device placement across model and criterion (Phase 10.4 Device Safety)
        self.model = model.to(self.device)
        if hasattr(criterion, "to"):
            self.criterion = criterion.to(self.device)
        else:
            self.criterion = criterion
        
        set_reproducibility(self.config.seed)
        self.checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)

    def _unpack_output(self, output: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(output, dict):
            logits = output["logits"]
            probs = output.get("probs", torch.softmax(logits, dim=1))
            preds = output.get("predicted_class", torch.argmax(probs, dim=1))
        else:
            logits = output
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return logits, probs, preds

    def train_one_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            else:
                images, labels = batch["image"], batch["label"]
                
            images = images.to(self.device)
            labels = labels.to(self.device, dtype=torch.long)
            
            self.optimizer.zero_grad()
            output = self.model(images)
            logits, _, _ = self._unpack_output(output)
            
            loss = self.criterion(logits, labels)
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
        
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            else:
                images, labels = batch["image"], batch["label"]
                
            images = images.to(self.device)
            labels = labels.to(self.device, dtype=torch.long)
            
            output = self.model(images)
            logits, probs, preds = self._unpack_output(output)
            loss = self.criterion(logits, labels)
            
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
        avg_loss = running_loss / max(1, total_samples)
        metrics = compute_baseline_metrics(
            y_true=np.array(all_targets),
            y_pred=np.array(all_preds),
            y_probs=np.array(all_probs),
            class_names=self.config.class_names
        )
        
        return avg_loss, metrics

    def train(self) -> Dict[str, Any]:
        print("==================================================")
        print(f"Starting Foot DFU Baseline Training [{self.config.model_name.upper()} - {self.config.loss_type.upper()}]")
        print("==================================================")
        print(f" Device: {self.device} | Epochs: {self.config.epochs} | Seed: {self.config.seed}")
        print(f" Checkpoint Directory: {self.config.checkpoint_dir}")
        
        best_val_loss = float("inf")
        best_val_macro_f1 = 0.0
        patience_counter = 0
        best_epoch = 0
        
        history = []
        start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
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
                "epoch_time_sec": ep_duration
            }
            history.append(epoch_log)
            
            print(
                f"Epoch [{epoch:02d}/{self.config.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Macro F1: {val_macro_f1:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | Time: {ep_duration}s",
                flush=True
            )
            
            # Checkpoint selection criterion: Lowest Validation Loss
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_val_macro_f1 = val_macro_f1
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                
            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                best_metric=best_val_macro_f1,
                config_dict={"model": self.config.model_name, "loss_type": self.config.loss_type},
                seed=self.config.seed,
                is_best=is_best
            )
            
            if patience_counter >= self.config.patience:
                print(f"\n Early stopping triggered at epoch {epoch} (No improvement for {self.config.patience} epochs).")
                break
                
        total_training_time = round(time.time() - start_time, 2)
        print(f"\nTraining Complete in {total_training_time}s. Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f}, Val Macro F1: {best_val_macro_f1:.4f})")
        
        # Save training history
        history_df = pd.DataFrame(history)
        history_csv_path = self.config.experiment_dir / "training_history.csv"
        history_df.to_csv(history_csv_path, index=False)
        
        summary = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_macro_f1": best_val_macro_f1,
            "total_epochs": len(history),
            "total_training_time_sec": total_training_time,
            "training_history": history
        }
        
        with open(self.config.experiment_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        return summary

    @torch.no_grad()
    def perform_error_analysis(self, test_loader: DataLoader) -> pd.DataFrame:
        """
        Executes Quality-Stratified Error Analysis on the test dataset partition (Phase 10.4.12 & 10.4.13).
        Records false classifications, predicted probabilities, true labels, and metadata.
        """
        self.model.eval()
        error_records = []
        
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    images, labels, metadata_list = batch[0], batch[1], batch[2]
                else:
                    images, labels = batch[0], batch[1]
                    metadata_list = None
            else:
                images = batch["image"]
                labels = batch["label"]
                metadata_list = batch.get("metadata", None)
                
            images = images.to(self.device)
            labels = labels.to(self.device, dtype=torch.long)
            
            output = self.model(images)
            _, probs, preds = self._unpack_output(output)
            
            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            targets_np = labels.cpu().numpy()
            
            for i in range(len(targets_np)):
                true_lbl = targets_np[i]
                pred_lbl = preds_np[i]
                
                # Check for misclassification or record all samples
                is_error = bool(true_lbl != pred_lbl)
                rec = {
                    "true_label": int(true_lbl),
                    "pred_label": int(pred_lbl),
                    "is_error": is_error,
                    "prob_g1": float(round(probs_np[i][0], 4)),
                    "prob_g2": float(round(probs_np[i][1], 4)),
                    "prob_g3": float(round(probs_np[i][2], 4)),
                    "prob_g4": float(round(probs_np[i][3], 4)),
                    "confidence": float(round(probs_np[i][pred_lbl], 4))
                }
                
                if metadata_list is not None and i < len(metadata_list):
                    m = metadata_list[i]
                    if isinstance(m, dict):
                        rec["id_code"] = m.get("id_code", "")
                        rec["image_path"] = m.get("image_path", "")
                        rec["source_image_id"] = m.get("source_image_id", "")
                        
                error_records.append(rec)
                
        df_errors = pd.DataFrame(error_records)
        error_csv_path = self.config.experiment_dir / "error_analysis.csv"
        df_errors.to_csv(error_csv_path, index=False)
        print(f" Error Analysis saved to: {error_csv_path} ({df_errors['is_error'].sum()} errors / {len(df_errors)} total test samples)")
        return df_errors
