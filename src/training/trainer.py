import sys
import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import os
import re

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Auto-increment the run version for a new training experiment
if "FUSIONMED_RUN_VERSION" not in os.environ:
    experiments_base = PROJECT_ROOT / "experiments"
    next_ver = 1
    if experiments_base.exists():
        existing_versions = []
        for path in experiments_base.iterdir():
            if path.is_dir():
                match = re.match(r"^v(\d+)_", path.name)
                if match:
                    existing_versions.append(int(match.group(1)))
        if existing_versions:
            next_ver = max(existing_versions) + 1
    os.environ["FUSIONMED_RUN_VERSION"] = f"v{next_ver:03d}"

import src.config as config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data.dataloader import create_dataloaders
from src.models.model_factory import load_model
from src.training.losses import get_loss_fn
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.train import train_epoch
from src.training.validate import validate_epoch

class Trainer:
    """
    Coordinator class to orchestrate training, validation, early stopping,
    checkpointing, TensorBoard logging, and history export.
    """
    
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        
        # 1. Setup seed and environment
        set_seed(config.SEED)
        
        # 2. Setup run directories
        self.run_dir = config.RUN_DIR
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Setup logger
        self.logger = setup_logger(
            name="Trainer",
            log_file=self.run_dir / "train.log"
        )
        self.logger.info(f"Initialized training run: {config.RUN_VERSION}_{config.RUN_NAME}")
        self.logger.info(f"Running on device: {config.DEVICE}")
        
        # 4. Dump experiment configuration
        self._dump_config()
        
        # 5. Initialize TensorBoard
        self.writer = SummaryWriter(log_dir=str(config.TENSORBOARD_DIR))
        
        # 6. Initialize DataLoaders
        self.logger.info("Initializing DataLoaders...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        
        # 7. Initialize Model
        self.logger.info(f"Loading model architecture '{config.MODEL_NAME}'...")
        self.model = load_model(
            name=config.MODEL_NAME,
            num_classes=config.NUM_CLASSES,
            pretrained=True
        )
        self.model = self.model.to(config.DEVICE)
        
        # 8. Initialize Loss, Optimizer, and Scheduler
        self.criterion = get_loss_fn(loss_type="ce")
        self.optimizer = get_optimizer(
            self.model.parameters(),
            opt_type="adamw",
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type="cosine",
            epochs=config.EPOCHS
        )
        
        # 9. Initialize Early Stopping
        self.early_stopping = EarlyStopping(
            patience=config.PATIENCE,
            mode="max"  # Monitoring validation QWK (maximizing)
        )
        
        # 10. Training Trace States
        self.start_epoch = 1
        self.best_qwk = -1.0
        self.best_f1 = -1.0
        self.best_epoch = 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_qwk": [],
            "val_f1": [],
            "lr": []
        }
        
    def _dump_config(self) -> None:
        """Serializes global config settings to a json file in the run directory."""
        config_dict = {}
        for key in dir(config):
            if key.isupper() and not key.startswith("__"):
                val = getattr(config, key)
                # Convert non-serializable objects (like Path) to string
                if isinstance(val, Path):
                    val = str(val)
                elif isinstance(val, set):
                    val = list(val)
                config_dict[key] = val
                
        with open(config.RUN_CONFIG_JSON, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        self.logger.info(f"Saved run configuration to {config.RUN_CONFIG_JSON}")
        
    def resume_from_checkpoint(self, checkpoint_path: Path) -> None:
        """Resumes training state from a saved checkpoint file."""
        self.logger.info(f"Resuming training state from checkpoint: {checkpoint_path}")
        checkpoint_data = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=config.DEVICE
        )
        
        self.start_epoch = checkpoint_data["epoch"] + 1
        self.best_qwk = checkpoint_data["best_qwk"]
        self.best_f1 = checkpoint_data["best_f1"]
        self.best_epoch = checkpoint_data["best_epoch"]
        self.history = checkpoint_data["history"]
        self.logger.info(
            f"Successfully restored weights. Next epoch: {self.start_epoch} | "
            f"Best QWK: {self.best_qwk:.4f} (Epoch {self.best_epoch})"
        )
        
    def train(self) -> None:
        """Executes the full training and validation loop."""
        self.logger.info("Starting baseline training pipeline...")
        total_epochs = 1 if self.dry_run else config.EPOCHS
        
        if self.dry_run:
            self.logger.warning("Dry-run mode active. Restricting run to 1 epoch and 2 batches.")
            
        scaler = torch.cuda.amp.GradScaler() if config.USE_AMP and config.DEVICE == "cuda" else None
        
        for epoch in range(self.start_epoch, total_epochs + 1):
            epoch_start_time = time.time()
            
            # --- Dry-run optimization ---
            if self.dry_run:
                # Wrap loaders in a subset for fast execution
                train_loader_subset = [next(iter(self.train_loader))] * 2
                val_loader_subset = [next(iter(self.val_loader))] * 2
            else:
                train_loader_subset = self.train_loader
                val_loader_subset = self.val_loader
                
            # 1. Train epoch
            train_loss, train_acc = train_epoch(
                model=self.model,
                loader=train_loader_subset,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=config.DEVICE,
                use_amp=config.USE_AMP,
                scaler=scaler
            )
            
            # 2. Validate epoch
            val_loss, val_metrics = validate_epoch(
                model=self.model,
                loader=val_loader_subset,
                criterion=self.criterion,
                device=config.DEVICE
            )
            
            # Step scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # 3. Log results
            epoch_duration = time.time() - epoch_start_time
            val_acc = val_metrics["accuracy"]
            val_qwk = val_metrics["qwk"]
            val_f1 = val_metrics["f1"]
            
            self.logger.info(
                f"Epoch [{epoch:02d}/{total_epochs:02d}] - "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Val QWK: {val_qwk:.4f} | LR: {current_lr:.6f} | "
                f"Time: {epoch_duration:.2f}s"
            )
            
            # 4. Update tensorboard
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/Val", val_acc, epoch)
            self.writer.add_scalar("Metrics/QWK", val_qwk, epoch)
            self.writer.add_scalar("Metrics/F1", val_f1, epoch)
            self.writer.add_scalar("Params/LearningRate", current_lr, epoch)
            self.writer.add_scalar("Params/EpochTime", epoch_duration, epoch)
            
            # 5. Append to history trace
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_qwk"].append(val_qwk)
            self.history["val_f1"].append(val_f1)
            self.history["lr"].append(current_lr)
            
            # 6. Save history to file
            self._save_history()
            
            # 7. Checkpointing
            is_best = False
            if val_qwk > self.best_qwk:
                self.best_qwk = val_qwk
                self.best_f1 = val_f1
                self.best_epoch = epoch
                is_best = True
                self.logger.info(f"[NEW BEST] New Best QWK: {val_qwk:.4f} achieved at epoch {epoch}!")
                
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "history": self.history,
                "best_qwk": self.best_qwk,
                "best_f1": self.best_f1,
                "best_epoch": self.best_epoch,
                "seed": config.SEED
            }
            
            save_checkpoint(
                state=checkpoint_state,
                checkpoint_dir=config.CHECKPOINT_DIR,
                is_best=is_best,
                epoch=epoch
            )
            
            # 8. Early Stopping check
            if self.early_stopping(val_qwk):
                self.logger.warning(
                    f"Early stopping triggered! No improvement for {config.PATIENCE} epochs. "
                    f"Halting training. Best epoch was {self.best_epoch} with QWK: {self.best_qwk:.4f}"
                )
                break
                
        self.writer.close()
        self.logger.info("Baseline training completed successfully!")
        
    def _save_history(self) -> None:
        """Writes current training history dict to CSV and JSON formats in the run directory."""
        df = pd.DataFrame(self.history)
        df.index.name = "epoch"
        df.index += 1  # 1-based index
        df.to_csv(config.RUN_HISTORY_CSV)
        
        # Write to JSON
        with open(config.RUN_HISTORY_JSON, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)

if __name__ == "__main__":
    # Support fast CPU debugging run
    trainer = Trainer(dry_run=True)
    trainer.train()
