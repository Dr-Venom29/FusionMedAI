import os
import sys
import gc
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

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
from src.training.metrics import calculate_metrics
from src.utils.visualization import plot_confusion_matrix

# Dynamic import helper for profiling libraries
def profile_model_complexity(model, device):
    """
    Attempts to profile FLOPs, MACs, and Parameter count using ptflops, fvcore, or torchinfo.
    Returns:
        dict: containing 'macs', 'flops', 'params_count'
    """
    profile = {"macs": "N/A", "flops": "N/A", "params_count": "N/A"}
    
    # 1. Total params count fallback
    try:
        profile["params_count"] = sum(p.numel() for p in model.parameters())
    except Exception:
        pass
        
    # 2. Try ptflops
    try:
        from ptflops import get_model_complexity_info
        # ptflops needs input on the same device or CPU
        with torch.no_grad():
            macs, params = get_model_complexity_info(
                model.to(device), (3, config.IMAGE_SIZE, config.IMAGE_SIZE),
                as_strings=False, print_per_layer_stat=False
            )
            profile["macs"] = int(macs)
            profile["flops"] = int(macs * 2) # Flops approx 2 * MACs
            profile["params_count"] = int(params)
            return profile
    except ImportError:
        pass
    except Exception:
        pass
        
    # 3. Try torchinfo
    try:
        from torchinfo import summary
        with torch.no_grad():
            model_stats = summary(
                model.to(device),
                input_size=(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE),
                verbose=0
            )
            profile["macs"] = int(model_stats.total_mult_adds)
            profile["flops"] = int(model_stats.total_mult_adds * 2)
            profile["params_count"] = int(model_stats.total_params)
            return profile
    except ImportError:
        pass
    except Exception:
        pass
        
    return profile


class BenchmarkRunner:
    """
    Scientific benchmarking coordinator that trains and evaluates multiple model
    architectures under identical experimental settings.
    """
    
    def __init__(self, dry_run: bool = False, model_name: str | None = None) -> None:
        self.dry_run = dry_run
        self.model_name = model_name
        
        # Setup directories
        config.BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        config.BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        config.BENCHMARK_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        config.BENCHMARK_TABLES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = setup_logger(
            name="BenchmarkRunner",
            log_file=config.BENCHMARK_RESULTS_DIR / "benchmark.log"
        )
        self.logger.info("Initializing Step 5 Model Benchmarking stage...")
        
        # Load Frozen Configuration
        self.config_path = PROJECT_ROOT / "benchmark_config.json"
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Frozen benchmark configuration not found at {self.config_path}. Run task initialization first."
            )
            
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.bench_config = json.load(f)
            
        self.logger.info(f"Loaded frozen benchmark configuration: {self.bench_config}")
        
        # Set seed for reproducibility
        set_seed(self.bench_config["seed"])
        
        # Initialize DataLoaders
        self.logger.info("Initializing unified DataLoaders...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            batch_size=self.bench_config["batch_size"],
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        
        # Compute Dynamic Class Weights on Training Dataset
        train_df = self.train_loader.dataset.dataframe
        class_counts = train_df[config.LABEL_COLUMN].value_counts().sort_index().values
        self.logger.info(f"Training class distribution: {dict(zip(config.CLASS_NAMES, class_counts))}")
        
        total_samples = len(train_df)
        num_classes = config.NUM_CLASSES
        # Inverse frequency class weighting
        self.class_weights = total_samples / (num_classes * class_counts)
        # Normalize weights to sum up to num_classes
        self.class_weights = self.class_weights / self.class_weights.sum() * num_classes
        self.class_weights_tensor = torch.tensor(self.class_weights, dtype=torch.float).to(config.DEVICE)
        self.logger.info(f"Dynamic class weights computed: {self.class_weights}")
        
        all_models = [
            "efficientnet_b0",
            "efficientnet_b3",
            "convnext_tiny",
            "swin_tiny",
            "vit_b16"
        ]
        
        if model_name:
            if model_name not in all_models:
                raise ValueError(f"Unknown model: {model_name}")
            self.architectures = [model_name]
        else:
            self.architectures = all_models
        
    def run(self) -> None:
        results_list = []
        roc_data = {}
        pr_data = {}
        
        for arch in self.architectures:
            self.logger.info(f"\n==================================================")
            self.logger.info(f"Benchmarking Model Architecture: {arch}")
            self.logger.info(f"==================================================")
            
            # Setup experiment specific folders
            exp_dir = config.BENCHMARK_DIR / arch
            ckpt_dir = exp_dir / "checkpoints"
            tb_dir = exp_dir / "tensorboard"
            
            for directory in [exp_dir, ckpt_dir, tb_dir]:
                directory.mkdir(parents=True, exist_ok=True)
                
            # Copy configuration copy into experiment directory
            with open(exp_dir / "benchmark_config.json", "w", encoding="utf-8") as f:
                json.dump(self.bench_config, f, indent=4)
                
            # Free VRAM/RAM before initializing model
            self._free_memory()
            
            # Load Model
            model = load_model(name=arch, num_classes=config.NUM_CLASSES, pretrained=True)
            model = model.to(config.DEVICE)
            
            # Profile Model Complexity (FLOPs, MACs, Params)
            complexity = profile_model_complexity(model, config.DEVICE)
            self.logger.info(
                f"Model Profile -> Params: {complexity['params_count']:,} | "
                f"MACs: {complexity['macs']} | FLOPs: {complexity['flops']}"
            )
            
            # Setup optimizer, scheduler, and early stopping
            criterion = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
            optimizer = get_optimizer(
                model.parameters(),
                opt_type=self.bench_config["optimizer"],
                lr=self.bench_config["learning_rate"],
                weight_decay=self.bench_config["weight_decay"]
            )
            scheduler = get_scheduler(
                optimizer,
                scheduler_type="cosine",
                epochs=self.bench_config["epochs"]
            )
            early_stopping = EarlyStopping(
                patience=self.bench_config["early_stopping_patience"],
                mode="max"
            )
            
            # TensorBoard writer
            writer = SummaryWriter(log_dir=str(tb_dir))
            
            # Dry-run modifications
            epochs_limit = 1 if self.dry_run else self.bench_config["epochs"]
            train_loader_run = [next(iter(self.train_loader))] * 2 if self.dry_run else self.train_loader
            val_loader_run = [next(iter(self.val_loader))] * 2 if self.dry_run else self.val_loader
            
            # Training loop
            history = {
                "train_loss": [], "train_acc": [],
                "val_loss": [], "val_acc": [], "val_qwk": [], "val_f1": []
            }
            
            best_qwk = -1.0
            best_epoch = 0
            training_start_time = time.time()
            epoch_durations = []
            
            scaler = torch.cuda.amp.GradScaler() if config.USE_AMP and config.DEVICE == "cuda" else None
            
            # Reset peak memory before training start
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                
            self.logger.info(f"Starting training run on device '{config.DEVICE}'...")
            
            for epoch in range(1, epochs_limit + 1):
                epoch_start = time.time()
                
                train_loss, train_acc = train_epoch(
                    model=model,
                    loader=train_loader_run,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=config.DEVICE,
                    use_amp=config.USE_AMP,
                    scaler=scaler
                )
                
                val_loss, val_metrics = validate_epoch(
                    model=model,
                    loader=val_loader_run,
                    criterion=criterion,
                    device=config.DEVICE
                )
                
                scheduler.step()
                
                epoch_duration = time.time() - epoch_start
                epoch_durations.append(epoch_duration)
                
                val_acc = val_metrics["accuracy"]
                val_qwk = val_metrics["qwk"]
                val_f1 = val_metrics["f1"]
                
                self.logger.info(
                    f"Epoch [{epoch:02d}/{epochs_limit:02d}] - "
                    f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                    f"Val QWK: {val_qwk:.4f} | Duration: {epoch_duration:.2f}s"
                )
                
                # Log TensorBoard
                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Val", val_loss, epoch)
                writer.add_scalar("Accuracy/Train", train_acc, epoch)
                writer.add_scalar("Accuracy/Val", val_acc, epoch)
                writer.add_scalar("Metrics/QWK", val_qwk, epoch)
                writer.add_scalar("Metrics/F1", val_f1, epoch)
                
                # Update history
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["val_qwk"].append(val_qwk)
                history["history_f1" if "history_f1" in history else "val_f1"].append(val_f1)
                
                # Save best checkpoint
                if val_qwk > best_qwk:
                    best_qwk = val_qwk
                    best_epoch = epoch
                    checkpoint_state = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_qwk": best_qwk,
                        "seed": self.bench_config["seed"]
                    }
                    torch.save(checkpoint_state, ckpt_dir / "best_model.pt")
                    self.logger.info(f"[NEW BEST] Saved best checkpoint (Val QWK: {best_qwk:.4f})")
                    
                # Early Stopping Check
                if early_stopping(val_qwk):
                    self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break
                    
            total_training_time = time.time() - training_start_time
            avg_epoch_time = float(np.mean(epoch_durations))
            writer.close()
            
            # Export history
            pd.DataFrame(history).to_csv(exp_dir / "history.csv", index_label="epoch")
            with open(exp_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)
                
            # Log peak training VRAM
            peak_vram = 0.0
            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB
            self.logger.info(f"Peak VRAM during training: {peak_vram:.2f} MB")
            
            # Load Best Model for Testing & Inference Latency Profiling
            self.logger.info("Loading best model weights for final evaluation...")
            best_ckpt = ckpt_dir / "best_model.pt"
            if not best_ckpt.exists():
                self.logger.warning("No best checkpoint found! Saving current weights as best...")
                torch.save({"model_state_dict": model.state_dict()}, best_ckpt)
                
            load_checkpoint(checkpoint_path=best_ckpt, model=model, device=config.DEVICE)
            model.eval()
            
            # Inference Latency Benchmark
            self.logger.info("Benchmarking inference latency and throughput...")
            avg_latency_single, avg_latency_batch, throughput_fps = self._profile_latency(model)
            self.logger.info(
                f"Latency: Single: {avg_latency_single:.2f} ms | "
                f"Batch-100: {avg_latency_batch:.2f} ms | "
                f"Throughput: {throughput_fps:.2f} img/sec"
            )
            
            # Evaluate on Test Set
            self.logger.info("Evaluating best model on test set...")
            test_loader_run = [next(iter(self.test_loader))] * 2 if self.dry_run else self.test_loader
            
            test_targets = []
            test_preds = []
            test_probs = []
            
            with torch.no_grad():
                for inputs, targets in test_loader_run:
                    inputs = inputs.to(config.DEVICE)
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    
                    test_targets.extend(targets.numpy())
                    test_preds.extend(preds.cpu().numpy())
                    test_probs.extend(probs.cpu().numpy())
                    
            test_targets = np.array(test_targets)
            test_preds = np.array(test_preds)
            test_probs = np.array(test_probs)
            
            # Compute classification metrics
            metrics = calculate_metrics(y_true=test_targets, y_pred=test_preds, y_probs=test_probs)
            
            # Save Predictions CSV
            pred_df = pd.DataFrame({
                "true_label": test_targets,
                "predicted_label": test_preds,
            })
            for c in range(config.NUM_CLASSES):
                pred_df[f"prob_class_{c}"] = test_probs[:, c]
            pred_df.to_csv(exp_dir / "predictions.csv", index=False)
            
            # Plot model-specific confusion matrix
            plot_confusion_matrix(
                cm=metrics["confusion_matrix"],
                class_names=config.CLASS_NAMES,
                save_path=exp_dir / "confusion_matrix.png"
            )
            
            # Keep ROC and PR data for plotting aggregated curves
            roc_data[arch] = (test_targets, test_probs)
            pr_data[arch] = (test_targets, test_probs)
            
            # Plot model-specific ROC
            self._plot_individual_roc(test_targets, test_probs, exp_dir / "roc.png")
            
            # Save model performance json
            exp_metrics = {
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "qwk": metrics["qwk"],
                "auc": metrics.get("auc", 0.5),
                "peak_vram_mb": peak_vram,
                "avg_epoch_time_sec": avg_epoch_time,
                "total_training_time_sec": total_training_time,
                "single_latency_ms": avg_latency_single,
                "batch_latency_ms": avg_latency_batch,
                "throughput_fps": throughput_fps,
                "total_parameters": complexity["params_count"],
                "flops": complexity["flops"],
                "macs": complexity["macs"]
            }
            with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(exp_metrics, f, indent=4)
                
            # Save results summary row
            arch_row = {
                "Model": arch,
                "Accuracy": metrics["accuracy"],
                "Balanced Accuracy": metrics["balanced_accuracy"],
                "Macro F1": metrics["f1"],
                "QWK": metrics["qwk"],
                "ROC AUC": metrics.get("auc", 0.5),
                "Params": complexity["params_count"],
                "FLOPs": complexity["flops"],
                "MACs": complexity["macs"],
                "Peak VRAM (MB)": peak_vram,
                "Single Latency (ms)": avg_latency_single,
                "Batch Latency (ms)": avg_latency_batch,
                "Throughput (FPS)": throughput_fps,
                "Avg Epoch Time (s)": avg_epoch_time,
                "Total Train Time (s)": total_training_time
            }
            results_list.append(arch_row)
            
            # Clean up model
            self.logger.info(f"Finished benchmarking {arch}. Cleaning up memory...")
            del model
            self._free_memory()
            
        # Post-benchmarking calculations
        df_results = pd.DataFrame(results_list)
        df_results.to_csv(config.BENCHMARK_RESULTS_DIR / "benchmark_results.csv", index=False)
        df_results.to_excel(config.BENCHMARK_RESULTS_DIR / "benchmark_results.xlsx", index=False)
        
        # Print results table
        self.logger.info("\n==================================================")
        self.logger.info("BENCHMARKING COMPLETED! SUMMARY TABLE:")
        self.logger.info("==================================================")
        self.logger.info(df_results.to_string(index=False))
        
        # Metric-by-metric ranking
        self.logger.info("\nComputing metric rankings...")
        self._compute_rankings(df_results)
        
        # Plot aggregated curves & comparisons
        self.logger.info("Generating comparative visualization plots...")
        self._generate_visualizations(df_results, roc_data, pr_data)
        
        # Save reproduction manifest
        self.logger.info("Writing reproduction manifest.json...")
        self._save_manifest()
        
        self.logger.info("\nAll benchmark outputs saved to results/benchmark/ and experiments/benchmark/")
        
    def _profile_latency(self, model):
        """Measures CPU/GPU latency for single inference and batched inference."""
        # 1. Warmup the GPU for 20 runs
        dummy_input_single = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)
        
        with torch.no_grad():
            for _ in range(20):
                _ = model(dummy_input_single)
            if config.DEVICE == "cuda":
                torch.cuda.synchronize()
                
            # 2. Timing Single Latency (100 runs)
            runs = 5 if self.dry_run else 100
            single_times = []
            for _ in range(runs):
                start = time.time()
                _ = model(dummy_input_single)
                if config.DEVICE == "cuda":
                    torch.cuda.synchronize()
                single_times.append((time.time() - start) * 1000) # ms
                
            avg_latency_single = float(np.mean(single_times))
            
            # 3. Timing Batch Latency (100 runs, Batch Size = 100)
            dummy_input_batch = torch.randn(100, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)
            batch_times = []
            for _ in range(runs):
                start = time.time()
                _ = model(dummy_input_batch)
                if config.DEVICE == "cuda":
                    torch.cuda.synchronize()
                batch_times.append((time.time() - start) * 1000) # ms
                
            avg_latency_batch = float(np.mean(batch_times))
            
            # 4. Compute throughput (images per second) using the single latencies average
            # Throughput FPS = 1000 / avg_latency_single
            throughput_fps = 1000.0 / avg_latency_single if avg_latency_single > 0 else 0.0
            
        return avg_latency_single, avg_latency_batch, throughput_fps
        
    def _free_memory(self):
        """Resets cache and gathers garbage collector items."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
    def _plot_individual_roc(self, targets, probs, save_path):
        """Plots and saves the ROC curve for a single model (multiclass OvR)."""
        plt.figure(figsize=(7, 5))
        for c in range(config.NUM_CLASSES):
            y_true_binary = (targets == c).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, probs[:, c])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {config.CLASS_NAMES[c]} (AUC = {roc_auc:.4f})")
            
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("One-vs-Rest ROC Curves")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    def _compute_rankings(self, df_results):
        """Computes and saves separate rankings for key metrics."""
        rankings = {}
        
        # Sort values: Higher is better
        rankings["Accuracy"] = df_results.sort_values(by="Accuracy", ascending=False)["Model"].tolist()
        rankings["QWK"] = df_results.sort_values(by="QWK", ascending=False)["Model"].tolist()
        rankings["Macro F1"] = df_results.sort_values(by="Macro F1", ascending=False)["Model"].tolist()
        rankings["ROC AUC"] = df_results.sort_values(by="ROC AUC", ascending=False)["Model"].tolist()
        
        # Sort values: Lower is better
        rankings["Single Latency (Speed)"] = df_results.sort_values(by="Single Latency (ms)", ascending=True)["Model"].tolist()
        rankings["Batch Latency (Speed)"] = df_results.sort_values(by="Batch Latency (ms)", ascending=True)["Model"].tolist()
        rankings["Throughput (FPS)"] = df_results.sort_values(by="Throughput (FPS)", ascending=False)["Model"].tolist()
        rankings["Parameters (Smaller)"] = df_results.sort_values(by="Params", ascending=True)["Model"].tolist()
        rankings["FLOPs (Efficiency)"] = df_results.sort_values(by="FLOPs", ascending=True)["Model"].tolist()
        
        if torch.cuda.is_available():
            rankings["Peak GPU Memory (Smaller)"] = df_results.sort_values(by="Peak VRAM (MB)", ascending=True)["Model"].tolist()
            
        df_rankings = pd.DataFrame(rankings)
        df_rankings.index = range(1, len(df_results) + 1)
        df_rankings.index.name = "Rank"
        
        df_rankings.to_csv(config.BENCHMARK_RESULTS_DIR / "model_ranking.csv")
        self.logger.info("\nRankings per metric:")
        self.logger.info(df_rankings.to_string())
        
    def _generate_visualizations(self, df_results, roc_data, pr_data):
        """Generates performance plots and saves them under results/benchmark/figures/."""
        # Set seaborn style
        sns.set_theme(style="whitegrid")
        palette = "viridis"
        
        # 1. Bar Chart: Accuracy vs QWK vs Macro F1 comparison
        df_melted = df_results.melt(id_vars="Model", value_vars=["Accuracy", "Macro F1", "QWK"], var_name="Metric", value_name="Score")
        plt.figure(figsize=(9, 5))
        sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette=palette)
        plt.title("Classification Performance Metrics Comparison", fontsize=13, fontweight="bold", pad=15)
        plt.ylim(0, 1.05)
        plt.xlabel("Model Architecture")
        plt.ylabel("Score")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(config.BENCHMARK_FIGURES_DIR / "performance_comparison.png", dpi=300)
        plt.close()
        
        # 2. Bar Chart: Parameter Count Comparison (in Millions)
        plt.figure(figsize=(8, 5))
        params_m = df_results["Params"] / 1e6
        sns.barplot(x=df_results["Model"], y=params_m, palette=palette, hue=df_results["Model"], legend=False)
        plt.title("Model Parameter Size (Millions)", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Model Architecture")
        plt.ylabel("Parameters (M)")
        plt.tight_layout()
        plt.savefig(config.BENCHMARK_FIGURES_DIR / "parameter_comparison.png", dpi=300)
        plt.close()
        
        # 3. Bar Chart: Throughput (FPS) Comparison
        plt.figure(figsize=(8, 5))
        sns.barplot(x=df_results["Model"], y=df_results["Throughput (FPS)"], palette=palette, hue=df_results["Model"], legend=False)
        plt.title("Inference Throughput (Images/Second)", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Model Architecture")
        plt.ylabel("Throughput (FPS)")
        plt.tight_layout()
        plt.savefig(config.BENCHMARK_FIGURES_DIR / "throughput_comparison.png", dpi=300)
        plt.close()
        
        # 4. Bar Chart: Latency (Single vs Batch-100) Comparison
        df_lat = df_results.melt(id_vars="Model", value_vars=["Single Latency (ms)", "Batch Latency (ms)"], var_name="Type", value_name="Latency")
        plt.figure(figsize=(9, 5))
        sns.barplot(data=df_lat, x="Model", y="Latency", hue="Type", palette="muted")
        plt.title("Inference Latency Profile (Lower is Better)", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Model Architecture")
        plt.ylabel("Latency (ms)")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(config.BENCHMARK_FIGURES_DIR / "latency_comparison.png", dpi=300)
        plt.close()
        
        # 5. Bar Chart: Peak VRAM Usage (if CUDA was active)
        if torch.cuda.is_available():
            plt.figure(figsize=(8, 5))
            sns.barplot(x=df_results["Model"], y=df_results["Peak VRAM (MB)"], palette=palette, hue=df_results["Model"], legend=False)
            plt.title("Peak GPU Memory Footprint during Training", fontsize=13, fontweight="bold", pad=15)
            plt.xlabel("Model Architecture")
            plt.ylabel("VRAM (MB)")
            plt.tight_layout()
            plt.savefig(config.BENCHMARK_FIGURES_DIR / "vram_comparison.png", dpi=300)
            plt.close()
            
        # 6. Aggregated ROC Curve Comparison (Macro averaged curves per model)
        plt.figure(figsize=(9, 7))
        for arch, (targets, probs) in roc_data.items():
            # Calculate macro ROC Curve
            fpr_grid = np.linspace(0.0, 1.0, 100)
            tpr_list = []
            
            for c in range(config.NUM_CLASSES):
                y_true_binary = (targets == c).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, probs[:, c])
                tpr_list.append(np.interp(fpr_grid, fpr, tpr))
                
            macro_tpr = np.mean(tpr_list, axis=0)
            macro_auc = auc(fpr_grid, macro_tpr)
            plt.plot(fpr_grid, macro_tpr, lw=2, label=f"{arch} (AUC = {macro_auc:.4f})")
            
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Aggregated One-vs-Rest ROC Curve Comparison", fontsize=13, fontweight="bold", pad=15)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(config.BENCHMARK_FIGURES_DIR / "aggregated_roc.png", dpi=300)
        plt.close()
        
        # 7. Aggregated Precision-Recall Curve Comparison (Macro averaged)
        plt.figure(figsize=(9, 7))
        for arch, (targets, probs) in pr_data.items():
            recall_grid = np.linspace(0.0, 1.0, 100)
            precision_list = []
            
            for c in range(config.NUM_CLASSES):
                y_true_binary = (targets == c).astype(int)
                prec, rec, _ = precision_recall_curve(y_true_binary, probs[:, c])
                # Interpolate to common recall grid (reversing so it aligns correctly)
                precision_list.append(np.interp(recall_grid, rec[::-1], prec[::-1]))
                
            macro_prec = np.mean(precision_list, axis=0)
            plt.plot(recall_grid, macro_prec, lw=2, label=f"{arch}")
            
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Aggregated Precision-Recall Curve Comparison", fontsize=13, fontweight="bold", pad=15)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(config.BENCHMARK_FIGURES_DIR / "aggregated_pr.png", dpi=300)
        plt.close()

    def _save_manifest(self):
        """Generates results/benchmark/manifest.json containing reproduction metadata."""
        import torchvision
        gpu_model = "N/A"
        if torch.cuda.is_available():
            gpu_model = torch.cuda.get_device_name(0)
            
        manifest = {
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "gpu_model": gpu_model,
            "dataset_fingerprint": {
                "num_train_samples": len(self.train_loader.dataset),
                "num_val_samples": len(self.val_loader.dataset),
                "num_test_samples": len(self.test_loader.dataset)
            },
            "seed": self.bench_config["seed"],
            "benchmark_config": self.bench_config,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(config.BENCHMARK_RESULTS_DIR / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FusionMedAI Step 5 Model Benchmarking Run"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute in fast debugging mode"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "efficientnet_b0",
            "efficientnet_b3",
            "convnext_tiny",
            "swin_tiny",
            "vit_b16"
        ],
        help="Benchmark only a single architecture"
    )
    args = parser.parse_args()
    
    runner = BenchmarkRunner(
        dry_run=args.dry_run,
        model_name=args.model
    )
    runner.run()
