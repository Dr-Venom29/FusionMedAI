import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.training.config import BaselineConfig
from src.foot.data.dataloader import create_foot_dataloaders
from src.foot.models.factory import create_model, MODEL_REGISTRY
from src.foot.training.losses import get_loss_function
from src.foot.training.trainer import FootBaselineTrainer, set_reproducibility

BENCHMARK_MODELS = ["efficientnet_b0", "efficientnet_b3", "convnext_tiny", "swin_tiny", "vit_b16"]

def profile_model(model_name: str, num_classes: int = 4, device: str = "cpu") -> Dict[str, Any]:
    """
    Programmatically profiles model parameters, memory size, and throughput (Phase 10.5.5).
    """
    model = create_model(model_name=model_name, num_classes=num_classes, pretrained=True, device=device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = round(total_params * 4 / (1024 * 1024), 2)
    
    # Warmup and Latency profiling (Batch 32, Image 224x224)
    dummy_input = torch.randn(32, 3, 224, 224).to(device)
    with torch.inference_mode():
        for _ in range(3):
            _ = model(dummy_input)
            
        if torch.cuda.is_available() and "cuda" in str(device):
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()
        for _ in range(5):
            _ = model(dummy_input)
            
        if torch.cuda.is_available() and "cuda" in str(device):
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        
    elapsed = end_time - start_time
    avg_batch_latency_ms = round((elapsed / 5) * 1000, 2)
    throughput_fps = round((32 * 5) / elapsed, 2)
    
    return {
        "model_name": model_name,
        "profiling_device": str(device),
        "total_params": total_params,
        "total_params_m": round(total_params / 1e6, 2),
        "trainable_params": trainable_params,
        "param_size_mb": param_size_mb,
        "avg_batch_latency_ms": avg_batch_latency_ms,
        "throughput_fps": throughput_fps
    }

def run_architecture_benchmark(selected_models: List[str], profile_only: bool = False):
    print("==================================================")
    print("Phase 10.5 — Foot Ulcer Architecture Benchmarking")
    print("==================================================")
    
    base_benchmark_dir = Path(__file__).resolve().parents[3] / "experiments" / "foot" / "architecture_benchmark"
    base_benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    config = BaselineConfig()
    
    # 0. Save Global Benchmark Contract
    bench_cfg = {
        "dataset": "Frozen Phase 10.2 Foot DFU Split",
        "splits_dir": str(config.splits_dir),
        "seed": config.seed,
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "pretrained": True,
        "dropout_rate": 0.2,
        "profiling_device": str(config.device),
        "loss": "unweighted_cross_entropy",
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "epochs": config.epochs,
        "checkpoint_metric": "val_loss",
        "candidate_models": selected_models
    }
    with open(base_benchmark_dir / "benchmark_config.json", "w") as f:
        json.dump(bench_cfg, f, indent=2)
        
    # 1. Profile Models
    print("\n1. Parameter and Computational Profiling (10.5.5)...")
    profile_results = []
    for m_name in selected_models:
        prof = profile_model(m_name, num_classes=4, device=config.device)
        profile_results.append(prof)
        print(f"  {m_name:18s} | Device: {prof['profiling_device']} | Params: {prof['total_params_m']:5.2f}M | Size: {prof['param_size_mb']:6.2f}MB | Latency: {prof['avg_batch_latency_ms']:6.2f}ms")
        
    df_prof = pd.DataFrame(profile_results)
    df_prof.to_csv(base_benchmark_dir / "model_profiling.csv", index=False)
    
    if profile_only:
        print("\nModel profiling complete (--profile_only requested).")
        return
        
    # 2. Setup DataLoaders
    set_reproducibility(config.seed)
    
    print("\n2. Loading Frozen Dataset Splits (Train: 8,038 | Val: 1,006 | Test: 1,006)...")
    train_loader, val_loader, test_loader = create_foot_dataloaders(
        train_csv=config.splits_dir / "train.csv",
        val_csv=config.splits_dir / "val.csv",
        test_csv=config.splits_dir / "test.csv",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed
    )
    
    # 3. Train & Evaluate Candidate Models
    benchmark_summary_list = []
    
    for m_name in selected_models:
        print(f"\n==================================================")
        print(f"Benchmarking Candidate Architecture: {m_name.upper()}")
        print(f"==================================================")
        
        m_exp_dir = base_benchmark_dir / m_name
        m_config = BaselineConfig(
            model_name=m_name,
            loss_type="unweighted",
            dropout_rate=0.2,
            epochs=config.epochs,
            batch_size=config.batch_size,
            seed=config.seed,
            custom_experiment_dir=m_exp_dir
        )
        
        model = create_model(model_name=m_name, num_classes=4, pretrained=True, device=config.device)
        
        # Explicit Output Format and Logits Shape Verification Contract
        dummy_test = torch.randn(2, 3, config.image_size, config.image_size).to(config.device)
        with torch.inference_mode():
            out_test = model(dummy_test)
        if not isinstance(out_test, dict) or "logits" not in out_test:
            raise RuntimeError(f"{m_name}: invalid model output format (must be dict containing 'logits')")
        if out_test["logits"].shape != (2, config.num_classes):
            raise RuntimeError(f"{m_name}: expected logits shape (2, {config.num_classes}), got {out_test['logits'].shape}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
        criterion = get_loss_function("unweighted", device=config.device)
        
        trainer = FootBaselineTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            config=m_config
        )
        
        # Execute Training
        train_summary = trainer.train()
        
        # Evaluate Best Checkpoint on Test Set
        trainer.checkpoint_manager.load(model=model, device=config.device)
        test_loss, test_metrics = trainer.evaluate(test_loader)
        
        # Quality-Stratified Error Analysis
        trainer.perform_error_analysis(test_loader)
        
        # Save individual evaluation report & confusion matrix plot (10.5.14)
        cm_np = np.array(test_metrics["confusion_matrix"])
        meta_bench_dir = base_benchmark_dir / "metadata"
        meta_bench_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(cm_np, cmap="Blues")
            ax.set(xticks=np.arange(4), yticks=np.arange(4), xticklabels=config.class_names, yticklabels=config.class_names,
                   xlabel="Predicted Label", ylabel="True Label", title=f"Foot DFU — {m_name.upper()} Confusion Matrix")
            thresh = cm_np.max() / 2.
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{cm_np[i, j]}", ha="center", va="center", color="white" if cm_np[i, j] > thresh else "black")
            fig.colorbar(im, ax=ax)
            plt.tight_layout()
            
            fig.savefig(m_exp_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
            fig.savefig(meta_bench_dir / f"{m_name}_confusion_matrix.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"  Warning: Could not save confusion matrix plot: {e}")
            
        prof = next(p for p in profile_results if p["model_name"] == m_name)
        
        res = {
            "model_name": m_name,
            "profiling_device": prof["profiling_device"],
            "total_params_m": prof["total_params_m"],
            "param_size_mb": prof["param_size_mb"],
            "avg_batch_latency_ms": prof["avg_batch_latency_ms"],
            "throughput_fps": prof["throughput_fps"],
            "training_time_sec": train_summary["total_training_time_sec"],
            "best_epoch": train_summary["best_epoch"],
            "best_val_loss": train_summary["best_val_loss"],
            "best_val_macro_f1": train_summary["best_val_macro_f1"],
            "test_loss": round(test_loss, 4),
            "test_macro_f1": test_metrics["macro_f1"],
            "macro_f1_ci": test_metrics.get("macro_f1_ci", [test_metrics["macro_f1"], test_metrics["macro_f1"]]),
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "balanced_accuracy_ci": test_metrics.get("balanced_accuracy_ci", [test_metrics["balanced_accuracy"], test_metrics["balanced_accuracy"]]),
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_macro_roc_auc": test_metrics.get("macro_roc_auc", None),
            "grade_1_f1": test_metrics["class_metrics"]["Grade 1"]["f1_score"],
            "grade_2_f1": test_metrics["class_metrics"]["Grade 2"]["f1_score"],
            "grade_3_f1": test_metrics["class_metrics"]["Grade 3"]["f1_score"],
            "grade_4_f1": test_metrics["class_metrics"]["Grade 4"]["f1_score"]
        }
        benchmark_summary_list.append(res)
        
        # Save individual experiment config & validation report (10.5.9)
        cfg_dict = {
            "architecture": m_name,
            "num_classes": 4,
            "seed": config.seed,
            "image_size": config.image_size,
            "batch_size": config.batch_size,
            "pretrained": True,
            "loss": "cross_entropy",
            "checkpoint_metric": "val_loss"
        }
        with open(m_exp_dir / "config.json", "w") as f:
            json.dump(cfg_dict, f, indent=2)

        with open(m_exp_dir / "validation_metrics.json", "w") as f:
            json.dump({"model": m_name, "best_val_loss": train_summary["best_val_loss"], "best_val_macro_f1": train_summary["best_val_macro_f1"]}, f, indent=2)

        with open(m_exp_dir / "test_evaluation.json", "w") as f:
            json.dump({"model": m_name, "test_loss": test_loss, "metrics": test_metrics}, f, indent=2)
            
        # Release GPU memory and prevent state contamination (10.5.8)
        del model, optimizer, scheduler, trainer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # 4. Save Overall Benchmark Artifacts
    df_bm = pd.DataFrame(benchmark_summary_list)
    df_bm.to_csv(base_benchmark_dir / "benchmark_results.csv", index=False)
    
    with open(base_benchmark_dir / "benchmark_results.json", "w") as f:
        json.dump(benchmark_summary_list, f, indent=2)
        
    try:
        excel_path = base_benchmark_dir / "benchmark_results.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_bm.to_excel(writer, sheet_name="Benchmark Summary", index=False)
            
            # Class-wise F1 sheet
            class_rows = [{
                "Model": r["model_name"],
                "Grade 1 F1": r["grade_1_f1"],
                "Grade 2 F1": r["grade_2_f1"],
                "Grade 3 F1": r["grade_3_f1"],
                "Grade 4 F1": r["grade_4_f1"]
            } for r in benchmark_summary_list]
            pd.DataFrame(class_rows).to_excel(writer, sheet_name="Class-wise F1", index=False)
            
            # Computational profile sheet
            prof_rows = [{
                "Model": r["model_name"],
                "Parameters (M)": r["total_params_m"],
                "Model Size (MB)": r["param_size_mb"],
                "Latency (ms)": r["avg_batch_latency_ms"],
                "Throughput (fps)": r["throughput_fps"],
                "Training Time (s)": r["training_time_sec"],
                "Device": r["profiling_device"]
            } for r in benchmark_summary_list]
            pd.DataFrame(prof_rows).to_excel(writer, sheet_name="Computational Profile", index=False)
            
            # CIs sheet
            ci_rows = [{
                "Model": r["model_name"],
                "Test Macro F1": r["test_macro_f1"],
                "Macro F1 95% CI Lower": r["macro_f1_ci"][0],
                "Macro F1 95% CI Upper": r["macro_f1_ci"][1],
                "Balanced Accuracy": r["test_balanced_accuracy"],
                "Bal Acc 95% CI Lower": r["balanced_accuracy_ci"][0],
                "Bal Acc 95% CI Upper": r["balanced_accuracy_ci"][1]
            } for r in benchmark_summary_list]
            pd.DataFrame(ci_rows).to_excel(writer, sheet_name="Confidence Intervals", index=False)
    except Exception as e:
        print(f"  Warning: Could not save Excel workbook: {e}")
        
    # Generate Comprehensive Markdown Summary Report
    md_lines = [
        "# Phase 10.5 — Foot Ulcer Architecture Benchmarking Summary Report",
        "",
        "*Note: Checkpoints were selected using minimum validation loss (`val_loss`). Validation Macro F1 is reported for reference and was not used for model selection.*",
        "",
        "## 1. Candidate Architecture Comparative Benchmark Matrix",
        "",
        "| Model | Params (M) | Size (MB) | Latency (ms) | Throughput (fps) | Test Macro F1 (95% CI) | Balanced Acc. | Macro ROC-AUC | Grade 1 F1 | Grade 2 F1 | Grade 3 F1 | Grade 4 F1 | Train Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in benchmark_summary_list:
        f1_ci_str = f"[{r['macro_f1_ci'][0]:.4f}, {r['macro_f1_ci'][1]:.4f}]"
        roc_auc_str = f"{r['test_macro_roc_auc']:.4f}" if r['test_macro_roc_auc'] is not None else "N/A"
        md_lines.append(
            f"| **{r['model_name']}** | {r['total_params_m']:.2f} | {r['param_size_mb']:.2f} | {r['avg_batch_latency_ms']:.2f} | {r['throughput_fps']:.2f} | "
            f"**`{r['test_macro_f1']:.4f}`** {f1_ci_str} | `{r['test_balanced_accuracy']:.4f}` | `{roc_auc_str}` | "
            f"`{r['grade_1_f1']:.4f}` | `{r['grade_2_f1']:.4f}` | `{r['grade_3_f1']:.4f}` | `{r['grade_4_f1']:.4f}` | `{r['training_time_sec']:.1f}s` |"
        )
        
    with open(base_benchmark_dir / "benchmark_report.md", "w") as f:
        f.write("\n".join(md_lines))
        
    print("\nPhase 10.5 Architecture Benchmarking Completed Successfully!")

def main():
    parser = argparse.ArgumentParser(description="Run Phase 10.5 Foot Ulcer Architecture Benchmarking")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Models to benchmark: 'all' or comma-separated names (e.g. 'efficientnet_b0,convnext_tiny')"
    )
    parser.add_argument("--profile_only", action="store_true", help="Only run model parameter & computational profiling")
    args = parser.parse_args()
    
    if args.models.lower() == "all":
        selected = BENCHMARK_MODELS
    else:
        selected = [m.strip().lower().replace("-", "_") for m in args.models.split(",") if m.strip()]
        
    invalid = [m for m in selected if m not in MODEL_REGISTRY]
    if invalid:
        raise ValueError(
            f"Invalid candidate architecture(s): {invalid}. "
            f"Supported options in MODEL_REGISTRY: {list(MODEL_REGISTRY.keys())}"
        )
        
    run_architecture_benchmark(selected, profile_only=args.profile_only)

if __name__ == "__main__":
    main()
