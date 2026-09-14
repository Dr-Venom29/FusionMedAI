import os
import sys
import argparse
import json
from pathlib import Path
import torch

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.training.config import BaselineConfig
from src.foot.data.dataloader import create_foot_dataloaders
from src.foot.models.factory import create_model
from src.foot.training.losses import get_loss_function
from src.foot.training.trainer import FootBaselineTrainer, set_reproducibility

def main():
    parser = argparse.ArgumentParser(description="Run Foot DFU Wagner 4-Class Baseline Model Training (Phase 10.4)")
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"], help="Baseline architecture (default: resnet50)")
    parser.add_argument("--loss", type=str, default="unweighted", choices=["unweighted", "weighted"], help="Loss function type (default: unweighted)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    
    # Initialize Configuration
    config = BaselineConfig(
        model_name=args.arch,
        loss_type=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed
    )
    
    set_reproducibility(config.seed)
    
    print("==================================================")
    print(f"Phase 10.4 — Foot DFU Baseline Training [{config.model_name.upper()}]")
    print("==================================================")
    
    # 1. Create DataLoaders
    train_csv = config.splits_dir / "train.csv"
    val_csv = config.splits_dir / "val.csv"
    test_csv = config.splits_dir / "test.csv"
    
    print(f"Creating DataLoaders (Batch Size: {config.batch_size}, Workers: {config.num_workers}, Seed: {config.seed})...", flush=True)
    train_loader, val_loader, test_loader = create_foot_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed
    )
    print(" DataLoaders created successfully.", flush=True)
    
    # 2. Build Baseline Model
    print(f"Building {config.model_name.upper()} Baseline Model on device: {config.device}...", flush=True)
    model = create_model(
        name=config.model_name,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        dropout_rate=config.dropout_rate,
        device=config.device
    )
    print(" Baseline Model built successfully.", flush=True)
    
    # 3. Setup Loss Function
    criterion = get_loss_function(
        loss_type=config.loss_type,
        class_weights=config.class_weights,
        device=config.device
    )
    
    # 4. Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    
    # 5. Execute Training
    trainer = FootBaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config=config
    )
    
    training_summary = trainer.train()
    
    # 6. Final Test Evaluation (ONE Evaluation after Configuration Freeze)
    print("\n==================================================")
    print("Evaluating Best Model Checkpoint on Frozen Test Partition...")
    print("==================================================")
    
    trainer.checkpoint_manager.load(model=model, device=config.device)
    test_loss, test_metrics = trainer.evaluate(test_loader)
    
    print("\n--- Frozen Test Set Evaluation Results ---")
    print(f" Test Loss:          {test_loss:.4f}")
    print(f" Macro F1-Score:     {test_metrics['macro_f1']:.4f} (PRIMARY METRIC)")
    print(f" Top-1 Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f" Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}")
    print(f" Weighted F1-Score:  {test_metrics['weighted_f1']:.4f}")
    if test_metrics.get("macro_roc_auc") is not None:
        print(f" Macro ROC-AUC:      {test_metrics['macro_roc_auc']:.4f}")
        
    print("\nClass-Wise Metrics:")
    for c_name, c_m in test_metrics["class_metrics"].items():
        print(f"  {c_name:10s} | Precision: {c_m['precision']:.4f} | Recall: {c_m['recall']:.4f} | F1: {c_m['f1_score']:.4f} | Support: {c_m['support']}")
        
    print("\n4x4 Confusion Matrix (Rows=True, Cols=Pred):")
    for row in test_metrics["confusion_matrix"]:
        print(f"  {row}")
        
    # Save Test evaluation JSON
    test_report = {
        "model_name": config.model_name,
        "loss_type": config.loss_type,
        "best_val_loss": training_summary["best_val_loss"],
        "best_val_macro_f1": training_summary["best_val_macro_f1"],
        "test_loss": round(test_loss, 4),
        "test_metrics": test_metrics
    }
    
    test_json_path = config.experiment_dir / "test_evaluation.json"
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)
    print(f"\nSaved Test evaluation JSON to: {test_json_path}")
    
    # 7. Quality-Stratified Error Analysis
    print("\nPerforming Error Analysis...")
    trainer.perform_error_analysis(test_loader)
    
    print("\nPhase 10.4 Baseline Training Experiment Completed Successfully!")

if __name__ == "__main__":
    main()
