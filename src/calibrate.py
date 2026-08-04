import argparse
import sys
import torch
import pandas as pd
from pathlib import Path
import hashlib
from torch.utils.tensorboard import SummaryWriter

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    PROJECT_ROOT,
    VAL_SPLIT_CSV,
    CALIBRATION_EXPERIMENTS_DIR,
    CALIBRATION_RESULTS_DIR,
    CALIBRATION_FIGURES_DIR,
    CALIBRATION_TABLES_DIR,
    CALIBRATION_REPORTS_DIR,
    CALIBRATION_RELIABILITY_DIR
)
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.data.dataloader import create_dataloaders
from src.models.model_factory import load_model

from src.calibration.temperature_scaling import calibrate_model
from src.calibration.metrics import compute_all_metrics
from src.calibration.visualization import generate_all_plots
from src.calibration.report_generator import (
    save_summary_json,
    save_summary_csv,
    save_comparison_md,
    save_calibrated_predictions,
    generate_pdf_report
)
from src.calibration.utils import generate_manifest, log_to_tensorboard

def parse_args():
    parser = argparse.ArgumentParser(description="Retinal Model Calibration")
    parser.add_argument("--model", type=str, required=True, help="Model architecture name (e.g. efficientnet_b3)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--bins", type=int, default=15, help="Number of bins for ECE")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for validation loader")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--save-plots", action="store_true", help="Save reliability and confidence plots")
    parser.add_argument("--save-report", action="store_true", help="Generate PDF report")
    return parser.parse_args()

def get_experiment_version(base_dir: Path, prefix: str = "v") -> str:
    """Finds the next available version number (e.g., v001, v002)."""
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not existing_dirs:
        return f"{prefix}001"
    
    versions = []
    for d in existing_dirs:
        try:
            versions.append(int(d.name.split('_')[0].replace(prefix, '')))
        except ValueError:
            pass
            
    if not versions:
        return f"{prefix}001"
        
    next_version = max(versions) + 1
    return f"{prefix}{next_version:03d}"

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 1. Setup paths
    version = get_experiment_version(CALIBRATION_EXPERIMENTS_DIR)
    exp_name = f"{version}_temperature_scaling"
    exp_dir = CALIBRATION_EXPERIMENTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    tb_dir = exp_dir / "tensorboard"
    logs_dir = exp_dir / "logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("Calibration", log_file=logs_dir / "calibration.log")
    logger.info("Starting Calibration Pipeline...")
    logger.info(f"Experiment Directory: {exp_dir}")
    
    writer = SummaryWriter(log_dir=str(tb_dir))
    
    # 2. Load Model
    logger.info(f"Loading model: {args.model}")
    model = load_model(args.model, pretrained=False)
    
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    # Check if checkpoint has state_dict key or is just the state dict itself
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
         model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # 3. Dataloaders
    logger.info("Preparing data loaders...")
    # We only need validation loader for calibration
    _, val_loader, _ = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    val_dataset_df = pd.read_csv(VAL_SPLIT_CSV)
    
    # 4. Calibration
    logger.info("Collecting validation logits and optimizing temperature...")
    scaled_model, calib_state, raw_logits, labels = calibrate_model(
        model=model,
        valid_loader=val_loader,
        output_dir=exp_dir,
        device=args.device
    )
    
    temperature = calib_state["temperature"]
    logger.info(f"Optimization complete. Temperature = {temperature:.4f}")
    
    # 5. Metrics
    logger.info("Evaluating before calibration...")
    metrics_before = compute_all_metrics(raw_logits, labels, num_bins=args.bins)
    log_to_tensorboard(writer, "Before_Calibration", metrics_before, step=0)
    
    logger.info("Evaluating after calibration...")
    # we can use the pre-computed logits and divide by temperature
    calibrated_logits = raw_logits / temperature
    metrics_after = compute_all_metrics(calibrated_logits, labels, num_bins=args.bins)
    log_to_tensorboard(writer, "After_Calibration", metrics_after, step=0)
    writer.add_scalar("Calibration/Temperature", temperature, 0)
    
    # Save Entropies
    import torch.nn.functional as F
    import numpy as np
    probs_before = F.softmax(raw_logits, dim=1)
    probs_after = F.softmax(calibrated_logits, dim=1)
    entropy_before = -(probs_before * torch.log(probs_before + 1e-10)).sum(dim=1)
    entropy_after = -(probs_after * torch.log(probs_after + 1e-10)).sum(dim=1)
    np.save(exp_dir / "entropy_before.npy", entropy_before.numpy())
    np.save(exp_dir / "entropy_after.npy", entropy_after.numpy())
    
    # 6. Save Summaries
    logger.info("Saving metrics and summaries...")
    save_summary_json(metrics_before, metrics_after, exp_dir / "calibration_metrics.json")
    save_summary_csv(metrics_before, metrics_after, exp_dir / "calibration_metrics.csv")
    save_comparison_md(metrics_before, metrics_after, CALIBRATION_TABLES_DIR / "comparison.md")
    save_calibrated_predictions(val_dataset_df, raw_logits, calibrated_logits, labels, temperature, CALIBRATION_RESULTS_DIR / "calibrated_predictions.csv")
    
    # 7. Visualization
    if args.save_plots:
        logger.info("Generating reliability diagrams and plots...")
        generate_all_plots(raw_logits, calibrated_logits, labels, CALIBRATION_FIGURES_DIR, num_bins=args.bins)
        
    # 8. PDF Report
    if args.save_report:
        logger.info("Generating PDF report...")
        generate_pdf_report(
            metrics_before,
            metrics_after,
            temperature,
            CALIBRATION_FIGURES_DIR,
            CALIBRATION_REPORTS_DIR / "calibration_report.pdf"
        )
        
    # 9. Manifest
    logger.info("Generating manifest...")
    
    # Calculate SHA256 of checkpoint
    sha256_hash = hashlib.sha256()
    with open(args.checkpoint, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    checkpoint_sha256 = sha256_hash.hexdigest()
    
    generate_manifest(
        model_name=args.model,
        checkpoint_name=Path(args.checkpoint).name,
        checkpoint_sha256=checkpoint_sha256,
        dataset_version="1.0",
        temperature=temperature,
        num_bins=args.bins,
        num_validation_samples=len(labels),
        seed=args.seed,
        save_path=exp_dir / "manifest.json"
    )
    
    logger.info("Calibration completed successfully.")
    
if __name__ == "__main__":
    main()
