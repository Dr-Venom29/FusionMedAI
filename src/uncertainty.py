import argparse
import sys
import logging
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data.dataloader import create_dataloaders
from src.models.model_factory import load_model
from src.training.checkpoint import load_checkpoint

# Import package modules
from src.uncertainty import (
    load_and_verify_calibration,
    run_deterministic_inference,
    run_stochasticity_validation,
    run_mc_dropout_inference,
    run_convergence_analysis,
    compute_stochastic_metrics,
    select_uncertainty_cases,
    generate_all_uncertainty_plots,
    generate_all_tables_and_reports
)

def parse_args():
    parser = argparse.ArgumentParser(description="Retinal Model Prediction Uncertainty Estimation")
    parser.add_argument("--checkpoint", type=str, default=str(config.BEST_CHECKPOINT), help="Path to best_model.pt checkpoint")
    parser.add_argument("--model", type=str, default="efficientnet_b3", help="Model architecture name")
    parser.add_argument("--mc-passes", type=int, default=25, help="Number of MC passes to evaluate")
    parser.add_argument("--calibration-dir", type=str, default=None, help="Specific calibration directory path to load temperature scaling")
    parser.add_argument("--temperature", type=float, default=None, help="Manual override temperature scaling factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--save-plots", action="store_true", default=True, help="Save uncertainty analysis plots")
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
    
    # 1. Setup paths and logging
    version = get_experiment_version(config.UNCERTAINTY_EXPERIMENTS_DIR)
    exp_name = f"{version}_mc_dropout"
    exp_dir = config.UNCERTAINTY_EXPERIMENTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup loggers
    # Add a root-level logger for standard stdout and file logging
    logger = setup_logger("Uncertainty_Pipeline", log_file=logs_dir / "uncertainty.log")
    
    logger.info("================================================================")
    logger.info("Starting prediction uncertainty estimation pipeline (Step 8)...")
    logger.info(f"Experiment Directory: {exp_dir}")
    logger.info(f"Checkpoint Target: {args.checkpoint}")
    logger.info(f"Model Architecture: {args.model}")
    logger.info(f"MC Passes: {args.mc_passes}")
    logger.info("================================================================")
    
    checkpoint_path = Path(args.checkpoint)
    calibration_dir = Path(args.calibration_dir) if args.calibration_dir is not None else None
    
    # 2. Dynamic Temperature and Hash Verification
    try:
        temperature, verification_info = load_and_verify_calibration(
            checkpoint_path=checkpoint_path,
            calibration_dir=calibration_dir,
            override_temp=args.temperature
        )
        logger.info(f"Calibration scaling verified. Loaded temperature T = {temperature:.4f}")
    except Exception as e:
        logger.error(f"Calibration Verification Failed: {e}")
        sys.exit(1)
        
    # 3. Load Model
    logger.info(f"Loading model: {args.model}")
    try:
        model = load_model(args.model, pretrained=False)
        load_checkpoint(checkpoint_path=checkpoint_path, model=model, device=args.device)
        model = model.to(args.device)
        logger.info("Model loaded and weights restored successfully.")
    except Exception as e:
        logger.error(f"Failed to load model or restore checkpoint: {e}")
        sys.exit(1)
        
    # 4. Prepare Dataloaders (test split)
    logger.info("Preparing data loaders...")
    try:
        # We need the test loader for uncertainty evaluation
        _, _, test_loader = create_dataloaders(
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        logger.info(f"Test split dataloader initialized with {len(test_loader.dataset)} samples.")
    except Exception as e:
        logger.error(f"Failed to initialize dataloaders: {e}")
        sys.exit(1)
        
    # 5. Programmatic Dropout Discovery & Stochasticity Verification (5-pass check)
    try:
        validation_report = run_stochasticity_validation(
            model=model,
            test_loader=test_loader,
            temperature=temperature,
            device=args.device
        )
        logger.info("Stochasticity check successful. The model produces variable outputs under MC Dropout.")
    except Exception as e:
        logger.error(f"Stochasticity Validation Failed: {e}")
        sys.exit(1)
        
    # 6. Run Deterministic Inference Baseline
    det_df = run_deterministic_inference(
        model=model,
        test_loader=test_loader,
        temperature=temperature,
        device=args.device
    )
    logger.info("Deterministic baseline inference completed.")
    
    # 7. Run MC Dropout Inference (N passes)
    # We always run the requested mc-passes
    mc_probs, targets = run_mc_dropout_inference(
        model=model,
        test_loader=test_loader,
        temperature=temperature,
        num_passes=args.mc_passes,
        device=args.device
    )
    logger.info(f"MC Dropout inference completed. Probability tensor shape: {mc_probs.shape}")
    
    # 8. Compute Stochastic Metrics
    stoch_metrics = compute_stochastic_metrics(mc_probs)
    logger.info("Stochastic uncertainty metrics calculated.")
    
    # Combine deterministic and stochastic results into one master DataFrame
    # Note: det_df has results sorted in dataset order.
    # mc_probs and stoch_metrics also align with dataset order.
    df_results = det_df.copy()
    
    # Add stochastic metrics to results df
    df_results["predictive_entropy"] = stoch_metrics["predictive_entropy"]
    df_results["predictive_entropy_norm"] = stoch_metrics["predictive_entropy_norm"]
    df_results["expected_entropy"] = stoch_metrics["expected_entropy"]
    df_results["expected_entropy_norm"] = stoch_metrics["expected_entropy_norm"]
    df_results["predictive_variance"] = stoch_metrics["predictive_variance"]
    df_results["mutual_information"] = stoch_metrics["mutual_information"]
    
    # 9. Run MC Convergence Analysis
    # We do a subset check for N in {5, 10, 25, 50}
    # To run convergence analysis up to N=50, we need to run 50 passes.
    # If the user requested mc-passes >= 50, we can use the main mc_probs.
    # If not, we run a small subset of 50 images for 50 passes to produce convergence curves.
    logger.info("Preparing convergence analysis data...")
    if args.mc_passes >= 50:
        convergence_data = run_convergence_analysis(mc_probs, subset_size=50)
    else:
        # Run 50 passes on a small subset of the test loader (first 50 samples)
        subset_dataset = torch.utils.data.Subset(test_loader.dataset, range(min(50, len(test_loader.dataset))))
        subset_loader = DataLoader(subset_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        mc_probs_50, _ = run_mc_dropout_inference(
            model=model,
            test_loader=subset_loader,
            temperature=temperature,
            num_passes=50,
            device=args.device
        )
        convergence_data = run_convergence_analysis(mc_probs_50, subset_size=50)
        
    # 10. Case Selection (20-30 representative cases)
    selected_cases = select_uncertainty_cases(df_results, num_cases=25)
    
    # Cross-reference with Grad-CAM maps from Step 6 if they exist
    logger.info("Correlating with explainability (Grad-CAM) artifacts...")
    cam_intensities = []
    for _, row in selected_cases.iterrows():
        image_id = row["image_id"]
        # Look for raw gradcam.npy in results/xai/image_{image_id}/gradcam.npy
        cam_path = config.XAI_RESULTS_DIR / f"image_{image_id}" / "gradcam.npy"
        if cam_path.exists():
            try:
                cam_map = np.load(cam_path)
                cam_intensities.append(float(np.mean(cam_map)))
            except Exception:
                cam_intensities.append(np.nan)
        else:
            cam_intensities.append(np.nan)
    selected_cases["gradcam_mean_intensity"] = cam_intensities
    
    # Save selected cases list in the results folder
    selected_cases.to_csv(config.UNCERTAINTY_RESULTS_DIR / "selected_cases.csv", index=False)
    
    # Add selected_reason to the master results dataframe for Figure 10 plotting
    df_results["selected_reason"] = df_results["image_id"].map(selected_cases.set_index("image_id")["selected_reason"])
    
    # Save master predictions CSV in results folder
    df_results.to_csv(config.UNCERTAINTY_RESULTS_DIR / "predictions.csv", index=False)
    
    # Save a lighter uncertainty_metrics.csv containing only image ids, targets, predictions, correctness, and uncertainty scores
    metric_cols = ["image_id", "ground_truth", "ground_truth_label", "prediction", "prediction_label", "is_correct",
                   "calib_confidence", "calib_entropy_norm", "predictive_entropy_norm", "predictive_variance", "mutual_information"]
    df_results[metric_cols].to_csv(config.UNCERTAINTY_RESULTS_DIR / "uncertainty_metrics.csv", index=False)
    
    # 11. Tables & JSON Reporting
    generate_all_tables_and_reports(
        df=df_results,
        mc_probs=mc_probs,
        validation_report=validation_report,
        verification_info=verification_info,
        output_dir=config.UNCERTAINTY_RESULTS_DIR,
        exp_dir=exp_dir
    )
    
    # 12. Visualization Plots
    if args.save_plots:
        generate_all_uncertainty_plots(
            df=df_results,
            convergence_data=convergence_data,
            output_dir=config.UNCERTAINTY_RESULTS_DIR / "figures"
        )
        
    logger.info("================================================================")
    logger.info("Prediction uncertainty estimation pipeline completed successfully!")
    logger.info(f"All artifacts saved in results/uncertainty/ and {exp_dir}")
    logger.info("================================================================")

if __name__ == "__main__":
    main()
