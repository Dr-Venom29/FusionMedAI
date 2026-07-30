import os
import sys
import json
import time
import argparse
import hashlib
from pathlib import Path

import torch
import numpy as np
import cv2
from tqdm import tqdm

from src import config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.xai.inference import run_xai_inference
from src.xai.selector import select_representative_cases
from src.xai.cam import BaseCAM
from src.xai.gradcam import GradCAM
from src.xai.gradcampp import GradCAMPlusPlus
from src.xai.clinical_interpreter import interpret_cam_clinical
from src.xai.visualization import generate_xai_panel
from src.xai.report_generator import generate_all_reports
from src.xai.utils import get_target_layer, get_git_commit_hash
import pandas as pd

def get_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="FusionMedAI Step 6: XAI Pipeline")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--model", type=str, default="efficientnet_b3", help="Model architecture name")
    parser.add_argument("--method", type=str, default="gradcam", choices=["gradcam", "gradcam++"])
    parser.add_argument("--device", type=str, default=config.DEVICE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--num-images", type=int, default=25)
    parser.add_argument("--save-csv", action="store_true", default=True)
    parser.add_argument("--save-raw-cam", action="store_true", default=True)
    parser.add_argument("--save-overlays", action="store_true", default=True)
    parser.add_argument("--generate-pdf", action="store_true", default=True)
    parser.add_argument("--generate-gallery", action="store_true", default=True)
    parser.add_argument("--generate-failure-report", action="store_true", default=True)
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger("XAI_Pipeline", config.XAI_LOGS_DIR / "xai_pipeline.log")
    logger.info("Starting FusionMedAI XAI Pipeline...")

    set_seed(args.seed)

    # 1. Inference
    logger.info("Running full test set inference...")
    results_df, model, test_loader = run_xai_inference(args.checkpoint, args.model, args.batch_size, args.device)
    
    if args.save_csv:
        results_df.to_csv(config.FINAL_MODEL_RESULTS_DIR / "predictions.csv", index=False)
        prob_cols = [c for c in results_df.columns if c.startswith('prob_')]
        results_df[['image_id', 'ground_truth', 'prediction'] + prob_cols].to_csv(
            config.FINAL_MODEL_RESULTS_DIR / "probabilities.csv", index=False)

    # 2. Representative Selection
    logger.info(f"Selecting {args.num_images} representative cases...")
    selected_df = select_representative_cases(results_df, args.num_images)
    if args.save_csv:
        selected_df.to_csv(config.XAI_RESULTS_DIR / "selected_cases.csv", index=False)

    # 3. CAM Generation Setup
    target_layer = get_target_layer(model, args.model)

    if args.method == "gradcam++":
        cam_extractor = GradCAMPlusPlus(model, target_layer)
    else:
        cam_extractor = GradCAM(model, target_layer)

    # 4. Generate Visualizations
    logger.info("Generating CAMs and Visualizations...")
    
    # We need a way to get the original un-normalized image.
    # The dataloader applies normalization. We can invert it.
    mean = np.array(config.NORMALIZATION_MEAN)
    std = np.array(config.NORMALIZATION_STD)
    
    cam_intensities = []
    summary_data = []
    
    for idx, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="CAM Gen"):
        image_id = row['image_id']
        
        # Find the image in the test set
        # This is a bit inefficient but fine for ~30 images
        dataset = test_loader.dataset
        img_idx = dataset.dataframe.index[dataset.dataframe[config.ID_COLUMN] == image_id].tolist()
        if not img_idx:
            continue
        img_idx = img_idx[0]
        
        input_tensor, _ = dataset[img_idx]
        input_tensor = input_tensor.unsqueeze(0).to(args.device)
        
        # Get CAM
        raw_cam = cam_extractor.forward(input_tensor, class_idx=row['prediction_idx'])
        norm_cam = BaseCAM.normalize_cam(raw_cam)
        cam_intensities.append(np.mean(norm_cam))
        
        # Recover original image for overlay
        img_np = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        original_img = std * img_np + mean
        original_img = np.clip(original_img, 0, 1)
        
        overlay, heatmap = BaseCAM.overlay_cam(original_img, norm_cam)
        
        # Clinical observation
        observation = interpret_cam_clinical(norm_cam)
        
        # Save artifacts
        out_dir = config.XAI_RESULTS_DIR / f"image_{image_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare Metadata
        meta = row.to_dict()
        meta['cam_method'] = args.method
        meta['observation'] = observation
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=4)
            
        if args.save_raw_cam:
            np.save(out_dir / "gradcam.npy", norm_cam)
            
        if args.save_overlays:
            # Convert float RGB to uint8 RGB
            cv2.imwrite(str(out_dir / "original.png"), cv2.cvtColor((original_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / "heatmap.png"), cv2.cvtColor((heatmap * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / "overlay.png"), cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            
        # Panel Generation
        panel_path = out_dir / "panel.png"
        generate_xai_panel(original_img, heatmap, overlay, meta, observation, save_path=panel_path)
        
        summary_data.append(meta)
        
    cam_extractor.remove_hooks()
    
    if args.save_csv:
        pd.DataFrame(summary_data).to_csv(config.XAI_RESULTS_DIR / "summary.csv", index=False)
    
    # 5. Metrics & Manifest
    logger.info("Computing metrics and manifest...")
    
    metrics = {
        "Total images evaluated": len(results_df),
        "Correct predictions": int(results_df['is_correct'].sum()),
        "Incorrect predictions": int((~results_df['is_correct']).sum()),
        "Average confidence": float(results_df['confidence'].mean()),
        "Average entropy": float(results_df['entropy'].mean()),
        "Average inference time (ms)": float(results_df['execution_time_ms'].mean()),
        "Average CAM intensity": float(np.mean(cam_intensities)) if cam_intensities else 0.0,
        "Representative selection count": len(selected_df),
        "Generation timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(config.XAI_RESULTS_DIR / "xai_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    import torchvision
    manifest = {
        "Dataset version": "APTOS 2019 Blindness Detection",
        "Dataset test split checksum": get_file_hash(config.TEST_SPLIT_CSV) if config.TEST_SPLIT_CSV.exists() else "N/A",
        "Image size": config.IMAGE_SIZE,
        "Normalization": "ImageNet",
        "Checkpoint SHA256": get_file_hash(args.checkpoint),
        "Git commit hash": get_git_commit_hash(),
        "XAI method version": config.XAI_VERSION,
        "CLI Command": " ".join(sys.argv),
        "Selected representative count": len(selected_df),
        "Python version": sys.version,
        "Torch version": torch.__version__,
        "Torchvision version": torchvision.__version__,
        "CUDA version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "OS": os.name
    }
    
    with open(config.XAI_RESULTS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    # 6. Reports
    if args.generate_pdf:
        logger.info("Generating PDF reports...")
        generate_all_reports(metrics, manifest, config.XAI_RESULTS_DIR, config.REPORTS_DIR)
        
    logger.info("XAI Pipeline complete. Outputs saved to results/xai/ and reports/")

if __name__ == "__main__":
    main()
