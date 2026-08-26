import sys
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import src.config as config
from src.data.dataloader import create_dataloaders
from src.models.model_factory import load_model
from src.training.checkpoint import load_checkpoint

logger = logging.getLogger("Uncertainty_Inference")

def compute_entropy_np(probs: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy using base e."""
    eps = 1e-9
    clipped_probs = np.clip(probs, eps, 1.0 - eps)
    return -np.sum(clipped_probs * np.log(clipped_probs), axis=-1)

def compute_margin_np(probs: np.ndarray) -> np.ndarray:
    """Compute difference between top 2 probabilities."""
    sorted_probs = np.sort(probs, axis=-1)[:, ::-1]
    return sorted_probs[:, 0] - sorted_probs[:, 1]

def run_deterministic_inference(
    model: nn.Module,
    test_loader: DataLoader,
    temperature: float,
    device: str = "cuda"
) -> pd.DataFrame:
    """
    Runs deterministic baseline inference (no dropout, weights frozen).
    Computes both raw and calibrated probabilities and uncertainty proxies.
    """
    logger.info("Starting deterministic baseline inference...")
    model.eval()
    
    # Check if dropout is active (it should be inactive under eval())
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            logger.info(f"Verified dropout layer '{name}' training status: {module.training}")
            
    all_image_ids = []
    all_targets = []
    all_raw_logits = []
    
    test_df = test_loader.dataset.dataframe
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Deterministic Inference")):
            inputs = inputs.to(device)
            logits = model(inputs)
            
            all_targets.append(targets.numpy())
            all_raw_logits.append(logits.cpu().numpy())
            
            # Map back to image_ids
            start_idx = batch_idx * test_loader.batch_size
            for i in range(len(inputs)):
                global_idx = start_idx + i
                if global_idx < len(test_df):
                    image_id = test_df.iloc[global_idx][config.ID_COLUMN]
                else:
                    image_id = f"unknown_{global_idx}"
                all_image_ids.append(image_id)
                
    targets_arr = np.concatenate(all_targets)
    raw_logits_arr = np.concatenate(all_raw_logits)
    
    # 1. Compute Raw Probabilities
    # softmax on raw logits
    raw_logits_tensor = torch.tensor(raw_logits_arr)
    raw_probs_arr = F.softmax(raw_logits_tensor, dim=1).numpy()
    
    # 2. Compute Calibrated Probabilities
    # logits scaled by T first, then softmax
    calib_logits_tensor = raw_logits_tensor / temperature
    calib_probs_arr = F.softmax(calib_logits_tensor, dim=1).numpy()
    
    # 3. Predict Classes (highest probability - unaffected by T)
    preds = np.argmax(calib_probs_arr, axis=1)
    
    # 4. Compute Proxies
    raw_conf = np.max(raw_probs_arr, axis=1)
    calib_conf = np.max(calib_probs_arr, axis=1)
    
    raw_entropy = compute_entropy_np(raw_probs_arr)
    calib_entropy = compute_entropy_np(calib_probs_arr)
    
    # Normalized versions
    raw_entropy_norm = raw_entropy / np.log(config.NUM_CLASSES)
    calib_entropy_norm = calib_entropy / np.log(config.NUM_CLASSES)
    
    raw_margin = compute_margin_np(raw_probs_arr)
    calib_margin = compute_margin_np(calib_probs_arr)
    
    results = []
    for idx, img_id in enumerate(all_image_ids):
        target_idx = int(targets_arr[idx])
        pred_idx = int(preds[idx])
        
        row = {
            "image_id": img_id,
            "ground_truth": target_idx,
            "ground_truth_label": config.CLASS_NAMES[target_idx],
            "prediction": pred_idx,
            "prediction_label": config.CLASS_NAMES[pred_idx],
            "is_correct": bool(target_idx == pred_idx),
            "raw_confidence": float(raw_conf[idx]),
            "calib_confidence": float(calib_conf[idx]),
            "raw_entropy": float(raw_entropy[idx]),
            "raw_entropy_norm": float(raw_entropy_norm[idx]),
            "calib_entropy": float(calib_entropy[idx]),
            "calib_entropy_norm": float(calib_entropy_norm[idx]),
            "raw_margin": float(raw_margin[idx]),
            "calib_margin": float(calib_margin[idx])
        }
        
        # Add class-wise raw and calib probabilities
        for c in range(config.NUM_CLASSES):
            row[f"raw_prob_{c}"] = float(raw_probs_arr[idx, c])
            row[f"calib_prob_{c}"] = float(calib_probs_arr[idx, c])
            
        results.append(row)
        
    return pd.DataFrame(results)
