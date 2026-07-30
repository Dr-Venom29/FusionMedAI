import torch
import pandas as pd
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

from src import config
from src.data.dataloader import create_dataloaders
from src.models.model_factory import load_model
from src.training.checkpoint import load_checkpoint
from src.xai.utils import compute_entropy, compute_margin
from src.utils.logger import setup_logger

logger = setup_logger("XAI_Inference", config.XAI_LOGS_DIR / "xai_inference.log")

def run_xai_inference(checkpoint_path, model_name, batch_size=32, device="cuda"):
    """
    Run inference on the test set and gather predictions, probabilities,
    confidence, entropy, margin, and execution time.
    """
    logger.info("Initializing DataLoaders for XAI inference...")
    
    # We only need test_loader, but create_dataloaders returns train, val, test
    _, _, test_loader = create_dataloaders(
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = load_model(name=model_name, num_classes=config.NUM_CLASSES, pretrained=False)
    load_checkpoint(checkpoint_path=Path(checkpoint_path), model=model, device=device)
    
    model = model.to(device)
    model.eval()
    
    results = []
    
    # Map index to image_id for the test dataset
    # We need to access the underlying dataset's dataframe
    test_df = test_loader.dataset.dataframe
    
    logger.info("Starting inference on test set...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Inference")):
            inputs = inputs.to(device)
            
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            outputs = model(inputs)
            
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            targets = targets.numpy()
            
            batch_time_ms = ((end_time - start_time) / len(inputs)) * 1000
            
            entropy = compute_entropy(probs)
            margin = compute_margin(probs)
            confidence = np.max(probs, axis=1)
            
            start_idx = batch_idx * batch_size
            
            for i in range(len(inputs)):
                global_idx = start_idx + i
                if global_idx < len(test_df):
                    image_id = test_df.iloc[global_idx][config.ID_COLUMN]
                else:
                    image_id = f"unknown_{global_idx}"
                
                row = {
                    "image_id": image_id,
                    "ground_truth": config.CLASS_NAMES[targets[i]],
                    "prediction": config.CLASS_NAMES[preds[i]],
                    "ground_truth_idx": targets[i],
                    "prediction_idx": preds[i],
                    "confidence": confidence[i],
                    "entropy": entropy[i],
                    "margin": margin[i],
                    "execution_time_ms": batch_time_ms,
                    "is_correct": bool(targets[i] == preds[i])
                }
                for c in range(config.NUM_CLASSES):
                    row[f"prob_{c}"] = probs[i, c]
                    
                results.append(row)
                
    results_df = pd.DataFrame(results)
    logger.info(f"Inference completed. Processed {len(results_df)} images.")
    return results_df, model, test_loader
