import sys
import time
import argparse
from pathlib import Path
from typing import Union, Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import src.config as config
from src.models.model_factory import load_model
from src.training.checkpoint import load_checkpoint
from src.data.transforms import get_val_transforms

class InferenceEngine:
    """
    Inference Engine providing high-performance predictions for single ocular fundus scans
    or batch-wise data loading. Exposes both programmatic Python APIs and a CLI wrapper.
    """
    
    def __init__(self, checkpoint_path: Union[str, Path], device: Optional[str] = None) -> None:
        """
        Args:
            checkpoint_path: Path to the .pt checkpoint containing trained weights.
            device: Target device. If None, resolves from configuration.
        """
        self.device = device if device is not None else config.DEVICE
        self.checkpoint_path = Path(checkpoint_path)
        
        # 1. Load model architecture
        self.model = load_model(
            name=config.MODEL_NAME,
            num_classes=config.NUM_CLASSES,
            pretrained=False
        )
        
        # 2. Restore best weights
        load_checkpoint(
            checkpoint_path=self.checkpoint_path,
            model=self.model,
            device=self.device
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 3. Setup transforms
        self.transform = get_val_transforms()
        
    def predict(self, image: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Performs a single-image prediction.
        
        Args:
            image: Path to the image file, or an opened PIL Image.
            
        Returns:
            Dict[str, Any]: Prediction outputs containing:
              - predicted_class: int (0 to 4)
              - predicted_label: str (class name)
              - confidence: float (highest probability)
              - probabilities: list of floats (probabilities for each class)
              - latency_ms: float (inference time in milliseconds)
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at '{img_path}'")
            with Image.open(img_path) as img:
                pil_image = img.convert("RGB")
        else:
            pil_image = image.convert("RGB")
            
        # Apply transformation
        input_tensor = self.transform(pil_image).unsqueeze(0) # add batch dim
        input_tensor = input_tensor.to(self.device)
        
        # Inference step
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
        latency = (time.time() - start_time) * 1000 # in ms
        pred_class = int(np.argmax(probs))
        
        return {
            "predicted_class": pred_class,
            "predicted_label": config.CLASS_NAMES[pred_class],
            "confidence": float(probs[pred_class]),
            "probabilities": [float(p) for p in probs],
            "latency_ms": latency
        }
        
    def predict_batch(self, loader: DataLoader) -> pd.DataFrame:
        """
        Performs predictions over an entire PyTorch DataLoader.
        
        Args:
            loader: PyTorch DataLoader yielding (inputs, labels) or (inputs,).
            
        Returns:
            pd.DataFrame: DataFrame containing predictions, true labels (if available),
                           confidence scores, and class probabilities.
        """
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                # Handle loaders returning either (inputs, labels) or just (inputs,)
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                    if len(batch) > 1:
                        labels = batch[1]
                        all_labels.extend(labels.numpy())
                else:
                    inputs = batch
                    
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                all_preds.extend(preds)
                all_probs.extend(probs)
                
        all_probs = np.array(all_probs)
        
        result_dict = {
            "prediction": all_preds,
            "confidence": [float(all_probs[i, pred]) for i, pred in enumerate(all_preds)]
        }
        
        if len(all_labels) > 0:
            result_dict["label"] = all_labels
            
        # Append class-wise probabilities
        for c in range(config.NUM_CLASSES):
            result_dict[f"prob_class_{c}"] = all_probs[:, c]
            
        return pd.DataFrame(result_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FusionMedAI Inference Engine")
    parser.add_argument("--image", type=str, required=True, help="Path to fundus image scan.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained checkpoint (.pt).")
    args = parser.parse_args()
    
    try:
        engine = InferenceEngine(checkpoint_path=args.model)
        result = engine.predict(image=args.image)
        
        print("\n=== Prediction Summary ===")
        print(f"File: {args.image}")
        print(f"Diagnosis Class: {result['predicted_class']} ({result['predicted_label']})")
        print(f"Confidence Score: {result['confidence']:.2%}")
        print(f"Inference Latency: {result['latency_ms']:.2f} ms")
        print("Class Probabilities:")
        for idx, name in enumerate(config.CLASS_NAMES):
            print(f"  - {name}: {result['probabilities'][idx]:.4f}")
    except Exception as e:
        print(f"Inference execution failed: {e}")
