import os
import sys
import time
import json
from pathlib import Path
from typing import Union, Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import src.foot.config as config
from src.foot.models.factory import load_foot_final_model
from src.foot.calibration.vector_scaling import FootVectorScaler
from src.foot.uncertainty.dropout import enable_foot_mc_dropout
from src.foot.xai.gradcam import FootGradCAM
from src.foot.data.transforms import get_foot_val_transforms

class FootModule:
    """
    Unified Foot Ulcer Classification Module (Phase 10.9).
    
    Integrates:
    1. Frozen EfficientNet-B3 Wagner 4-class classifier.
    2. Post-hoc Vector Scaling probability calibration.
    3. Stochastic MC Dropout uncertainty estimation (N*=10 passes, Option B pipeline).
    4. Deterministic Grad-CAM spatial attribution heatmap overlay.
    """
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        calibration_path: Optional[Union[str, Path]] = None,
        uncertainty_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Initializes the integrated Foot Ulcer module with frozen model weights and parameters.
        
        Args:
            checkpoint_path: Path to trained EfficientNet-B3 best_model.pt checkpoint.
            calibration_path: Path to calibration.json vector scaling parameters.
            uncertainty_path: Path to uncertainty.json configuration parameters.
            device: Target execution device ('cpu' or 'cuda'). If None, uses config.DEVICE.
        """
        self.device = torch.device(device) if device is not None else torch.device(config.DEVICE)
        
        # 1. Resolve Frozen Paths
        if checkpoint_path is None:
            checkpoint_path = config.PROJECT_ROOT / "experiments" / "foot" / "architecture_benchmark" / "efficientnet_b3" / "checkpoints" / "best_model.pt"
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Foot model checkpoint not found at '{self.checkpoint_path}'")
            
        if calibration_path is None:
            calibration_path = config.PROJECT_ROOT / "experiments" / "foot" / "final_model" / "calibration.json"
        self.calibration_path = Path(calibration_path)
        if not self.calibration_path.exists():
            raise FileNotFoundError(f"Foot calibration artifact not found at '{self.calibration_path}'")
            
        if uncertainty_path is None:
            uncertainty_path = config.PROJECT_ROOT / "experiments" / "foot" / "final_model" / "uncertainty.json"
        self.uncertainty_path = Path(uncertainty_path)
        if not self.uncertainty_path.exists():
            raise FileNotFoundError(f"Foot uncertainty artifact not found at '{self.uncertainty_path}'")
            
        # 2. Load Frozen Model
        self.model = load_foot_final_model(checkpoint_path=self.checkpoint_path, device=str(self.device))
        self.model.eval()
        
        # 3. Load Frozen Vector Scaling Parameters
        with open(self.calibration_path, "r") as f:
            self.calib_meta = json.load(f)
            
        self.vector_scaler = FootVectorScaler(num_classes=config.NUM_CLASSES)
        weights_arr = torch.tensor(self.calib_meta["weights"], dtype=torch.float32)
        bias_arr = torch.tensor(self.calib_meta["bias"], dtype=torch.float32)
        self.vector_scaler.weights = nn.Parameter(weights_arr)
        self.vector_scaler.bias = nn.Parameter(bias_arr)
        self.vector_scaler.to(self.device)
        self.vector_scaler.eval()
        
        # 4. Load Frozen Uncertainty Config
        with open(self.uncertainty_path, "r") as f:
            self.uncertainty_meta = json.load(f)
        self.default_mc_passes = int(self.uncertainty_meta.get("stochastic_passes_N", 10))
        
        # 5. Preprocessing Transforms
        self.transform = get_foot_val_transforms(image_size=config.IMAGE_SIZE)
        
        # 6. Grad-CAM Explainer
        self.gradcam = FootGradCAM(model=self.model, device=str(self.device))
        
    def _validate_and_load_image(self, image: Union[str, Path, Image.Image]) -> Tuple[Image.Image, np.ndarray]:
        """
        Validates input image source and returns PIL image in RGB mode alongside 224x224 uint8 array.
        """
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Input image file does not exist at '{img_path}'")
            if not img_path.is_file():
                raise ValueError(f"Input image path '{img_path}' is not a file.")
            try:
                pil_img = Image.open(img_path)
                pil_img.verify() # Verify file integrity
                pil_img = Image.open(img_path).convert("RGB") # Re-open after verify
            except Exception as e:
                raise ValueError(f"Failed to decode image file at '{img_path}': {str(e)}")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type '{type(image)}'. Expected str, Path, or PIL.Image.")
            
        # Check image dimensions
        w, h = pil_img.size
        if w < 10 or h < 10:
            raise ValueError(f"Input image resolution ({w}x{h}) is too small for clinical inference.")
            
        # Resize uint8 RGB copy for exact spatial Grad-CAM overlays
        img_resized = pil_img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        orig_rgb_np = np.array(img_resized, dtype=np.uint8)
        
        return pil_img, orig_rgb_np

    def predict(
        self,
        image: Union[str, Path, Image.Image],
        mc_passes: Optional[int] = None,
        generate_cam: bool = True
    ) -> Dict[str, Any]:
        """
        Executes unified diagnostic inference on a single foot ulcer scan.
        
        Pipeline:
        1. Input Validation & RGB Normalization
        2. Deterministic EfficientNet-B3 Forward Pass
        3. Post-hoc Vector Scaling Probability Calibration
        4. Stochastic MC Dropout Uncertainty Estimation (Option B Pipeline)
        5. Grad-CAM Spatial Attribution Map Generation (if requested)
        
        Args:
            image: File path or PIL Image.
            mc_passes: Pass count for uncertainty estimation. If None, uses frozen N*=10.
            generate_cam: Whether to generate Grad-CAM overlay and heatmap arrays.
            
        Returns:
            Dict[str, Any]: Unified diagnostic result dictionary.
        """
        start_time = time.time()
        
        # 1. Input Validation
        pil_img, orig_rgb_np = self._validate_and_load_image(image)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        passes_N = mc_passes if mc_passes is not None else self.default_mc_passes
        
        # 2. Deterministic Inference Pass
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            logits_raw = output["logits"] if isinstance(output, dict) else output
            
            # Uncalibrated probabilities
            raw_probs = F.softmax(logits_raw, dim=1).cpu().numpy()[0]
            raw_pred_class = int(np.argmax(raw_probs))
            raw_conf = float(raw_probs[raw_pred_class])
            
            # Calibrated probabilities via Vector Scaling
            scaled_logits = self.vector_scaler(logits_raw)
            calib_probs = F.softmax(scaled_logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(calib_probs))
            calib_conf = float(calib_probs[pred_class])
            
            # Normalized calibrated entropy & margin
            entropy_calib = -np.sum(calib_probs * np.log(calib_probs + 1e-10))
            calib_entropy_norm = float(entropy_calib / np.log(config.NUM_CLASSES))
            
            sorted_calib = np.sort(calib_probs)
            calib_margin = float(sorted_calib[-1] - sorted_calib[-2])

        # 3. Stochastic MC Dropout Uncertainty Estimation
        mc_probs_list = []
        if passes_N > 0:
            enable_foot_mc_dropout(self.model) # Dropout -> .train(), BatchNorm -> .eval()
            with torch.no_grad():
                for _ in range(passes_N):
                    out_mc = self.model(input_tensor)
                    logits_mc = out_mc["logits"] if isinstance(out_mc, dict) else out_mc
                    scaled_logits_mc = self.vector_scaler(logits_mc)
                    probs_mc = F.softmax(scaled_logits_mc, dim=1).cpu().numpy()[0]
                    mc_probs_list.append(probs_mc)
            
            # Restore deterministic eval mode
            self.model.eval()
            
            mc_probs = np.array(mc_probs_list) # shape (N, 4)
            mean_probs = np.mean(mc_probs, axis=0) # shape (4,)
            
            # Total Predictive Entropy H(p)
            p_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
            mc_pred_entropy = float(p_entropy)
            mc_pred_entropy_norm = float(p_entropy / np.log(config.NUM_CLASSES))
            
            # Aleatoric Expected Entropy E[H(p)]
            each_entropy = -np.sum(mc_probs * np.log(mc_probs + 1e-10), axis=1)
            mc_exp_entropy = float(np.mean(each_entropy))
            mc_exp_entropy_norm = float(mc_exp_entropy / np.log(config.NUM_CLASSES))
            
            # Predictive Variance Var(p)
            mc_pred_var = float(np.mean(np.var(mc_probs, axis=0, ddof=1 if passes_N > 1 else 0)))
            
            # Epistemic Mutual Information MI
            mc_mi = float(max(0.0, mc_pred_entropy - mc_exp_entropy))
        else:
            mc_pred_entropy = float(entropy_calib)
            mc_pred_entropy_norm = calib_entropy_norm
            mc_exp_entropy = float(entropy_calib)
            mc_exp_entropy_norm = calib_entropy_norm
            mc_pred_var = 0.0
            mc_mi = 0.0

        # 4. Grad-CAM Explainability Generation
        cam_overlay = None
        cam_heatmap = None
        cam_mean_intensity = 0.0
        
        if generate_cam:
            try:
                cam_mask, _ = self.gradcam.generate_cam(input_tensor, class_idx=pred_class)
                cam_resized = cv2.resize(cam_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                cam_mean_intensity = float(np.mean(cam_resized))
                
                heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
                
                overlay = np.uint8(0.6 * orig_rgb_np + 0.4 * heatmap_rgb)
                
                cam_overlay = overlay
                cam_heatmap = heatmap_rgb
            except Exception as e:
                # Log or handle CAM failure silently to preserve primary inference output
                pass

        latency = (time.time() - start_time) * 1000.0
        
        # Return Unified Contract Dictionary
        return {
            "modality": "foot",
            "prediction": pred_class,
            "prediction_label": config.CLASS_NAMES[pred_class],
            "raw_confidence": raw_conf,
            "calib_confidence": calib_conf,
            "raw_probabilities": [float(p) for p in raw_probs],
            "calib_probabilities": [float(p) for p in calib_probs],
            "calib_entropy_norm": calib_entropy_norm,
            "calib_margin": calib_margin,
            "mc_passes_N": passes_N,
            "mc_predictive_entropy": mc_pred_entropy,
            "mc_predictive_entropy_norm": mc_pred_entropy_norm,
            "mc_expected_entropy": mc_exp_entropy,
            "mc_expected_entropy_norm": mc_exp_entropy_norm,
            "mc_predictive_variance": mc_pred_var,
            "mc_mutual_information": mc_mi,
            "cam_overlay": cam_overlay,
            "cam_heatmap": cam_heatmap,
            "cam_mean_intensity": cam_mean_intensity,
            "latency_ms": latency,
            "model_info": {
                "name": "EfficientNet-B3",
                "calibration": "vector_scaling",
                "stochastic_passes_N": passes_N
            }
        }
