import os
import sys
import time
from pathlib import Path
from typing import Union, Dict, Any, List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.models.model_factory import load_model
from src.training.checkpoint import load_checkpoint
from src.data.transforms import get_val_transforms
from src.xai.gradcam import GradCAM
from src.xai.cam import BaseCAM
from src.xai.utils import get_target_layer
from src.uncertainty.utils import load_and_verify_calibration

class RetinaModule:
    """
    RetinaModule integrates the trained EfficientNet-B3 classifier, post-hoc temperature scaling,
    stochastic MC Dropout uncertainty estimation, and explainability via Grad-CAM.
    """
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        calibration_dir: Optional[Union[str, Path]] = None,
        override_temp: Optional[float] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Initializes the integrated Retina Module.
        
        Args:
            checkpoint_path: Path to the trained EfficientNet-B3 model. If None, resolves config.
            calibration_dir: Path to temperature scaling calibration. If None, resolves config.
            override_temp: Optional manual temperature value to override file loading.
            device: Execution device (cpu or cuda). If None, resolves config.
        """
        self.device = device if device is not None else config.DEVICE
        
        # 1. Resolve path to checkpoint
        if checkpoint_path is None:
            checkpoint_path = Path(config.PROJECT_ROOT) / "experiments" / "efficientnet_b3" / "checkpoints" / "best_model.pt"
        self.checkpoint_path = Path(checkpoint_path)
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at '{self.checkpoint_path}'")
            
        # 2. Load model architecture
        self.model = load_model(
            name=config.MODEL_NAME,
            num_classes=config.NUM_CLASSES,
            pretrained=False
        )
        
        # 3. Load weights
        load_checkpoint(
            checkpoint_path=self.checkpoint_path,
            model=self.model,
            device=self.device
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 4. Load Temperature Calibration factor T dynamically
        self.calibration_dir = Path(calibration_dir) if calibration_dir is not None else None
        self.temperature, self.calib_meta = load_and_verify_calibration(
            checkpoint_path=self.checkpoint_path,
            calibration_dir=self.calibration_dir,
            override_temp=override_temp
        )
        
        # 5. Transforms
        self.transform = get_val_transforms()
        
        # 6. Locate Target Layer for Grad-CAM
        self.target_layer = get_target_layer(self.model, config.MODEL_NAME)
        
        # 7. Identify Dropout layer for MC Dropout
        self.dropout_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                self.dropout_layer = module
                break
                
    def predict(
        self,
        image: Union[str, Path, Image.Image],
        mc_passes: int = 25,
        generate_cam: bool = True
    ) -> Dict[str, Any]:
        """
        Performs a comprehensive prediction on a single ocular scan.
        
        Incorporates:
        - Deterministic inference
        - Temperature calibration
        - MC Dropout ensembling (uncertainty estimation)
        - Grad-CAM heatmap overlay
        
        Args:
            image: Path to input fundus image, or pre-loaded PIL Image.
            mc_passes: Number of Monte Carlo passes for uncertainty estimation.
            generate_cam: Whether to generate Grad-CAM explainability heatmaps.
            
        Returns:
            Dict[str, Any]: Integrated diagnostic outputs.
        """
        start_time = time.time()
        
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at '{img_path}'")
            with Image.open(img_path) as img:
                pil_image = img.convert("RGB")
        else:
            pil_image = image.convert("RGB")
            
        # Transform image
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # 1. Deterministic Inference Pass
        self.model.eval()
        with torch.no_grad():
            deterministic_logits = self.model(input_tensor)
            
            # Uncalibrated outputs
            raw_probs = F.softmax(deterministic_logits, dim=1).cpu().numpy()[0]
            raw_pred_class = int(np.argmax(raw_probs))
            raw_conf = float(raw_probs[raw_pred_class])
            
            # Calibrated outputs
            calib_logits = deterministic_logits / self.temperature
            calib_probs = F.softmax(calib_logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(calib_probs))
            calib_conf = float(calib_probs[pred_class])
            
            # Calibrated entropy and margin
            entropy = -np.sum(calib_probs * np.log(calib_probs + 1e-10))
            calib_entropy_norm = float(entropy / np.log(config.NUM_CLASSES))
            
            sorted_probs = np.sort(calib_probs)
            calib_margin = float(sorted_probs[-1] - sorted_probs[-2])
            
        # 2. Stochastic MC Dropout Pass
        mc_probs = []
        if mc_passes > 0 and self.dropout_layer is not None:
            # Set dropout to train mode, keep the rest of the model in eval
            self.model.eval()
            self.dropout_layer.train()
            
            with torch.no_grad():
                for _ in range(mc_passes):
                    logits = self.model(input_tensor)
                    # Apply temperature scaling before softmax
                    calib_logits_mc = logits / self.temperature
                    probs_mc = F.softmax(calib_logits_mc, dim=1).cpu().numpy()[0]
                    mc_probs.append(probs_mc)
            
            # Restore model state
            self.dropout_layer.eval()
            
            mc_probs = np.array(mc_probs) # shape (N, num_classes)
            mean_probs = np.mean(mc_probs, axis=0) # shape (num_classes,)
            
            # Compute predictive entropy (total uncertainty)
            p_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
            mc_pred_entropy = float(p_entropy)
            mc_pred_entropy_norm = float(p_entropy / np.log(config.NUM_CLASSES))
            
            # Compute expected entropy (aleatoric proxy)
            each_entropy = -np.sum(mc_probs * np.log(mc_probs + 1e-10), axis=1)
            mc_exp_entropy = float(np.mean(each_entropy))
            mc_exp_entropy_norm = float(mc_exp_entropy / np.log(config.NUM_CLASSES))
            
            # Compute predictive variance
            mc_pred_var = float(np.mean(np.var(mc_probs, axis=0)))
            
            # Compute mutual information (epistemic proxy)
            mc_mi = float(mc_pred_entropy - mc_exp_entropy)
            
        else:
            # Fallback if no dropout layer or mc_passes = 0
            mc_pred_entropy = calib_entropy_norm * np.log(config.NUM_CLASSES)
            mc_pred_entropy_norm = calib_entropy_norm
            mc_exp_entropy = mc_pred_entropy
            mc_exp_entropy_norm = calib_entropy_norm
            mc_pred_var = 0.0
            mc_mi = 0.0
            
        # 3. Explainability (Grad-CAM)
        cam_overlay = None
        cam_heatmap = None
        cam_mean_intensity = 0.0
        
        if generate_cam and self.target_layer is not None:
            # Temporarily activate Grad-CAM extractor
            cam_extractor = GradCAM(self.model, self.target_layer)
            self.model.eval()
            
            try:
                # Forward pass to record activations and gradients
                raw_cam = cam_extractor.forward(input_tensor, class_idx=pred_class)
                norm_cam = BaseCAM.normalize_cam(raw_cam)
                cam_mean_intensity = float(np.mean(norm_cam))
                
                # Recover original image for overlay
                mean = np.array(config.NORMALIZATION_MEAN)
                std = np.array(config.NORMALIZATION_STD)
                img_np = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
                original_img = std * img_np + mean
                original_img = np.clip(original_img, 0, 1)
                
                overlay, heatmap = BaseCAM.overlay_cam(original_img, norm_cam)
                
                # Convert float RGB (0..1) to uint8
                cam_overlay = (overlay * 255).astype(np.uint8)
                cam_heatmap = (heatmap * 255).astype(np.uint8)
            except Exception as e:
                # Silent fallback / log error if CAM fails
                pass
            finally:
                cam_extractor.remove_hooks()
                
        latency = (time.time() - start_time) * 1000
        
        return {
            "prediction": pred_class,
            "prediction_label": config.CLASS_NAMES[pred_class],
            "raw_confidence": raw_conf,
            "calib_confidence": calib_conf,
            "raw_probabilities": [float(p) for p in raw_probs],
            "calib_probabilities": [float(p) for p in calib_probs],
            "calib_entropy_norm": calib_entropy_norm,
            "calib_margin": calib_margin,
            "mc_predictive_entropy": mc_pred_entropy,
            "mc_predictive_entropy_norm": mc_pred_entropy_norm,
            "mc_expected_entropy": mc_exp_entropy,
            "mc_expected_entropy_norm": mc_exp_entropy_norm,
            "mc_predictive_variance": mc_pred_var,
            "mc_mutual_information": mc_mi,
            "cam_overlay": cam_overlay,
            "cam_heatmap": cam_heatmap,
            "cam_mean_intensity": cam_mean_intensity,
            "latency_ms": latency
        }
