import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union
from src.foot.models.factory import build_foot_final_model
from src.foot.xai.gradcam import FootGradCAM

def compute_cam_concentration(cam: np.ndarray, top_k_percentages: Tuple[float, ...] = (0.10, 0.20)) -> Dict[str, float]:
    """
    Calculates attribution mass concentration within top K% highest activation pixels.
    
    Args:
        cam: 2D array of shape (H, W), values in [0, 1]
        top_k_percentages: Tuple of fractions e.g. (0.10, 0.20)
        
    Returns:
        Dict mapping metric name to ratio of total attribution sum.
    """
    total_mass = np.sum(cam)
    if total_mass < 1e-8:
        return {f"top_{int(p*100)}pct_concentration": 0.0 for p in top_k_percentages}
        
    flat_cam = np.sort(cam.flatten())[::-1]
    total_pixels = len(flat_cam)
    
    res = {}
    for p in top_k_percentages:
        k_pixels = int(p * total_pixels)
        top_sum = np.sum(flat_cam[:k_pixels])
        res[f"top_{int(p*100)}pct_concentration"] = round(float(top_sum / total_mass), 4)
    return res

def run_model_randomization_test(
    trained_model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    target_layer_name: str = "backbone.features.8"
) -> Dict[str, Union[float, bool]]:
    """
    Adebayo et al. Model Randomization Sanity Check.
    Compares Grad-CAM from trained model vs an untrained (randomly initialized) model.
    """
    # Trained CAM
    gradcam_trained = FootGradCAM(model=trained_model, target_layer_name=target_layer_name, device="cpu")
    cam_trained, _ = gradcam_trained.generate_cam(input_tensor, class_idx=target_class)
    gradcam_trained.remove_hooks()
    
    # Randomized model
    random_model = build_foot_final_model(pretrained=False)
    gradcam_random = FootGradCAM(model=random_model, target_layer_name=target_layer_name, device="cpu")
    cam_random, _ = gradcam_random.generate_cam(input_tensor, class_idx=target_class)
    gradcam_random.remove_hooks()
    
    flat_t = cam_trained.flatten()
    flat_r = cam_random.flatten()
    
    std_t = np.std(flat_t)
    std_r = np.std(flat_r)
    
    degenerate = bool(std_t < 1e-8 or std_r < 1e-8)
    if degenerate:
        pcc = 0.0
    else:
        pcc = float(np.corrcoef(flat_t, flat_r)[0, 1])
        
    return {
        "randomization_pcc": round(pcc, 4),
        "degenerate_cam": degenerate,
        "cam_trained_max": round(float(cam_trained.max()), 4),
        "cam_random_max": round(float(cam_random.max()), 4)
    }

def run_target_class_test(
    model: nn.Module,
    input_tensor: torch.Tensor,
    pred_class: int,
    alt_class: int,
    target_layer_name: str = "backbone.features.8"
) -> Dict[str, Union[float, bool]]:
    """
    Target-Class Sensitivity Check.
    Compares Grad-CAM for predicted class vs an alternative target class.
    """
    gradcam = FootGradCAM(model=model, target_layer_name=target_layer_name, device="cpu")
    cam_pred, _ = gradcam.generate_cam(input_tensor, class_idx=pred_class)
    cam_alt, _ = gradcam.generate_cam(input_tensor, class_idx=alt_class)
    gradcam.remove_hooks()
    
    flat_p = cam_pred.flatten()
    flat_a = cam_alt.flatten()
    
    std_p = np.std(flat_p)
    std_a = np.std(flat_a)
    
    degenerate = bool(std_p < 1e-8 or std_a < 1e-8)
    if degenerate:
        pcc = float("nan")
    else:
        pcc = float(np.corrcoef(flat_p, flat_a)[0, 1])
        
    return {
        "class_contrast_pcc": round(pcc, 4) if not np.isnan(pcc) else None,
        "degenerate_cam": degenerate,
        "cam_pred_max": round(float(cam_pred.max()), 4),
        "cam_alt_max": round(float(cam_alt.max()), 4)
    }
