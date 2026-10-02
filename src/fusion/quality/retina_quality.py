"""
Retina Fundus Quality & Availability Engine (Phase C11.2).
Evaluates raw retinal fundus image quality based on Laplacian sharpness and illumination adequacy.
Strictly decoupled from model predictions and ground-truth disease labels.
"""

from typing import Union, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from src.fusion.quality.quality_result import QualityResult, make_unavailable_quality
from src.fusion.quality.common import load_and_decode_image

# Frozen empirical constants from training split
RETINA_SHARP_LOW: float = 4.0
RETINA_SHARP_HIGH: float = 55.0

RETINA_ILLUM_DARK_THRESHOLD: float = 10.0
RETINA_ILLUM_OPT_LOW: float = 35.0
RETINA_ILLUM_OPT_HIGH: float = 95.0
RETINA_ILLUM_BRIGHT_THRESHOLD: float = 140.0
RETINA_ILLUM_STD_TARGET: float = 20.0


def compute_retina_sharpness(gray_img: np.ndarray) -> Tuple[float, float]:
    """
    Computes raw Laplacian variance and normalized sharpness score in [0.0, 1.0].
    """
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    s_raw = float(laplacian.var())
    
    # Clip and normalize
    q_sharp = float(np.clip(
        (s_raw - RETINA_SHARP_LOW) / (RETINA_SHARP_HIGH - RETINA_SHARP_LOW),
        0.0,
        1.0
    ))
    return s_raw, q_sharp


def compute_retina_illumination(gray_img: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes mean luminance, contrast dispersion, and composite illumination score in [0.0, 1.0].
    """
    mu_i = float(gray_img.mean())
    sigma_i = float(gray_img.std())
    
    # Mean luminance score (penalizes underexposure and overexposure)
    if mu_i < RETINA_ILLUM_OPT_LOW:
        q_mean = np.clip(
            (mu_i - RETINA_ILLUM_DARK_THRESHOLD) / (RETINA_ILLUM_OPT_LOW - RETINA_ILLUM_DARK_THRESHOLD),
            0.0,
            1.0
        )
    elif mu_i <= RETINA_ILLUM_OPT_HIGH:
        q_mean = 1.0
    else:
        q_mean = np.clip(
            (RETINA_ILLUM_BRIGHT_THRESHOLD - mu_i) / (RETINA_ILLUM_BRIGHT_THRESHOLD - RETINA_ILLUM_OPT_HIGH),
            0.0,
            1.0
        )
    
    # Dynamic range / contrast score
    q_std = np.clip(sigma_i / RETINA_ILLUM_STD_TARGET, 0.0, 1.0)
    
    q_illum = float(np.clip(q_mean * q_std, 0.0, 1.0))
    return mu_i, sigma_i, q_illum


def compute_retina_quality(
    image_input: Optional[Union[str, Path, np.ndarray, Image.Image]]
) -> QualityResult:
    """
    Evaluates input availability and signal quality for a retinal fundus scan.
    
    Args:
        image_input: File path, numpy array, or PIL Image of fundus scan.
        
    Returns:
        QualityResult with availability: bool, quality: float in [0.0, 1.0], and diagnostics.
    """
    if image_input is None:
        return make_unavailable_quality("INPUT_MISSING")

    success, rgb_img, reason = load_and_decode_image(image_input)
    if not success or rgb_img is None:
        return make_unavailable_quality(reason)

    if rgb_img.shape[0] < 16 or rgb_img.shape[1] < 16:
        return make_unavailable_quality(f"IMAGE_TOO_SMALL_{rgb_img.shape[:2]}")

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        
        # 1. Sharpness
        s_raw, q_sharp = compute_retina_sharpness(gray)
        
        # 2. Illumination
        mu_i, sigma_i, q_illum = compute_retina_illumination(gray)
        
        # 3. Deterministic aggregation
        q_retina = float(np.clip(0.5 * (q_sharp + q_illum), 0.0, 1.0))
        
        diagnostics = {
            "status": "AVAILABLE",
            "sharpness_raw": round(s_raw, 4),
            "q_sharpness": round(q_sharp, 4),
            "mean_luminance": round(mu_i, 4),
            "std_luminance": round(sigma_i, 4),
            "q_illumination": round(q_illum, 4),
            "image_shape": list(rgb_img.shape),
        }
        
        return QualityResult(
            availability=True,
            quality=q_retina,
            diagnostics=diagnostics
        )
        
    except Exception as e:
        return make_unavailable_quality(f"QUALITY_COMPUTATION_ERROR_{type(e).__name__}")
