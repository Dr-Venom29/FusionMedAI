"""
Diabetic Foot Ulcer Quality & Availability Engine (Phase C11.2).
Evaluates raw foot ulcer image quality using unsupervised tissue CNR and boundary edge clarity.
Zero ground-truth segmentation masks or disease labels used.
"""

from typing import Union, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from src.fusion.quality.quality_result import QualityResult, make_unavailable_quality
from src.fusion.quality.common import load_and_decode_image

# Frozen empirical constants from training split
FOOT_CNR_LOW: float = 1.8
FOOT_CNR_HIGH: float = 13.0

FOOT_EDGE_LOW: float = 15.0
FOOT_EDGE_HIGH: float = 60.0


def compute_unsupervised_otsu_cnr(gray_img: np.ndarray) -> Tuple[float, float]:
    """
    Computes unsupervised Contrast-to-Noise Ratio (CNR) between foreground and background
    derived entirely via Otsu thresholding without ground-truth masks.
    """
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    fg = gray_img[thresh == 255]
    bg = gray_img[thresh == 0]
    
    # Fallback if partition is trivial (< 1% pixels)
    if len(fg) < (0.01 * gray_img.size) or len(bg) < (0.01 * gray_img.size):
        median_val = np.median(gray_img)
        fg = gray_img[gray_img >= median_val]
        bg = gray_img[gray_img < median_val]
    
    mu_fg = float(fg.mean()) if len(fg) > 0 else 0.0
    mu_bg = float(bg.mean()) if len(bg) > 0 else 0.0
    sigma_bg = float(bg.std()) if len(bg) > 0 else 1.0
    
    cnr_raw = abs(mu_fg - mu_bg) / (sigma_bg + 1e-5)
    
    q_cnr = float(np.clip(
        (cnr_raw - FOOT_CNR_LOW) / (FOOT_CNR_HIGH - FOOT_CNR_LOW),
        0.0,
        1.0
    ))
    return cnr_raw, q_cnr


def compute_sobel_boundary_clarity(gray_img: np.ndarray) -> Tuple[float, float]:
    """
    Computes spatial boundary clarity using Sobel mean gradient magnitude across the image.
    """
    gx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    edge_raw = float(grad_mag.mean())
    
    q_boundary = float(np.clip(
        (edge_raw - FOOT_EDGE_LOW) / (FOOT_EDGE_HIGH - FOOT_EDGE_LOW),
        0.0,
        1.0
    ))
    return edge_raw, q_boundary


def compute_foot_quality(
    image_input: Optional[Union[str, Path, np.ndarray, Image.Image]]
) -> QualityResult:
    """
    Evaluates input availability and signal quality for a diabetic foot ulcer photograph.
    
    Args:
        image_input: File path, numpy array, or PIL Image of foot ulcer image.
        
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
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        
        # 1. Unsupervised Contrast-to-Noise Ratio (CNR)
        cnr_raw, q_cnr = compute_unsupervised_otsu_cnr(gray)
        
        # 2. Boundary / Edge Clarity (Sobel Gradient-Magnitude)
        edge_raw, q_boundary = compute_sobel_boundary_clarity(gray)
        
        # 3. Deterministic Aggregation
        q_foot = float(np.clip(0.5 * (q_cnr + q_boundary), 0.0, 1.0))
        
        diagnostics = {
            "status": "AVAILABLE",
            "otsu_cnr_raw": round(cnr_raw, 4),
            "q_cnr": round(q_cnr, 4),
            "sobel_gradient_magnitude_raw": round(edge_raw, 4),
            "q_boundary": round(q_boundary, 4),
            "image_shape": list(rgb_img.shape),
        }
        
        return QualityResult(
            availability=True,
            quality=q_foot,
            diagnostics=diagnostics
        )
        
    except Exception as e:
        return make_unavailable_quality(f"QUALITY_COMPUTATION_ERROR_{type(e).__name__}")
