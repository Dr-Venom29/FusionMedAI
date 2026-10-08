"""
src/fusion/degradation/image_degradation.py
Phase C11.10: Image Degradation Operators & Live Quality Computation Engine

Applies deterministic image degradations:
- D-R1 / D-F1: Gaussian Blur (attenuates spatial Laplacian variance / Sobel boundary clarity)
- D-R2 / D-F2: Contrast Attenuation (compresses dynamic range / Otsu CNR)
- D-R3 / D-F3: Illumination Shift (penalizes luminance in dark zone / intensity compression)
- D-R4 / D-F4: Synthetic Opacity & Boundary Smudge Artifacts

Coupled directly to the frozen unsupervised quality engines:
- compute_retina_quality()
- compute_foot_quality()
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image

from src.fusion.degradation.degradation_spec import (
    SEVERITY_D0_CLEAN,
    SEVERITY_D1_MILD,
    SEVERITY_D2_MODERATE,
    SEVERITY_D3_SEVERE,
    OP_RETINA_BLUR,
    OP_RETINA_CONTRAST,
    OP_RETINA_ILLUMINATION,
    OP_RETINA_ARTIFACT,
    OP_FOOT_BLUR,
    OP_FOOT_CONTRAST,
    OP_FOOT_ILLUMINATION,
    OP_FOOT_ARTIFACT,
    get_operator_params,
    deterministic_sample_seed,
)
from src.fusion.quality.retina_quality import compute_retina_quality
from src.fusion.quality.foot_quality import compute_foot_quality
from src.fusion.quality.quality_result import QualityResult
from src.fusion.baselines.decision_packet import ModalityRecord


# =============================================================================
# 1. Benchmark Raw Image Generators
# =============================================================================

def generate_benchmark_retina_image(sample_id: str = "retina_0", seed: int = 115) -> np.ndarray:
    """
    Generates a deterministic 256x256x3 synthetic fundus image resembling a clean scan.
    Produces high baseline sharpness (q_sharp >= 0.95) and optimal illumination (q_illum >= 0.95).
    Guarantees cross-process bitwise repeatability via cryptographic seed derivation.
    """
    h_seed = deterministic_sample_seed(sample_id, seed)
    rng = np.random.RandomState(h_seed)

    img = np.full((256, 256, 3), 45, dtype=np.uint8)
    # Fundus orange-red disk
    cv2.circle(img, (128, 128), 105, (55, 75, 185), -1)
    # Retinal vessel network (high Laplacian variance)
    for _ in range(15):
        x1, y1 = rng.randint(40, 216), rng.randint(40, 216)
        x2, y2 = rng.randint(40, 216), rng.randint(40, 216)
        cv2.line(img, (x1, y1), (x2, y2), (20, 30, 110), 2)
    # Fovea / macula region
    cv2.circle(img, (95, 115), 14, (15, 15, 25), -1)
    return img


def generate_benchmark_foot_image(sample_id: str = "foot_0", seed: int = 115) -> np.ndarray:
    """
    Generates a deterministic 256x256x3 synthetic foot ulcer image with clear wound boundary.
    Produces strong Otsu CNR and high Sobel boundary gradient magnitude.
    Guarantees cross-process bitwise repeatability via cryptographic seed derivation.
    """
    h_seed = deterministic_sample_seed(sample_id, seed)
    rng = np.random.RandomState(h_seed)

    img = np.full((256, 256, 3), 190, dtype=np.uint8)  # skin background
    # Dark ulcer center
    cv2.circle(img, (128, 128), 60, (25, 25, 65), -1)
    # High-contrast ulcer wound margin
    cv2.circle(img, (128, 128), 75, (75, 50, 110), 10)
    # Tissue granulation lines
    for x in range(30, 226, 12):
        cv2.line(img, (x, 30), (x, 226), (140, 140, 170), 3)
    return img



# =============================================================================
# 2. Pixel-Level Transformation Functions
# =============================================================================

def apply_gaussian_blur(img: np.ndarray, sigma: float, kernel_size: int) -> np.ndarray:
    """Applies deterministic Gaussian blur."""
    if sigma <= 0.0 or kernel_size <= 1:
        return img.copy()
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)


def apply_contrast_attenuation(img: np.ndarray, factor: float) -> np.ndarray:
    """Applies deterministic dynamic range contrast compression around mean intensity."""
    if factor >= 1.0:
        return img.copy()
    mean_val = np.mean(img, axis=(0, 1), keepdims=True)
    attenuated = mean_val + factor * (img.astype(np.float32) - mean_val)
    return np.clip(attenuated, 0, 255).astype(np.uint8)


def apply_illumination_shift(img: np.ndarray, shift: int) -> np.ndarray:
    """Applies deterministic luminance shift into the dark penalty zone."""
    if shift == 0:
        return img.copy()
    shifted = img.astype(np.int32) + shift
    return np.clip(shifted, 0, 255).astype(np.uint8)


def apply_foot_illumination(img: np.ndarray, lum_factor: float) -> np.ndarray:
    """Applies intensity scaling / compression for foot ulcer images."""
    if lum_factor >= 1.0:
        return img.copy()
    scaled = img.astype(np.float32) * lum_factor
    return np.clip(scaled, 0, 255).astype(np.uint8)


def apply_smudge_artifact(img: np.ndarray, radius: int, seed: int = 115) -> np.ndarray:
    """
    Applies deterministic opacity smudge / cataract / dressing occlusion artifact
    without false high-frequency edge discontinuities.
    """
    if radius <= 0:
        return img.copy()
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    cv2.circle(mask, (img.shape[1] // 2, img.shape[0] // 2), radius, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 10.0)

    blurred_bg = cv2.GaussianBlur(img, (35, 35), 15.0)
    out = img.copy()
    for c in range(img.shape[2] if img.ndim == 3 else 1):
        if img.ndim == 3:
            out[:, :, c] = ((1.0 - mask) * img[:, :, c] + mask * blurred_bg[:, :, c]).astype(np.uint8)
        else:
            out = ((1.0 - mask) * img + mask * blurred_bg).astype(np.uint8)
    return out


# =============================================================================
# 3. Live Raw-Input Operator Dispatcher
# =============================================================================

def apply_image_degradation(
    img: np.ndarray,
    operator: str,
    severity: str,
    seed: int = 115,
) -> np.ndarray:
    """Applies a specified image degradation operator to a raw image array."""
    params = get_operator_params(operator, severity)

    # Retina
    if operator == OP_RETINA_BLUR:
        return apply_gaussian_blur(img, params["sigma"], params["kernel_size"])
    elif operator == OP_RETINA_CONTRAST:
        return apply_contrast_attenuation(img, params["factor"])
    elif operator == OP_RETINA_ILLUMINATION:
        return apply_illumination_shift(img, params["shift"])
    elif operator == OP_RETINA_ARTIFACT:
        return apply_smudge_artifact(img, params["radius"], seed=seed)

    # Foot
    elif operator == OP_FOOT_BLUR:
        return apply_gaussian_blur(img, params["sigma"], params["kernel_size"])
    elif operator == OP_FOOT_CONTRAST:
        return apply_contrast_attenuation(img, params["factor"])
    elif operator == OP_FOOT_ILLUMINATION:
        return apply_foot_illumination(img, params["lum_factor"])
    elif operator == OP_FOOT_ARTIFACT:
        return apply_smudge_artifact(img, params["radius"], seed=seed)
    else:
        raise ValueError(f"Unsupported image degradation operator: {operator}")


def compute_degraded_image_quality(
    img: np.ndarray,
    modality: str,
    operator: str,
    severity: str,
    seed: int = 115,
) -> QualityResult:
    """Applies actual image degradation and evaluates the live frozen quality engine."""
    degraded_img = apply_image_degradation(img, operator, severity, seed=seed)
    if modality == "retina":
        return compute_retina_quality(degraded_img)
    elif modality == "foot":
        return compute_foot_quality(degraded_img)
    else:
        raise ValueError(f"Unsupported image modality: {modality}")


# =============================================================================
# 4. ModalityRecord Transformation with Live Quality Coupling
# =============================================================================

def apply_image_degradation_to_record(
    record: ModalityRecord,
    operator: str,
    severity: str,
    live_computed_quality: Optional[float] = None,
) -> ModalityRecord:
    """
    Updates a ModalityRecord with the live computed quality score from Experiment A.
    """
    if not record.availability:
        return record

    params = get_operator_params(operator, severity)
    u_boost = float(params.get("uncertainty_boost", 0.0))

    if live_computed_quality is not None:
        new_q = float(np.clip(live_computed_quality, 0.0, 1.0))
    else:
        # Fallback to parameter multiplier if live evaluation is bypassed
        q_mult = float(params.get("quality_multiplier", 1.0))
        new_q = float(np.clip(record.quality * q_mult, 0.0, 1.0))

    new_u = float(np.clip(record.uncertainty + u_boost, 0.0, 1.0))
    new_c = float(np.clip(record.confidence * (1.0 - 0.2 * u_boost), 0.0, 1.0))

    return ModalityRecord(
        sample_id=record.sample_id,
        modality=record.modality,
        risk=record.risk,
        calibrated_probability=record.calibrated_probability,
        confidence=new_c,
        uncertainty=new_u,
        quality=new_q,
        availability=record.availability,
        reliability=record.reliability,
        model_version=record.model_version,
    )
