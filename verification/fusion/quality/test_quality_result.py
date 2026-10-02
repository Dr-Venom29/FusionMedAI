"""
Unit Tests for Phase C11.2 Unified Input Quality & Availability Layer.
Verifies contract invariants, availability safeguards, monotonicity under degradation,
and complete independence from model predictions and ground-truth targets.
"""

import os
import tempfile
import pytest
import numpy as np
import cv2
from PIL import Image

from src.fusion.quality.quality_result import (
    QualityResult,
    QualityContractError,
    make_unavailable_quality,
)
from src.fusion.quality.retina_quality import compute_retina_quality
from src.fusion.quality.foot_quality import compute_foot_quality
from src.fusion.quality.clinical_quality import compute_clinical_quality, CLINICAL_EXPECTED_DIM


# =========================================================================
# 1. Contract Invariant Tests
# =========================================================================

def test_quality_result_valid_instantiation():
    """Valid results within bounds must instantiate cleanly."""
    res_avail = QualityResult(availability=True, quality=0.85, diagnostics={"test": 1})
    assert res_avail.availability is True
    assert res_avail.quality == 0.85
    assert res_avail.diagnostics["test"] == 1

    res_unavail = QualityResult(availability=False, quality=0.0)
    assert res_unavail.availability is False
    assert res_unavail.quality == 0.0


def test_quality_result_rejects_invalid_bounds():
    """QualityResult must reject out-of-bounds, NaN, Inf, or invalid types."""
    with pytest.raises(QualityContractError):
        QualityResult(availability=True, quality=-0.1)

    with pytest.raises(QualityContractError):
        QualityResult(availability=True, quality=1.0001)

    with pytest.raises(QualityContractError):
        QualityResult(availability=True, quality=float("nan"))

    with pytest.raises(QualityContractError):
        QualityResult(availability=True, quality=float("inf"))

    with pytest.raises(QualityContractError):
        QualityResult(availability="True", quality=0.5)  # non-bool


def test_quality_result_enforces_unavailability_zero_quality():
    """Invariant: A_i = 0 => Q_i = 0.0 strictly."""
    with pytest.raises(QualityContractError):
        QualityResult(availability=False, quality=0.5)


def test_make_unavailable_quality_helper():
    """Helper must produce a canonical unavailable QualityResult."""
    unavail = make_unavailable_quality("MISSING_TEST")
    assert unavail.availability is False
    assert unavail.quality == 0.0
    assert unavail.diagnostics["status"] == "UNAVAILABLE"
    assert unavail.diagnostics["reason"] == "MISSING_TEST"


# =========================================================================
# 2. Retina Quality & Availability Tests
# =========================================================================

def test_retina_quality_synthetic_image():
    """Valid synthetic fundus-like image produces A_R=1 and Q_R in [0, 1]."""
    # Create synthetic image with circles/textures resembling fundus
    np.random.seed(42)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(img, (128, 128), 100, (60, 80, 180), -1)  # orange-red fundus disk
    cv2.circle(img, (90, 110), 15, (10, 10, 10), -1)    # fovea
    cv2.line(img, (128, 128), (50, 50), (20, 20, 120), 3) # blood vessel

    res = compute_retina_quality(img)
    assert res.availability is True
    assert 0.0 <= res.quality <= 1.0
    assert res.diagnostics["status"] == "AVAILABLE"
    assert res.diagnostics["sharpness_raw"] > 0


def test_retina_quality_missing_and_corrupt_inputs():
    """Retina engine must return A_R=0, Q_R=0 for missing or invalid inputs."""
    # None input
    assert compute_retina_quality(None).availability is False
    assert compute_retina_quality(None).quality == 0.0

    # Non-existent file
    assert compute_retina_quality("non_existent_retina_path_xyz.png").availability is False
    assert compute_retina_quality("non_existent_retina_path_xyz.png").quality == 0.0

    # Corrupt file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tf.write(b"NOT_A_VALID_IMAGE_BYTES")
        tf_path = tf.name

    try:
        res = compute_retina_quality(tf_path)
        assert res.availability is False
        assert res.quality == 0.0
    finally:
        if os.path.exists(tf_path):
            os.remove(tf_path)


def test_retina_quality_blur_monotonicity():
    """Applying increasing Gaussian blur must monotonically decrease sharpness."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(img, (128, 128), 80, (150, 150, 150), -1)
    for i in range(20, 240, 20):
        cv2.line(img, (i, 0), (i, 255), (255, 255, 255), 2)

    blurred_mild = cv2.GaussianBlur(img, (7, 7), 2.0)
    blurred_heavy = cv2.GaussianBlur(img, (25, 25), 8.0)

    q_orig = compute_retina_quality(img)
    q_mild = compute_retina_quality(blurred_mild)
    q_heavy = compute_retina_quality(blurred_heavy)

    assert q_orig.diagnostics["q_sharpness"] >= q_mild.diagnostics["q_sharpness"]
    assert q_mild.diagnostics["q_sharpness"] >= q_heavy.diagnostics["q_sharpness"]


def test_retina_quality_illumination_sensitivity():
    """Severe underexposure must reduce illumination quality."""
    img = np.full((256, 256, 3), 70, dtype=np.uint8)
    cv2.circle(img, (128, 128), 60, (120, 120, 120), -1)

    dark_img = np.clip(img * 0.1, 0, 255).astype(np.uint8)  # near black

    q_normal = compute_retina_quality(img)
    q_dark = compute_retina_quality(dark_img)

    assert q_normal.diagnostics["q_illumination"] > q_dark.diagnostics["q_illumination"]


# =========================================================================
# 3. Foot Quality & Availability Tests
# =========================================================================

def test_foot_quality_synthetic_image():
    """Valid synthetic foot wound image produces A_F=1 and Q_F in [0, 1]."""
    img = np.full((256, 256, 3), 180, dtype=np.uint8)  # skin tone
    cv2.circle(img, (128, 128), 50, (40, 40, 120), -1) # ulcer center
    cv2.circle(img, (128, 128), 60, (70, 70, 160), 4)  # wound margin

    res = compute_foot_quality(img)
    assert res.availability is True
    assert 0.0 <= res.quality <= 1.0
    assert res.diagnostics["status"] == "AVAILABLE"
    assert res.diagnostics["otsu_cnr_raw"] > 0
    assert res.diagnostics["sobel_gradient_magnitude_raw"] > 0


def test_foot_quality_missing_and_corrupt_inputs():
    """Foot engine must return A_F=0, Q_F=0 for missing or invalid inputs."""
    assert compute_foot_quality(None).availability is False
    assert compute_foot_quality(None).quality == 0.0

    assert compute_foot_quality("missing_foot_path.jpg").availability is False
    assert compute_foot_quality("missing_foot_path.jpg").quality == 0.0


def test_foot_quality_contrast_and_blur_monotonicity():
    """Reducing contrast and blurring must degrade Foot CNR and edge quality."""
    img = np.full((256, 256, 3), 180, dtype=np.uint8)
    cv2.circle(img, (128, 128), 50, (30, 30, 90), -1)

    blurred = cv2.GaussianBlur(img, (21, 21), 6.0)

    q_orig = compute_foot_quality(img)
    q_blur = compute_foot_quality(blurred)

    assert q_orig.diagnostics["q_boundary"] >= q_blur.diagnostics["q_boundary"]


# =========================================================================
# 4. Clinical Completeness & Availability Tests
# =========================================================================

def test_clinical_quality_perfect_feature_vector():
    """119/119 valid features yields A_C=1 and Q_C=1.0."""
    feat_119 = np.ones(CLINICAL_EXPECTED_DIM, dtype=float)
    res = compute_clinical_quality(feat_119)
    assert res.availability is True
    assert res.quality == 1.0
    assert res.diagnostics["valid_features"] == 119
    assert res.diagnostics["missing_features"] == 0


def test_clinical_quality_partial_missingness_exact_ratio():
    """Partially missing features (NaNs) yield exact proportional quality."""
    feat_119 = np.ones(CLINICAL_EXPECTED_DIM, dtype=float)
    # Set 19 features to NaN
    feat_119[:19] = np.nan
    
    res = compute_clinical_quality(feat_119)
    assert res.availability is True
    expected_quality = 100.0 / 119.0
    assert abs(res.quality - expected_quality) < 1e-4
    assert res.diagnostics["valid_features"] == 100
    assert res.diagnostics["missing_features"] == 19


def test_clinical_quality_monotonicity():
    """Strict monotonicity: Q_C(n+1) > Q_C(n) as valid feature count increases."""
    qualities = []
    for n_valid in [20, 50, 80, 100, 119]:
        feat = np.full(CLINICAL_EXPECTED_DIM, np.nan)
        feat[:n_valid] = 1.0
        res = compute_clinical_quality(feat)
        qualities.append(res.quality)

    for i in range(len(qualities) - 1):
        assert qualities[i] < qualities[i + 1]


def test_clinical_quality_invalid_dimension_rejection():
    """Vectors not matching 119-D contract must be rejected with A_C=0, Q_C=0."""
    feat_wrong = np.ones(50, dtype=float)
    res = compute_clinical_quality(feat_wrong)
    assert res.availability is False
    assert res.quality == 0.0


def test_clinical_quality_missing_input():
    """Missing clinical input produces A_C=0, Q_C=0."""
    res = compute_clinical_quality(None)
    assert res.availability is False
    assert res.quality == 0.0


# =========================================================================
# 5. Independence from Model Predictions and Ground-Truth Labels
# =========================================================================

def test_quality_computation_independence():
    """
    Quality computation must produce identical (A, Q) regardless of hypothetical
    downstream model predictions, confidences, or ground-truth class labels.
    """
    img = np.full((128, 128, 3), 100, dtype=np.uint8)
    cv2.circle(img, (64, 64), 30, (200, 200, 200), -1)

    # Calling compute_retina_quality only requires the image
    q1 = compute_retina_quality(img)
    q2 = compute_retina_quality(img)

    assert q1.availability == q2.availability
    assert q1.quality == q2.quality
    assert q1.diagnostics == q2.diagnostics
