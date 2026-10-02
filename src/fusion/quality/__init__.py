"""
FusionMedAI Input Quality & Availability Layer (Phase C11.2).
"""

from src.fusion.quality.quality_result import (
    QualityResult,
    QualityContractError,
    make_unavailable_quality,
)
from src.fusion.quality.common import load_and_decode_image
from src.fusion.quality.retina_quality import (
    compute_retina_quality,
    compute_retina_sharpness,
    compute_retina_illumination,
)
from src.fusion.quality.foot_quality import (
    compute_foot_quality,
    compute_unsupervised_otsu_cnr,
    compute_sobel_boundary_clarity,
)
from src.fusion.quality.clinical_quality import (
    compute_clinical_quality,
    CLINICAL_EXPECTED_DIM,
)

__all__ = [
    "QualityResult",
    "QualityContractError",
    "make_unavailable_quality",
    "load_and_decode_image",
    "compute_retina_quality",
    "compute_retina_sharpness",
    "compute_retina_illumination",
    "compute_foot_quality",
    "compute_unsupervised_otsu_cnr",
    "compute_sobel_boundary_clarity",
    "compute_clinical_quality",
    "CLINICAL_EXPECTED_DIM",
]
