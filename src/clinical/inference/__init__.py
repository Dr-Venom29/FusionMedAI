"""
Clinical Inference Module (Phase C10).
Exposes ClinicalInferenceService, ClinicalOutput schema, and validation utilities.
"""

from .schema import (
    ClinicalOutput,
    UncertaintyOutput,
    PredictiveInterval,
    FeatureAttribution,
    ShiftDetection,
    ModelProvenance,
)
from .validator import (
    ClinicalValidationError,
    validate_single_encounter,
    validate_batch_encounters,
)
from .service import ClinicalInferenceService

__all__ = [
    "ClinicalOutput",
    "UncertaintyOutput",
    "PredictiveInterval",
    "FeatureAttribution",
    "ShiftDetection",
    "ModelProvenance",
    "ClinicalValidationError",
    "validate_single_encounter",
    "validate_batch_encounters",
    "ClinicalInferenceService",
]
