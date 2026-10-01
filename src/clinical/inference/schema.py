"""
Standardized Schema Definitions for Clinical Tabular Modality (Phase C10).
Conforms strictly to the Multimodal ClinicalOutput Contract.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class FeatureAttribution:
    feature: str
    shap_value: float
    rank: int
    feature_value: Optional[Any] = None


@dataclass
class PredictiveInterval:
    lower: float
    upper: float


@dataclass
class UncertaintyOutput:
    method: str
    std_probability: float
    percentile_in_cohort: float
    is_high_uncertainty: bool
    predictive_interval_95: PredictiveInterval
    aleatoric_entropy: float


@dataclass
class ShiftDetection:
    is_degraded: bool
    missingness_ratio: float
    blind_spot_warning: bool
    shift_alerts: List[str] = field(default_factory=list)


@dataclass
class ModelProvenance:
    model_name: str
    ensemble_size: int
    frozen_calibrator: str
    version: str
    manifest_sha256: str


@dataclass
class ClinicalOutput:
    modality: str
    encounter_id: Any
    prediction: int
    probability: float
    calibrated_probability: float
    calibration_method: str
    confidence: str
    uncertainty: UncertaintyOutput
    decision_tier: str
    operating_threshold: float
    feature_attributions: List[FeatureAttribution]
    shift_detection: ShiftDetection
    model_provenance: ModelProvenance

    def to_dict(self) -> Dict[str, Any]:
        """Convert structured ClinicalOutput object to standard JSON dictionary."""
        return asdict(self)
