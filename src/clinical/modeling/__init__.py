"""Clinical Baseline Modeling and Evaluation Module for FusionMedAI."""

from src.clinical.modeling.schema import (
    IDENTIFIERS,
    ZERO_VARIANCE_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_DEMOGRAPHICS,
    CATEGORICAL_CONTEXT,
    GLYCEMIC_LABS,
    DIAGNOSIS_COLUMNS,
    ACTIVE_MEDICATIONS,
    TREATMENT_DYNAMICS,
    TARGET_COLUMN,
    map_icd9_to_chapter,
)
from src.clinical.modeling.preprocessing import ClinicalPreprocessor
from src.clinical.modeling.models import (
    BaseClinicalModel,
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    get_baseline_models,
)
from src.clinical.modeling.metrics import (
    compute_classification_metrics,
    compute_threshold_metrics,
    compute_calibration_metrics,
)

__all__ = [
    "IDENTIFIERS",
    "ZERO_VARIANCE_FEATURES",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_DEMOGRAPHICS",
    "CATEGORICAL_CONTEXT",
    "GLYCEMIC_LABS",
    "DIAGNOSIS_COLUMNS",
    "ACTIVE_MEDICATIONS",
    "TREATMENT_DYNAMICS",
    "TARGET_COLUMN",
    "map_icd9_to_chapter",
    "ClinicalPreprocessor",
    "BaseClinicalModel",
    "LogisticRegressionModel",
    "RandomForestModel",
    "XGBoostModel",
    "LightGBMModel",
    "get_baseline_models",
    "compute_classification_metrics",
    "compute_threshold_metrics",
    "compute_calibration_metrics",
]
