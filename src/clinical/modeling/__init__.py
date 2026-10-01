"""Clinical Tabular Baseline Modeling Framework Package (Phase C4)."""

from src.clinical.modeling.schema import (
    IDENTIFIERS,
    ZERO_VARIANCE_FEATURES,
    HIGH_MISSINGNESS_DROPS,
    EXPECTED_FEATURE_DIM,
    TARGET_COLUMN,
    TARGET_MAPPING,
    NUMERICAL_FEATURES,
    CATEGORICAL_DEMOGRAPHICS,
    CATEGORICAL_CONTEXT,
    GLYCEMIC_LABS,
    GLYCEMIC_ORDINAL_MAPS,
    AGE_ORDINAL_MAP,
    DIAGNOSIS_COLUMNS,
    ACTIVE_MEDICATIONS,
    MEDICATION_EXPOSURE_MAP,
    TREATMENT_DYNAMICS,
    TREATMENT_DYNAMICS_MAPS,
    ICD9_CHAPTERS,
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
from src.clinical.modeling.evaluate import (
    evaluate_model_on_split,
    compute_stratified_error_analysis,
)
from src.clinical.modeling.train import run_training_pipeline

__all__ = [
    "IDENTIFIERS",
    "ZERO_VARIANCE_FEATURES",
    "HIGH_MISSINGNESS_DROPS",
    "EXPECTED_FEATURE_DIM",
    "TARGET_COLUMN",
    "TARGET_MAPPING",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_DEMOGRAPHICS",
    "CATEGORICAL_CONTEXT",
    "GLYCEMIC_LABS",
    "GLYCEMIC_ORDINAL_MAPS",
    "AGE_ORDINAL_MAP",
    "DIAGNOSIS_COLUMNS",
    "ACTIVE_MEDICATIONS",
    "MEDICATION_EXPOSURE_MAP",
    "TREATMENT_DYNAMICS",
    "TREATMENT_DYNAMICS_MAPS",
    "ICD9_CHAPTERS",
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
    "evaluate_model_on_split",
    "compute_stratified_error_analysis",
    "run_training_pipeline",
]
