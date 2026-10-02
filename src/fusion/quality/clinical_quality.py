"""
Structured Clinical EHR Quality & Availability Engine (Phase C11.2).
Evaluates clinical tabular record completeness strictly against the canonical 119-D feature contract.
Zero model predictions or readmission ground-truth targets used.
"""

from typing import Union, Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd

from src.fusion.quality.quality_result import QualityResult, make_unavailable_quality
from src.clinical.modeling.schema import EXPECTED_FEATURE_DIM
from src.clinical.modeling.preprocessing import ClinicalPreprocessor

CLINICAL_EXPECTED_DIM: int = EXPECTED_FEATURE_DIM  # 119

_FROZEN_CLINICAL_PREPROCESSOR: Optional[ClinicalPreprocessor] = None


def get_frozen_clinical_preprocessor() -> Optional[ClinicalPreprocessor]:
    """
    Lazily loads and fits the ClinicalPreprocessor exclusively on the training split.
    """
    global _FROZEN_CLINICAL_PREPROCESSOR
    if _FROZEN_CLINICAL_PREPROCESSOR is None:
        train_path = (
            Path(__file__).resolve().parents[3]
            / "datasets"
            / "clinical"
            / "processed"
            / "splits"
            / "train.csv"
        )
        if train_path.exists():
            try:
                train_df = pd.read_csv(train_path)
                prep = ClinicalPreprocessor(scale_numerical=True)
                prep.fit(train_df)
                _FROZEN_CLINICAL_PREPROCESSOR = prep
            except Exception:
                _FROZEN_CLINICAL_PREPROCESSOR = None
    return _FROZEN_CLINICAL_PREPROCESSOR


def compute_clinical_quality(
    clinical_input: Optional[Union[Dict[str, Any], pd.Series, pd.DataFrame, np.ndarray, List[float]]]
) -> QualityResult:
    """
    Evaluates input availability and feature completeness for a structured EHR record.
    Both 119-D feature vectors and raw encounters are evaluated strictly against the
    canonical 119-D representation space: Q_C = N_valid / 119.
    
    Args:
        clinical_input: 119-D feature vector (array/list), raw encounter dictionary, Series, or 1-row DataFrame.
        
    Returns:
        QualityResult with availability: bool, quality: float in [0.0, 1.0], and diagnostics.
    """
    if clinical_input is None:
        return make_unavailable_quality("INPUT_MISSING")

    try:
        # Case 1: Already processed 119-D feature vector (numpy array or list)
        if isinstance(clinical_input, (np.ndarray, list)):
            arr = np.asarray(clinical_input, dtype=float).squeeze()
            if arr.ndim != 1 or len(arr) != CLINICAL_EXPECTED_DIM:
                return make_unavailable_quality(
                    f"INVALID_FEATURE_DIMENSION_EXPECTED_{CLINICAL_EXPECTED_DIM}_GOT_{len(arr) if arr.ndim == 1 else arr.shape}"
                )
            
            # Count valid (finite, not NaN, not Inf) features
            valid_mask = np.isfinite(arr)
            n_valid = int(np.sum(valid_mask))
            n_total = CLINICAL_EXPECTED_DIM
            q_clinical = float(np.clip(n_valid / n_total, 0.0, 1.0))
            
            diagnostics = {
                "status": "AVAILABLE",
                "representation": "119D_VECTOR",
                "valid_features": n_valid,
                "total_features": n_total,
                "missing_features": n_total - n_valid,
                "completeness_ratio": round(q_clinical, 4),
            }
            return QualityResult(
                availability=True,
                quality=q_clinical,
                diagnostics=diagnostics
            )

        # Case 2: Raw dictionary, Series, or DataFrame -> transform to 119-D representation
        elif isinstance(clinical_input, (dict, pd.Series, pd.DataFrame)):
            if isinstance(clinical_input, pd.DataFrame):
                if len(clinical_input) == 0:
                    return make_unavailable_quality("EMPTY_DATAFRAME")
                raw_df = clinical_input.head(1).copy()
            elif isinstance(clinical_input, pd.Series):
                if clinical_input.empty:
                    return make_unavailable_quality("EMPTY_SERIES")
                raw_df = pd.DataFrame([clinical_input.to_dict()])
            else:
                if not clinical_input:
                    return make_unavailable_quality("EMPTY_DICTIONARY")
                raw_df = pd.DataFrame([clinical_input])

            preprocessor = get_frozen_clinical_preprocessor()
            if preprocessor is None or not preprocessor.is_fitted:
                return make_unavailable_quality("PREPROCESSOR_NOT_AVAILABLE")

            # Transform encounter through frozen preprocessor
            try:
                X_enc, _, _ = preprocessor.transform(raw_df)
            except Exception as ex:
                return make_unavailable_quality(f"PREPROCESSING_TRANSFORM_FAILED_{type(ex).__name__}")

            if X_enc.shape[1] != CLINICAL_EXPECTED_DIM:
                return make_unavailable_quality(
                    f"TRANSFORM_DIMENSION_MISMATCH_EXPECTED_{CLINICAL_EXPECTED_DIM}_GOT_{X_enc.shape[1]}"
                )

            arr_119 = X_enc[0]
            valid_mask = np.isfinite(arr_119)
            n_valid = int(np.sum(valid_mask))
            n_total = CLINICAL_EXPECTED_DIM
            q_clinical = float(np.clip(n_valid / n_total, 0.0, 1.0))

            diagnostics = {
                "status": "AVAILABLE",
                "representation": "TRANSFORMED_RAW_ENCOUNTER",
                "valid_features": n_valid,
                "total_features": n_total,
                "missing_features": n_total - n_valid,
                "completeness_ratio": round(q_clinical, 4),
            }
            return QualityResult(
                availability=True,
                quality=q_clinical,
                diagnostics=diagnostics
            )

        else:
            return make_unavailable_quality(f"UNSUPPORTED_INPUT_TYPE_{type(clinical_input).__name__}")

    except Exception as e:
        return make_unavailable_quality(f"QUALITY_COMPUTATION_ERROR_{type(e).__name__}")
