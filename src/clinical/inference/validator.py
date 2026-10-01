"""
Clinical Input Validation and Sanitization Suite (Phase C10).
Enforces schema validity and triggers controlled validation exceptions for malformed inputs.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from src.clinical.modeling.preprocessing import (
    ACTIVE_MEDICATIONS,
    GLYCEMIC_LABS,
    TREATMENT_DYNAMICS,
)


class ClinicalValidationError(ValueError):
    """Exception raised when clinical input fails schema or physiological validation."""
    pass


REQUIRED_CLINICAL_COLUMNS = [
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
]

NON_NEGATIVE_NUMERICAL_COLS = [
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

POSITIVE_NUMERICAL_COLS = [
    "time_in_hospital",
]


def validate_single_encounter(raw_input: Union[Dict[str, Any], pd.Series]) -> Dict[str, Any]:
    """
    Validate a single clinical encounter dictionary or pandas Series.
    
    Raises:
        ClinicalValidationError: If required fields are missing or physiological bounds are violated.
    Returns:
        Sanitized copy of the input dictionary with complete clinical feature schema.
    """
    if raw_input is None:
        raise ClinicalValidationError("Input encounter payload cannot be None.")

    if isinstance(raw_input, pd.Series):
        input_dict = raw_input.to_dict()
    elif isinstance(raw_input, dict):
        input_dict = dict(raw_input)
    else:
        raise ClinicalValidationError(f"Expected dict or pd.Series, received {type(raw_input)}.")

    if not input_dict:
        raise ClinicalValidationError("Input encounter payload is empty.")

    # 1. Required columns presence
    missing_cols = [col for col in REQUIRED_CLINICAL_COLUMNS if col not in input_dict]
    if missing_cols:
        raise ClinicalValidationError(f"Missing required clinical columns: {missing_cols}")

    # 2. Check non-negative constraints
    for col in NON_NEGATIVE_NUMERICAL_COLS:
        val = input_dict.get(col)
        if val is not None and not (isinstance(val, str) and val in ("?", "", "None", "nan")):
            try:
                num_val = float(val)
                if np.isnan(num_val):
                    continue
                if num_val < 0:
                    raise ClinicalValidationError(f"Column '{col}' must be non-negative, received {val}.")
            except (ValueError, TypeError):
                raise ClinicalValidationError(f"Column '{col}' expected numeric, received {val}.")

    # 3. Check positive constraints
    for col in POSITIVE_NUMERICAL_COLS:
        val = input_dict.get(col)
        if val is not None and not (isinstance(val, str) and val in ("?", "", "None", "nan")):
            try:
                num_val = float(val)
                if np.isnan(num_val):
                    continue
                if num_val <= 0:
                    raise ClinicalValidationError(f"Column '{col}' must be positive (> 0), received {val}.")
            except (ValueError, TypeError):
                raise ClinicalValidationError(f"Column '{col}' expected numeric, received {val}.")

    # 4. Check age group format if string
    age_val = input_dict.get("age")
    valid_age_patterns = [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ]
    if isinstance(age_val, str) and age_val not in valid_age_patterns and age_val not in ("?", "None", "nan"):
        try:
            num_age = float(age_val)
            if num_age < 0 or num_age > 120:
                raise ClinicalValidationError(f"Invalid physiological age: {age_val}.")
        except ValueError:
            raise ClinicalValidationError(f"Invalid age category format: '{age_val}'.")

    # 5. Check gender category
    gender_val = input_dict.get("gender")
    if gender_val is not None and str(gender_val) not in ("Female", "Male", "Unknown/Invalid", "?", "None", "nan"):
        raise ClinicalValidationError(f"Unrecognized gender category: '{gender_val}'.")

    # 6. Populate default clinical categories for unspecified optional fields
    for med in ACTIVE_MEDICATIONS:
        if med not in input_dict:
            input_dict[med] = "No"

    for lab in GLYCEMIC_LABS:
        if lab not in input_dict:
            input_dict[lab] = "None"

    for dyn in TREATMENT_DYNAMICS:
        if dyn not in input_dict:
            input_dict[dyn] = "No"

    if "race" not in input_dict:
        input_dict["race"] = "?"
    if "payer_code" not in input_dict:
        input_dict["payer_code"] = "?"
    if "medical_specialty" not in input_dict:
        input_dict["medical_specialty"] = "?"

    return input_dict


def validate_batch_encounters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a batch dataframe of encounters and populate missing optional columns.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ClinicalValidationError(f"Expected pandas DataFrame, received {type(df)}.")

    if len(df) == 0:
        raise ClinicalValidationError("Batch DataFrame is empty.")

    missing_cols = [col for col in REQUIRED_CLINICAL_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ClinicalValidationError(f"Batch DataFrame missing required columns: {missing_cols}")

    df_clean = df.copy()

    for med in ACTIVE_MEDICATIONS:
        if med not in df_clean.columns:
            df_clean[med] = "No"

    for lab in GLYCEMIC_LABS:
        if lab not in df_clean.columns:
            df_clean[lab] = "None"

    for dyn in TREATMENT_DYNAMICS:
        if dyn not in df_clean.columns:
            df_clean[dyn] = "No"

    if "race" not in df_clean.columns:
        df_clean["race"] = "?"
    if "payer_code" not in df_clean.columns:
        df_clean["payer_code"] = "?"
    if "medical_specialty" not in df_clean.columns:
        df_clean["medical_specialty"] = "?"

    return df_clean
