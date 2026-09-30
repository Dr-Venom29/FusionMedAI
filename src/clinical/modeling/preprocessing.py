"""Clinical Preprocessing Pipeline.

Strictly adheres to the train-only estimator fitting contract:
- fit() computes and locks all parameters exclusively on train.csv.
- transform() processes validation and test sets using the locked parameters.
- Zero fitting on validation or test sets.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
    map_icd9_to_chapter,
)


class ClinicalPreprocessor:
    """End-to-end clinical tabular preprocessor implementing the frozen C3 contract."""

    def __init__(self, scale_numerical: bool = True):
        self.scale_numerical = scale_numerical
        self.is_fitted: bool = False
        
        # Fitted parameters locked on training partition
        self.top_specialties_: List[str] = []
        self.top_payers_: List[str] = []
        self.scaler_: Optional[StandardScaler] = None
        self.ohe_demographics_: Optional[OneHotEncoder] = None
        self.ohe_context_: Optional[OneHotEncoder] = None
        self.ohe_diagnoses_: Optional[OneHotEncoder] = None
        
        # Metadata
        self.feature_names_: List[str] = []
        self.feature_dim_: int = 0
        self.train_provenance_: Dict[str, Any] = {}

    def _clean_categorical_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize raw categorical and context columns before encoding."""
        df_clean = df.copy()

        # Race: map '?' to 'Unknown'
        df_clean["race"] = df_clean["race"].fillna("Unknown").replace("?", "Unknown")

        # Gender: map 'Unknown/Invalid' to 'Unknown'
        df_clean["gender"] = df_clean["gender"].fillna("Unknown").replace("Unknown/Invalid", "Unknown")

        # Age: ordinal mapping
        df_clean["age_ordinal"] = df_clean["age"].map(AGE_ORDINAL_MAP).fillna(5).astype(int)

        # Admission context missing codes
        df_clean["admission_type_id"] = df_clean["admission_type_id"].astype(str).replace(
            {"5": "Missing", "6": "Missing", "8": "Missing"}
        )
        df_clean["admission_source_id"] = df_clean["admission_source_id"].astype(str).replace(
            {"9": "Missing", "17": "Missing", "20": "Missing"}
        )

        # Medical specialty grouping: Top-10 + Other + Missing
        spec = df_clean["medical_specialty"].fillna("Missing").replace("?", "Missing")
        if self.top_specialties_:
            df_clean["medical_specialty_grouped"] = spec.apply(
                lambda s: s if s in self.top_specialties_ or s == "Missing" else "Other"
            )
        else:
            df_clean["medical_specialty_grouped"] = spec

        # Payer code grouping: Top-8 + Other + Missing
        payer = df_clean["payer_code"].fillna("Missing").replace("?", "Missing")
        if self.top_payers_:
            df_clean["payer_code_grouped"] = payer.apply(
                lambda p: p if p in self.top_payers_ or p == "Missing" else "Other"
            )
        else:
            df_clean["payer_code_grouped"] = payer

        # Glycemic labs ordinal mapping (preserving 'None' as 0)
        for lab in GLYCEMIC_LABS:
            mapping = GLYCEMIC_ORDINAL_MAPS[lab]
            df_clean[f"{lab}_ordinal"] = df_clean[lab].fillna("None").map(mapping).fillna(0).astype(int)

        # ICD-9 Diagnosis mapping to chapters
        for diag_col in DIAGNOSIS_COLUMNS:
            df_clean[f"{diag_col}_chapter"] = df_clean[diag_col].apply(map_icd9_to_chapter)

        # 21 Active medications 4-level ordinal mapping (0, 1, 2, 3)
        for med in ACTIVE_MEDICATIONS:
            df_clean[f"{med}_exposure"] = (
                df_clean[med].fillna("No").map(MEDICATION_EXPOSURE_MAP).fillna(0).astype(int)
            )

        # Treatment dynamics binary flags (0 / 1)
        for dyn in TREATMENT_DYNAMICS:
            mapping = TREATMENT_DYNAMICS_MAPS[dyn]
            df_clean[f"{dyn}_binary"] = df_clean[dyn].fillna("No").map(mapping).fillna(0).astype(int)

        return df_clean

    def fit(self, df_train: pd.DataFrame) -> "ClinicalPreprocessor":
        """Fit all preprocessing estimators strictly on the training partition."""
        # 1. Fit Top-10 specialties (excluding '?' and missing)
        spec_counts = df_train["medical_specialty"].replace("?", np.nan).dropna().value_counts()
        self.top_specialties_ = list(spec_counts.head(10).index)

        # 2. Fit Top-8 payer codes (excluding '?' and missing)
        payer_counts = df_train["payer_code"].replace("?", np.nan).dropna().value_counts()
        self.top_payers_ = list(payer_counts.head(8).index)

        # Clean training data
        df_clean = self._clean_categorical_inputs(df_train)

        # 3. Fit Numerical Scaler
        if self.scale_numerical:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(df_clean[NUMERICAL_FEATURES])

        # 4. Fit Demographics OneHotEncoder
        demog_cols = ["race", "gender"]
        self.ohe_demographics_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.ohe_demographics_.fit(df_clean[demog_cols])

        # 5. Fit Context OneHotEncoder
        context_cols = [
            "admission_type_id",
            "admission_source_id",
            "medical_specialty_grouped",
            "payer_code_grouped",
        ]
        self.ohe_context_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.ohe_context_.fit(df_clean[context_cols])

        # 6. Fit ICD-9 Diagnoses OneHotEncoder
        diag_chapter_cols = [f"{col}_chapter" for col in DIAGNOSIS_COLUMNS]
        self.ohe_diagnoses_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.ohe_diagnoses_.fit(df_clean[diag_chapter_cols])

        # Build feature names
        feature_names = []
        # Numerical
        feature_names.extend(NUMERICAL_FEATURES)
        # Age ordinal
        feature_names.append("age_ordinal")
        # Demographics OHE
        for cat, feats in zip(demog_cols, self.ohe_demographics_.categories_):
            for val in feats:
                feature_names.append(f"{cat}_{val}")
        # Context OHE
        for cat, feats in zip(context_cols, self.ohe_context_.categories_):
            for val in feats:
                feature_names.append(f"{cat}_{val}")
        # Glycemic labs ordinal
        for lab in GLYCEMIC_LABS:
            feature_names.append(f"{lab}_ordinal")
        # Diagnoses OHE
        for cat, feats in zip(diag_chapter_cols, self.ohe_diagnoses_.categories_):
            for val in feats:
                feature_names.append(f"{cat}_{val}")
        # Medication ordinal exposures
        for med in ACTIVE_MEDICATIONS:
            feature_names.append(f"{med}_exposure")
        # Treatment dynamics
        for dyn in TREATMENT_DYNAMICS:
            feature_names.append(f"{dyn}_binary")

        self.feature_names_ = feature_names
        self.feature_dim_ = len(feature_names)
        assert (
            self.feature_dim_ == EXPECTED_FEATURE_DIM
        ), f"Feature dimension mismatch: expected {EXPECTED_FEATURE_DIM}, got {self.feature_dim_}"
        self.train_provenance_ = {
            "n_samples": len(df_train),
            "n_features": self.feature_dim_,
            "feature_names": self.feature_names_,
            "scale_numerical": self.scale_numerical,
            "top_specialties": self.top_specialties_,
            "top_payers": self.top_payers_,
        }
        self.is_fitted = True
        return self

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
        """Transform an encounter split using the locked training parameters.
        
        Returns:
            X: np.ndarray feature matrix (N, D)
            y: Optional[np.ndarray] target vector (N,) if target column is present
            trace_df: pd.DataFrame containing identifiers (encounter_id, patient_nbr)
        """
        if not self.is_fitted:
            raise RuntimeError("ClinicalPreprocessor must be fitted on train data before transform.")

        # Extract trace identifiers
        trace_cols = [col for col in IDENTIFIERS if col in df.columns]
        trace_df = df[trace_cols].copy()

        # Extract target if present
        y = None
        if TARGET_COLUMN in df.columns:
            y = df[TARGET_COLUMN].map(TARGET_MAPPING).fillna(0).astype(int).values

        # Clean inputs
        df_clean = self._clean_categorical_inputs(df)

        # 1. Numerical block
        if self.scale_numerical and self.scaler_ is not None:
            num_mat = self.scaler_.transform(df_clean[NUMERICAL_FEATURES])
        else:
            num_mat = df_clean[NUMERICAL_FEATURES].values

        # 2. Age ordinal
        age_mat = df_clean[["age_ordinal"]].values

        # 3. Demographics OHE
        demog_mat = self.ohe_demographics_.transform(df_clean[["race", "gender"]])

        # 4. Context OHE
        context_cols = [
            "admission_type_id",
            "admission_source_id",
            "medical_specialty_grouped",
            "payer_code_grouped",
        ]
        context_mat = self.ohe_context_.transform(df_clean[context_cols])

        # 5. Glycemic Labs Ordinal
        lab_cols = [f"{lab}_ordinal" for lab in GLYCEMIC_LABS]
        lab_mat = df_clean[lab_cols].values

        # 6. Diagnoses OHE
        diag_chapter_cols = [f"{col}_chapter" for col in DIAGNOSIS_COLUMNS]
        diag_mat = self.ohe_diagnoses_.transform(df_clean[diag_chapter_cols])

        # 7. Medications Ordinal
        med_cols = [f"{med}_exposure" for med in ACTIVE_MEDICATIONS]
        med_mat = df_clean[med_cols].values

        # 8. Treatment Dynamics
        dyn_cols = [f"{dyn}_binary" for dyn in TREATMENT_DYNAMICS]
        dyn_mat = df_clean[dyn_cols].values

        # Concatenate all blocks horizontally
        X = np.hstack([
            num_mat,
            age_mat,
            demog_mat,
            context_mat,
            lab_mat,
            diag_mat,
            med_mat,
            dyn_mat,
        ])

        assert (
            X.shape[1] == EXPECTED_FEATURE_DIM
        ), f"Transformed feature dimension mismatch: expected {EXPECTED_FEATURE_DIM}, got {X.shape[1]}"

        return X, y, trace_df
