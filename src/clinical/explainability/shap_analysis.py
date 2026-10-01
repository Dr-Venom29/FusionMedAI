"""Core SHAP analysis module for frozen CatBoost clinical model."""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clinical.modeling.preprocessing import ClinicalPreprocessor
from src.clinical.benchmarking.runtime import get_runtime_output_root

# Frozen CatBoost HPO Hyperparameter Configuration
FROZEN_CATBOOST_HPO_CONFIG: Dict[str, Any] = {
    "iterations": 350,
    "depth": 4,
    "learning_rate": 0.1383,
    "l2_leaf_reg": 2.911,
    "subsample": 0.655,
    "random_seed": 42,
    "early_stopping_rounds": 30,
    "eval_metric": "Logloss",
    "verbose": 0,
    "thread_count": -1,
}


class ClinicalSHAPExplainer:
    """Manages TreeSHAP computations for the frozen CatBoost clinical model."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        train_dir: Optional[str] = None,
    ):
        self.config = (config or FROZEN_CATBOOST_HPO_CONFIG).copy()
        output_root = get_runtime_output_root(REPO_ROOT, "explainability")
        target_train_dir = train_dir or str(output_root / "catboost_info")
        Path(target_train_dir).mkdir(parents=True, exist_ok=True)
        self.config["train_dir"] = target_train_dir

        self.model = CatBoostClassifier(**self.config)
        self.preprocessor = ClinicalPreprocessor(scale_numerical=True)
        self.is_fitted = False
        self.feature_names = []

    def fit_and_prepare(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> "ClinicalSHAPExplainer":
        """Fit preprocessor and frozen CatBoost model on training data."""
        self.preprocessor.fit(train_df)
        X_train, y_train, _ = self.preprocessor.transform(train_df)
        X_val, y_val, _ = self.preprocessor.transform(val_df)

        self.feature_names = list(self.preprocessor.feature_names_)
        assert len(self.feature_names) == 119, f"Expected 119 features, got {len(self.feature_names)}"

        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            verbose=0,
            use_best_model=True,
        )
        self.is_fitted = True
        return self

    def compute_shap_values(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        """Compute TreeSHAP values for a dataset partition.

        Returns:
            shap_values: Array of shape (N, 119) with feature SHAP contributions.
            base_value: Float baseline log-odds prediction.
            X_transformed: Array of shape (N, 119) with preprocessed features.
            predictions: Array of shape (N,) with predicted probabilities.
            y_true: Array of shape (N,) with true binary targets.
        """
        if not self.is_fitted:
            raise RuntimeError("Explainer must be fitted before computing SHAP values.")

        X, y, _ = self.preprocessor.transform(df)

        pool = Pool(X, y, feature_names=self.feature_names)
        shap_raw = self.model.get_feature_importance(pool, type="ShapValues")
        
        # In CatBoost binary classification, ShapValues is (N, D + 1) with bias in the final column
        shap_values = shap_raw[:, :-1]
        base_value = float(shap_raw[0, -1])
        probs = self.model.predict_proba(X)[:, 1]

        return shap_values, base_value, X, probs, y
