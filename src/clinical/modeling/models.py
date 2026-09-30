"""Baseline Tabular Model Definitions and Interface for Clinical Readmission Prediction.

Provides unified interface across all four baseline classifiers:
- fit(X_train, y_train)
- predict_proba(X) -> np.ndarray (N, 2)
- predict(X, threshold=0.5) -> np.ndarray (N,)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


class BaseClinicalModel(ABC):
    """Abstract base class for clinical tabular classification models."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaseClinicalModel":
        """Fit the model on training feature matrix and target vector."""
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities [P(y=0), P(y=1)]."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"Model '{self.model_name}' must be fitted before predict_proba.")
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels based on a probability threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def get_config(self) -> Dict[str, Any]:
        """Return the hyperparameter configuration dictionary."""
        return {
            "model_name": self.model_name,
            "config": self.config,
            "is_fitted": self.is_fitted,
        }


class LogisticRegressionModel(BaseClinicalModel):
    """Regularized Logistic Regression baseline (L2 / ElasticNet)."""

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        l1_ratio: Optional[float] = None,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        config = {
            "penalty": penalty,
            "C": C,
            "l1_ratio": l1_ratio,
            "solver": solver,
            "max_iter": max_iter,
            "random_state": random_state,
        }
        super().__init__(model_name=f"logistic_regression_{penalty}", config=config)
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            l1_ratio=l1_ratio,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegressionModel":
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self


class RandomForestModel(BaseClinicalModel):
    """Random Forest Classifier baseline."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 12,
        min_samples_leaf: int = 20,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        config = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        super().__init__(model_name="random_forest", config=config)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestModel":
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self


class XGBoostModel(BaseClinicalModel):
    """XGBoost Gradient Boosted Decision Tree baseline."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        config = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        super().__init__(model_name="xgboost", config=config)
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostModel":
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self


class LightGBMModel(BaseClinicalModel):
    """LightGBM Gradient Boosted Decision Tree baseline."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        config = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": n_jobs,
        }
        super().__init__(model_name="lightgbm", config=config)
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbose=-1,
            n_jobs=n_jobs,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LightGBMModel":
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self


def get_baseline_models(random_state: int = 42) -> Dict[str, BaseClinicalModel]:
    """Instantiate the 4 primary baseline models for Phase C4 benchmarking."""
    return {
        "logistic_regression": LogisticRegressionModel(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=random_state
        ),
        "logistic_regression_elasticnet": LogisticRegressionModel(
            penalty="elasticnet", C=1.0, l1_ratio=0.5, solver="saga", max_iter=500, random_state=random_state
        ),
        "random_forest": RandomForestModel(
            n_estimators=100, max_depth=12, min_samples_leaf=20, random_state=random_state
        ),
        "xgboost": XGBoostModel(
            n_estimators=100, max_depth=5, learning_rate=0.05, random_state=random_state
        ),
        "lightgbm": LightGBMModel(
            n_estimators=100, max_depth=5, num_leaves=31, learning_rate=0.05, random_state=random_state
        ),
    }
