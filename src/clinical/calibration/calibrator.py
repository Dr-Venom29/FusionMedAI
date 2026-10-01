"""
Calibrator implementations for Clinical 30-Day Readmission Risk.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class BaseClinicalCalibrator(ABC):
    """Abstract base class for clinical probability calibrators."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, y_prob_val: np.ndarray, y_true_val: np.ndarray, logits_val: Optional[np.ndarray] = None) -> "BaseClinicalCalibrator":
        """Fit calibration parameters strictly on validation predictions."""
        pass

    @abstractmethod
    def predict_proba(self, y_prob: np.ndarray, logits: Optional[np.ndarray] = None) -> np.ndarray:
        """Transform raw probabilities into calibrated probabilities."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return serialized calibration parameters."""
        pass


class PlattCalibrator(BaseClinicalCalibrator):
    """
    Platt / Logistic Scaling for binary probabilities:
        p_cal = sigma(a * z + b)
    where z is the raw model margin (log-odds).
    """

    def __init__(self):
        super().__init__(name="Platt")
        self.a = 1.0
        self.b = 0.0
        self.model: Optional[LogisticRegression] = None

    def fit(
        self,
        y_prob_val: np.ndarray,
        y_true_val: np.ndarray,
        logits_val: Optional[np.ndarray] = None,
    ) -> "PlattCalibrator":
        y_true = np.asarray(y_true_val).ravel()
        if logits_val is not None:
            z = np.asarray(logits_val).reshape(-1, 1)
        else:
            eps = 1e-12
            p = np.clip(np.asarray(y_prob_val).ravel(), eps, 1.0 - eps)
            z = np.log(p / (1.0 - p)).reshape(-1, 1)

        self.model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        try:
            self.model.fit(z, y_true)
        except Exception:
            self.model = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
            self.model.fit(z, y_true)

        self.a = float(self.model.coef_[0, 0])
        self.b = float(self.model.intercept_[0])
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        y_prob: np.ndarray,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PlattCalibrator is not fitted.")
        if logits is not None:
            z = np.asarray(logits).reshape(-1, 1)
        else:
            eps = 1e-12
            p = np.clip(np.asarray(y_prob).ravel(), eps, 1.0 - eps)
            z = np.log(p / (1.0 - p)).reshape(-1, 1)

        cal_logits = self.a * z.ravel() + self.b
        cal_prob = 1.0 / (1.0 + np.exp(-cal_logits))
        return np.clip(cal_prob, 1e-12, 1.0 - 1e-12)

    def get_params(self) -> Dict[str, Any]:
        return {"method": self.name, "slope_a": self.a, "intercept_b": self.b}


class IsotonicCalibrator(BaseClinicalCalibrator):
    """
    Non-parametric monotonic Isotonic Regression:
        p_cal = g(p_raw)
    """

    def __init__(self):
        super().__init__(name="Isotonic")
        self.iso_reg = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(
        self,
        y_prob_val: np.ndarray,
        y_true_val: np.ndarray,
        logits_val: Optional[np.ndarray] = None,
    ) -> "IsotonicCalibrator":
        y_prob = np.asarray(y_prob_val).ravel()
        y_true = np.asarray(y_true_val).ravel()
        self.iso_reg.fit(y_prob, y_true)
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        y_prob: np.ndarray,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("IsotonicCalibrator is not fitted.")
        p = np.asarray(y_prob).ravel()
        cal_prob = self.iso_reg.predict(p)
        return np.clip(cal_prob, 1e-12, 1.0 - 1e-12)

    def get_params(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "n_thresholds": len(self.iso_reg.X_thresholds_) if hasattr(self.iso_reg, "X_thresholds_") else 0,
        }


class BetaCalibrator(BaseClinicalCalibrator):
    """
    Parametric Beta Calibration (Kull et al., 2017):
        logit(p_cal) = a * ln(p_raw) - b * ln(1 - p_raw) + c
    Fitted via logistic regression on [ln(p), -ln(1 - p)].
    """

    def __init__(self):
        super().__init__(name="Beta")
        self.a = 1.0
        self.b = 1.0
        self.c = 0.0
        self.model: Optional[LogisticRegression] = None

    def fit(
        self,
        y_prob_val: np.ndarray,
        y_true_val: np.ndarray,
        logits_val: Optional[np.ndarray] = None,
    ) -> "BetaCalibrator":
        eps = 1e-12
        p = np.clip(np.asarray(y_prob_val).ravel(), eps, 1.0 - eps)
        y_true = np.asarray(y_true_val).ravel()

        x1 = np.log(p)
        x2 = -np.log(1.0 - p)
        X_feat = np.column_stack([x1, x2])

        self.model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        try:
            self.model.fit(X_feat, y_true)
        except Exception:
            self.model = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
            self.model.fit(X_feat, y_true)

        self.a = float(self.model.coef_[0, 0])
        self.b = float(self.model.coef_[0, 1])
        self.c = float(self.model.intercept_[0])
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        y_prob: np.ndarray,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("BetaCalibrator is not fitted.")
        eps = 1e-12
        p = np.clip(np.asarray(y_prob).ravel(), eps, 1.0 - eps)
        x1 = np.log(p)
        x2 = -np.log(1.0 - p)
        X_feat = np.column_stack([x1, x2])

        cal_logits = self.a * x1 + self.b * x2 + self.c
        cal_prob = 1.0 / (1.0 + np.exp(-cal_logits))
        return np.clip(cal_prob, 1e-12, 1.0 - 1e-12)

    def get_params(self) -> Dict[str, Any]:
        return {"method": self.name, "param_a": self.a, "param_b": self.b, "param_c": self.c}
