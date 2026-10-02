"""
FusionMedAI - Phase C11.4: Router Hyperparameter Coefficients
Defines non-negative hyperparameter configurations Theta = (alpha, beta, gamma, eta) bounded in [0.0, 5.0].
"""

from dataclasses import dataclass
from typing import Dict, Any
import math


class RouterCoefficientError(ValueError):
    """Raised when router coefficient bounds or types are violated."""
    pass


@dataclass(frozen=True)
class RouterCoefficients:
    """
    Hyperparameter vector Theta = (alpha, beta, gamma, eta) for ACARA-U v2 Router logit:
        z_i = alpha * C_i + beta * R_i - gamma * U_i + eta * Q_i
        
    Domain constraints (Protocol Phase C11.0):
        alpha in [0.0, 5.0] (Confidence weight)
        beta  in [0.0, 5.0] (Empirical validation reliability weight)
        gamma in [0.0, 5.0] (Predictive uncertainty penalty)
        eta   in [0.0, 5.0] (Input signal quality bonus)
    """
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    eta: float = 1.0

    def __post_init__(self) -> None:
        for name, val in [
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("gamma", self.gamma),
            ("eta", self.eta),
        ]:
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise RouterCoefficientError(
                    f"Coefficient '{name}' must be a finite float, got {val}."
                )
            if val < 0.0 - 1e-7 or val > 5.0 + 1e-7:
                raise RouterCoefficientError(
                    f"Coefficient '{name}' = {val:.6f} out of protocol bounds [0.0, 5.0]."
                )

    def to_dict(self) -> Dict[str, float]:
        """Returns coefficient dictionary."""
        return {
            "alpha": round(float(self.alpha), 6),
            "beta": round(float(self.beta), 6),
            "gamma": round(float(self.gamma), 6),
            "eta": round(float(self.eta), 6),
        }

    # =========================================================================
    # Standard Experimental Ladder Baseline Presets (B2 - B6)
    # =========================================================================
    
    @classmethod
    def uniform_baseline(cls) -> "RouterCoefficients":
        """Baseline B2: Uniform Average (alpha=0, beta=0, gamma=0, eta=0)."""
        return cls(alpha=0.0, beta=0.0, gamma=0.0, eta=0.0)

    @classmethod
    def confidence_only(cls, alpha: float = 1.0) -> "RouterCoefficients":
        """Baseline B3: Confidence Fusion (alpha>0, beta=0, gamma=0, eta=0)."""
        return cls(alpha=alpha, beta=0.0, gamma=0.0, eta=0.0)

    @classmethod
    def confidence_reliability(cls, alpha: float = 1.0, beta: float = 1.0) -> "RouterCoefficients":
        """Baseline B4: Confidence + Reliability (alpha>0, beta>0, gamma=0, eta=0)."""
        return cls(alpha=alpha, beta=beta, gamma=0.0, eta=0.0)

    @classmethod
    def uncertainty_ablation(cls, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0) -> "RouterCoefficients":
        """Baseline B5: Confidence + Reliability - Uncertainty (alpha>0, beta>0, gamma>0, eta=0)."""
        return cls(alpha=alpha, beta=beta, gamma=gamma, eta=0.0)

    @classmethod
    def full_acarau(cls, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0, eta: float = 1.0) -> "RouterCoefficients":
        """Baseline B6: Full ACARA-U (alpha>0, beta>0, gamma>0, eta>0)."""
        return cls(alpha=alpha, beta=beta, gamma=gamma, eta=eta)
