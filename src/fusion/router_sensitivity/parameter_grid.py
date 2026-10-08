"""
FusionMedAI - Phase C11.12: Sensitivity Parameter Grid & Perturbation Taxonomy
Defines the 20 pre-specified coefficient configurations for One-Factor-At-A-Time (OFAT)
and combined factorial sensitivity sweeps around frozen reference Theta_0 = (1.0, 1.5, 1.0, 0.5).
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import math

from src.fusion.router.coefficients import RouterCoefficients
from .sensitivity_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    COEFFICIENT_BOUNDS,
)


class ParameterGridError(ValueError):
    """Raised when parameter grid definitions violate bounds or protocol schemas."""
    pass


@dataclass(frozen=True)
class SensitivityConfigItem:
    """
    Immutable representation of a single coefficient configuration in the sensitivity grid.
    
    Attributes:
        config_id: Identifier (e.g. 'A1', 'B2', 'G3', 'Q4', 'L', 'M', 'H').
        name: Human-readable name.
        alpha: Confidence coefficient >= 0.0.
        beta: Reliability coefficient >= 0.0.
        gamma: Uncertainty coefficient >= 0.0.
        eta: Quality coefficient >= 0.0.
        sweep_type: Category ('alpha_sweep', 'beta_sweep', 'gamma_sweep', 'eta_sweep', 'combined_sweep').
        description: Brief description of perturbation rationale.
        is_reference: Boolean indicating if this matches Theta_0.
    """
    config_id: str
    name: str
    alpha: float
    beta: float
    gamma: float
    eta: float
    sweep_type: str
    description: str
    is_reference: bool = False

    def __post_init__(self) -> None:
        for name, val in [
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("gamma", self.gamma),
            ("eta", self.eta),
        ]:
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ParameterGridError(f"Config '{self.config_id}' {name} must be a finite float, got {val}")
            min_b, max_b = COEFFICIENT_BOUNDS[name]
            if val < min_b - 1e-7 or val > max_b + 1e-7:
                raise ParameterGridError(
                    f"Config '{self.config_id}' {name}={val} exceeds protocol domain [{min_b}, {max_b}]"
                )

        if self.sweep_type not in (
            "alpha_sweep",
            "beta_sweep",
            "gamma_sweep",
            "eta_sweep",
            "combined_sweep",
            "reference",
        ):
            raise ParameterGridError(f"Invalid sweep_type '{self.sweep_type}' for config '{self.config_id}'.")

    def to_coefficients(self) -> RouterCoefficients:
        """Converts to frozen RouterCoefficients object."""
        return RouterCoefficients(
            alpha=float(self.alpha),
            beta=float(self.beta),
            gamma=float(self.gamma),
            eta=float(self.eta),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes to JSON-compatible dictionary."""
        return {
            "config_id": self.config_id,
            "name": self.name,
            "alpha": round(float(self.alpha), 6),
            "beta": round(float(self.beta), 6),
            "gamma": round(float(self.gamma), 6),
            "eta": round(float(self.eta), 6),
            "sweep_type": self.sweep_type,
            "description": self.description,
            "is_reference": bool(self.is_reference),
        }


def get_reference_config() -> SensitivityConfigItem:
    """Returns the frozen reference configuration Theta_0 = (1.0, 1.5, 1.0, 0.5)."""
    return SensitivityConfigItem(
        config_id="REF",
        name="Reference Configuration (Theta_0)",
        alpha=ALPHA_REF,
        beta=BETA_REF,
        gamma=GAMMA_REF,
        eta=ETA_REF,
        sweep_type="reference",
        description="Frozen ACARA-U Reference Configuration Theta_0 = (1.0, 1.5, 1.0, 0.5)",
        is_reference=True,
    )


def get_sensitivity_parameter_grid() -> List[SensitivityConfigItem]:
    """
    Constructs the pre-specified 23-evaluation sensitivity grid:
    - 5 Alpha sweep configurations (A1–A5)
    - 5 Beta sweep configurations (B1–B5)
    - 5 Gamma sweep configurations (G1–G5)
    - 5 Eta sweep configurations (Q1–Q5)
    - 3 Combined perturbation levels (L, M, H)
    
    Structure:
        Total named evaluations: 23 configuration IDs
        Unique parameter tuples: 19 distinct (alpha, beta, gamma, eta) vectors
        Shared reference points: A3 = B3 = G3 = Q3 = M = Theta_0 = (1.0, 1.5, 1.0, 0.5)
    """
    grid: List[SensitivityConfigItem] = []

    # 1. Alpha Sweep (Confidence: alpha in {0.50, 0.75, 1.00, 1.25, 1.50}, beta=1.5, gamma=1.0, eta=0.5)
    grid.extend([
        SensitivityConfigItem("A1", "alpha=0.50", 0.50, 1.50, 1.00, 0.50, "alpha_sweep", "Low confidence weighting (-50%)", False),
        SensitivityConfigItem("A2", "alpha=0.75", 0.75, 1.50, 1.00, 0.50, "alpha_sweep", "Moderate-low confidence weighting (-25%)", False),
        SensitivityConfigItem("A3", "alpha=1.00 (Ref)", 1.00, 1.50, 1.00, 0.50, "alpha_sweep", "Reference confidence weighting (0%)", True),
        SensitivityConfigItem("A4", "alpha=1.25", 1.25, 1.50, 1.00, 0.50, "alpha_sweep", "Moderate-high confidence weighting (+25%)", False),
        SensitivityConfigItem("A5", "alpha=1.50", 1.50, 1.50, 1.00, 0.50, "alpha_sweep", "High confidence weighting (+50%)", False),
    ])

    # 2. Beta Sweep (Reliability: beta in {1.00, 1.25, 1.50, 1.75, 2.00}, alpha=1.0, gamma=1.0, eta=0.5)
    grid.extend([
        SensitivityConfigItem("B1", "beta=1.00", 1.00, 1.00, 1.00, 0.50, "beta_sweep", "Low reliability weighting (-33%)", False),
        SensitivityConfigItem("B2", "beta=1.25", 1.00, 1.25, 1.00, 0.50, "beta_sweep", "Moderate-low reliability weighting (-17%)", False),
        SensitivityConfigItem("B3", "beta=1.50 (Ref)", 1.00, 1.50, 1.00, 0.50, "beta_sweep", "Reference reliability weighting (0%)", True),
        SensitivityConfigItem("B4", "beta=1.75", 1.00, 1.75, 1.00, 0.50, "beta_sweep", "Moderate-high reliability weighting (+17%)", False),
        SensitivityConfigItem("B5", "beta=2.00", 1.00, 2.00, 1.00, 0.50, "beta_sweep", "High reliability weighting (+33%)", False),
    ])

    # 3. Gamma Sweep (Uncertainty: gamma in {0.50, 0.75, 1.00, 1.25, 1.50}, alpha=1.0, beta=1.5, eta=0.5)
    grid.extend([
        SensitivityConfigItem("G1", "gamma=0.50", 1.00, 1.50, 0.50, 0.50, "gamma_sweep", "Low uncertainty penalty (-50%)", False),
        SensitivityConfigItem("G2", "gamma=0.75", 1.00, 1.50, 0.75, 0.50, "gamma_sweep", "Moderate-low uncertainty penalty (-25%)", False),
        SensitivityConfigItem("G3", "gamma=1.00 (Ref)", 1.00, 1.50, 1.00, 0.50, "gamma_sweep", "Reference uncertainty penalty (0%)", True),
        SensitivityConfigItem("G4", "gamma=1.25", 1.00, 1.50, 1.25, 0.50, "gamma_sweep", "Moderate-high uncertainty penalty (+25%)", False),
        SensitivityConfigItem("G5", "gamma=1.50", 1.00, 1.50, 1.50, 0.50, "gamma_sweep", "High uncertainty penalty (+50%)", False),
    ])

    # 4. Eta Sweep (Quality: eta in {0.25, 0.375, 0.50, 0.625, 0.75}, alpha=1.0, beta=1.5, gamma=1.0)
    grid.extend([
        SensitivityConfigItem("Q1", "eta=0.250", 1.00, 1.50, 1.00, 0.250, "eta_sweep", "Low quality bonus (-50%)", False),
        SensitivityConfigItem("Q2", "eta=0.375", 1.00, 1.50, 1.00, 0.375, "eta_sweep", "Moderate-low quality bonus (-25%)", False),
        SensitivityConfigItem("Q3", "eta=0.500 (Ref)", 1.00, 1.50, 1.00, 0.500, "eta_sweep", "Reference quality bonus (0%)", True),
        SensitivityConfigItem("Q4", "eta=0.625", 1.00, 1.50, 1.00, 0.625, "eta_sweep", "Moderate-high quality bonus (+25%)", False),
        SensitivityConfigItem("Q5", "eta=0.750", 1.00, 1.50, 1.00, 0.750, "eta_sweep", "High quality bonus (+50%)", False),
    ])

    # 5. Combined Factorial Sweep (Low, Reference, High)
    grid.extend([
        SensitivityConfigItem("L", "Combined Low", 0.75, 1.25, 0.75, 0.375, "combined_sweep", "Simultaneous moderate low perturbation (-25%)", False),
        SensitivityConfigItem("M", "Combined Ref", 1.00, 1.50, 1.00, 0.500, "combined_sweep", "Simultaneous reference configuration (Theta_0)", True),
        SensitivityConfigItem("H", "Combined High", 1.25, 1.75, 1.25, 0.625, "combined_sweep", "Simultaneous moderate high perturbation (+25%)", False),
    ])

    return grid
