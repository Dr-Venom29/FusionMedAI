"""
FusionMedAI - Phase C11.1: Unified Modality Output Contract
Defines the immutable 8-tuple data structure and validation invariants for modality outputs.
"""

from dataclasses import dataclass, field
from typing import List, Union, Optional
import numpy as np


class ModalityContractValidationError(ValueError):
    """Raised when a modality output violates contract domain, sum, or type invariants."""
    pass


def project_retina_risk(calibrated_probability: Union[List[float], np.ndarray]) -> float:
    """
    Deterministic ordinal severity projection for 5-class Diabetic Retinopathy:
        r_R = sum_{k=0}^4 (k / 4) * p_k

    Args:
        calibrated_probability: 5-class calibrated probability distribution vector.

    Returns:
        float: Scalar continuous risk projection in [0.0, 1.0].
    """
    if len(calibrated_probability) != 5:
        raise ModalityContractValidationError(
            f"Retina risk projection requires 5 class probabilities, got {len(calibrated_probability)}"
        )
    weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    probs = np.asarray(calibrated_probability, dtype=np.float64)
    risk_val = float(np.sum(weights * probs))
    return float(np.clip(risk_val, 0.0, 1.0))


def project_foot_risk(calibrated_probability: Union[List[float], np.ndarray]) -> float:
    """
    Deterministic severity projection for 4-class Wagner Diabetic Foot Ulcers:
        r_F = sum_{k=0}^3 (k / 3) * p_k

    Args:
        calibrated_probability: 4-class calibrated probability distribution vector.

    Returns:
        float: Scalar continuous risk projection in [0.0, 1.0].
    """
    if len(calibrated_probability) != 4:
        raise ModalityContractValidationError(
            f"Foot risk projection requires 4 class probabilities, got {len(calibrated_probability)}"
        )
    weights = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float64)
    probs = np.asarray(calibrated_probability, dtype=np.float64)
    risk_val = float(np.sum(weights * probs))
    return float(np.clip(risk_val, 0.0, 1.0))


def project_clinical_risk(calibrated_probability: Union[List[float], np.ndarray]) -> float:
    """
    Deterministic risk projection for binary 30-day hospital readmission:
        r_C = p_1 = P(Y = 1 | x)

    Args:
        calibrated_probability: 2-class calibrated probability distribution [p0, p1] or single probability p1.

    Returns:
        float: Calibrated binary event risk probability in [0.0, 1.0].
    """
    if len(calibrated_probability) == 2:
        return float(np.clip(calibrated_probability[1], 0.0, 1.0))
    elif len(calibrated_probability) == 1:
        return float(np.clip(calibrated_probability[0], 0.0, 1.0))
    else:
        raise ModalityContractValidationError(
            f"Clinical risk projection requires binary probabilities (len 1 or 2), got {len(calibrated_probability)}"
        )


@dataclass(frozen=True)
class ModalityOutput:
    """
    Unified Modality Output Contract (8-Tuple).
    Strictly immutable payload emitted by all constituent modality pipelines before fusion routing.

    Attributes:
        risk: Continuous scalar risk projection in [0.0, 1.0].
        calibrated_probability: Full class distribution vector summing to 1.0 (empty list if unavailable).
        confidence: Normalized model certainty in [0.0, 1.0].
        uncertainty: Normalized predictive uncertainty in [0.0, 1.0].
        quality: Input signal quality index in [0.0, 1.0].
        availability: Binary presence indicator (True/False).
        reliability: Validation-anchored reliability index in [0.0, 1.0].
        model_version: Checkpoint / version tracking string.
    """
    risk: float
    calibrated_probability: List[float]
    confidence: float
    uncertainty: float
    quality: float
    availability: bool
    reliability: float
    model_version: str

    def __post_init__(self):
        # 1. Type and Identity Validations
        if not isinstance(self.availability, bool):
            raise ModalityContractValidationError(
                f"availability must be a bool, got {type(self.availability).__name__}"
            )
        if not isinstance(self.model_version, str) or len(self.model_version.strip()) == 0:
            raise ModalityContractValidationError(
                f"model_version must be a non-empty string, got {self.model_version!r}"
            )

        # 2. Invariant C: Unavailable Handling
        if not self.availability:
            if self.risk != 0.0:
                raise ModalityContractValidationError(
                    f"When availability is False, risk must be 0.0, got {self.risk}"
                )
            if len(self.calibrated_probability) != 0:
                raise ModalityContractValidationError(
                    f"When availability is False, calibrated_probability must be empty list [], got {self.calibrated_probability}"
                )
            # Ensure scalar invariants still hold for metadata fields
            self._validate_scalar("confidence", self.confidence)
            self._validate_scalar("uncertainty", self.uncertainty)
            self._validate_scalar("quality", self.quality)
            self._validate_scalar("reliability", self.reliability)
            return

        # 3. Invariant A: Scalar Range Validations for Available Modalities
        self._validate_scalar("risk", self.risk)
        self._validate_scalar("confidence", self.confidence)
        self._validate_scalar("uncertainty", self.uncertainty)
        self._validate_scalar("quality", self.quality)
        self._validate_scalar("reliability", self.reliability)

        # 4. Invariant B: Probability Vector Invariants
        if not isinstance(self.calibrated_probability, (list, tuple)) or len(self.calibrated_probability) == 0:
            raise ModalityContractValidationError(
                "calibrated_probability cannot be empty when availability is True."
            )

        for idx, p in enumerate(self.calibrated_probability):
            if not isinstance(p, (int, float, np.floating)) or np.isnan(p) or p < 0.0 or p > 1.0:
                raise ModalityContractValidationError(
                    f"calibrated_probability[{idx}] = {p} is out of bounds [0.0, 1.0] or invalid."
                )

        prob_sum = float(sum(self.calibrated_probability))
        if abs(prob_sum - 1.0) > 1e-5:
            raise ModalityContractValidationError(
                f"calibrated_probability sum must equal 1.0 ± 1e-5, got sum = {prob_sum:.8f}"
            )

    @staticmethod
    def _validate_scalar(name: str, val: float) -> None:
        if not isinstance(val, (int, float, np.floating)) or np.isnan(val) or val < 0.0 or val > 1.0:
            raise ModalityContractValidationError(
                f"Scalar field '{name}' must be within [0.0, 1.0], got {val}"
            )

    @classmethod
    def unavailable(
        cls,
        model_version: str = "frozen_unavailable_v1",
        reliability: float = 0.0,
        uncertainty: float = 1.0,
        quality: float = 0.0,
        confidence: float = 0.0,
    ) -> "ModalityOutput":
        """Factory method to generate a standard compliant unavailable modality contract payload."""
        return cls(
            risk=0.0,
            calibrated_probability=[],
            confidence=confidence,
            uncertainty=uncertainty,
            quality=quality,
            availability=False,
            reliability=reliability,
            model_version=model_version,
        )
