"""
FusionMedAI - Phase C11.1: Clinical Module Contract Adapter
Adapts outputs from the frozen ClinicalInferenceService (CatBoost HPO / Isotonic Calibration / Bootstrap Ensemble)
into the unified immutable ModalityOutput contract.
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np

from src.fusion.contracts.modality_output import (
    ModalityOutput,
    project_clinical_risk,
    ModalityContractValidationError,
)


class ClinicalAdapter:
    """
    Adapter converting raw Clinical diagnostic outputs or running frozen ClinicalInferenceService inference
    to produce a verified ModalityOutput 8-tuple.
    """

    DEFAULT_VERSION = "clinical_catboost_hpo_v1.0"
    DEFAULT_RELIABILITY = 0.820  # Placeholder prior to Phase C11.3 Global Reliability anchor

    def __init__(
        self,
        clinical_service: Optional[Any] = None,
        frozen_reliability: float = DEFAULT_RELIABILITY,
        model_version: str = DEFAULT_VERSION,
    ):
        self.clinical_service = clinical_service
        self.frozen_reliability = float(np.clip(frozen_reliability, 0.0, 1.0))
        self.model_version = model_version

    def adapt_prediction_output(
        self,
        clinical_output: Optional[Any],
        quality: Optional[float] = None,
        reliability: Optional[float] = None,
    ) -> ModalityOutput:
        """
        Adapts a ClinicalOutput dataclass or dictionary into a ModalityOutput.

        Args:
            clinical_output: Object or dict from ClinicalInferenceService. If None, returns unavailable.
            quality: Optional feature completeness metric in [0.0, 1.0].
            reliability: Optional override for frozen reliability.

        Returns:
            ModalityOutput: Immutable 8-tuple contract.
        """
        rel = reliability if reliability is not None else self.frozen_reliability

        if clinical_output is None:
            return ModalityOutput.unavailable(
                model_version=self.model_version,
                reliability=rel,
            )

        # Extract calibrated probability p1
        if hasattr(clinical_output, "calibrated_probability"):
            p1 = float(clinical_output.calibrated_probability)
        elif isinstance(clinical_output, dict):
            p1 = float(clinical_output.get("calibrated_probability", 0.0))
        else:
            raise ModalityContractValidationError(
                f"Unexpected clinical output format: {type(clinical_output).__name__}"
            )

        p1 = float(np.clip(p1, 0.0, 1.0))
        calib_vector = [float(1.0 - p1), float(p1)]

        # Deterministic Risk: r_C = p1
        risk = project_clinical_risk(calib_vector)

        # Confidence: Inverted margin distance 1 - 2*|p1 - 0.5| in [0.0, 1.0]
        if hasattr(clinical_output, "confidence"):
            confidence = float(np.clip(clinical_output.confidence, 0.0, 1.0))
        elif isinstance(clinical_output, dict) and "confidence" in clinical_output:
            confidence = float(np.clip(clinical_output["confidence"], 0.0, 1.0))
        else:
            confidence = float(np.clip(1.0 - 2.0 * abs(p1 - 0.5), 0.0, 1.0))

        # Uncertainty: Normalized bootstrap standard deviation / variance
        if hasattr(clinical_output, "uncertainty") and hasattr(clinical_output.uncertainty, "std_probability"):
            std_prob = float(clinical_output.uncertainty.std_probability)
        elif isinstance(clinical_output, dict) and "uncertainty" in clinical_output:
            unc = clinical_output["uncertainty"]
            std_prob = float(unc.get("std_probability", 0.0)) if isinstance(unc, dict) else 0.0
        else:
            std_prob = 0.0

        # Normalized to [0, 1] relative to typical binary variance scale (max theoretical std is 0.5)
        norm_unc = float(np.clip(std_prob / 0.25, 0.0, 1.0))

        # Quality: Tabular completeness ratio (1.0 - missingness_ratio)
        if quality is not None:
            q_val = float(np.clip(quality, 0.0, 1.0))
        elif hasattr(clinical_output, "shift_detection") and hasattr(clinical_output.shift_detection, "missingness_ratio"):
            q_val = float(np.clip(1.0 - clinical_output.shift_detection.missingness_ratio, 0.0, 1.0))
        elif isinstance(clinical_output, dict) and "shift_detection" in clinical_output:
            sd = clinical_output["shift_detection"]
            m_ratio = float(sd.get("missingness_ratio", 0.0)) if isinstance(sd, dict) else 0.0
            q_val = float(np.clip(1.0 - m_ratio, 0.0, 1.0))
        else:
            q_val = 1.0

        return ModalityOutput(
            risk=risk,
            calibrated_probability=calib_vector,
            confidence=confidence,
            uncertainty=norm_unc,
            quality=q_val,
            availability=True,
            reliability=rel,
            model_version=self.model_version,
        )

    def adapt_encounter(
        self,
        encounter_data: Optional[Union[Dict[str, Any], Any]],
        quality: Optional[float] = None,
    ) -> ModalityOutput:
        """
        Runs inference on the frozen ClinicalInferenceService and adapts output into ModalityOutput.

        Args:
            encounter_data: Raw EHR encounter dictionary. If None, emits unavailable contract.
            quality: Feature completeness index.

        Returns:
            ModalityOutput: Immutable 8-tuple contract.
        """
        if encounter_data is None:
            return ModalityOutput.unavailable(
                model_version=self.model_version,
                reliability=self.frozen_reliability,
            )

        if self.clinical_service is None:
            raise RuntimeError("ClinicalInferenceService instance required for direct encounter adaptation.")

        pred_output = self.clinical_service.predict_single(encounter_data)
        return self.adapt_prediction_output(pred_output, quality=quality)
