"""
FusionMedAI - Phase C11.1: Foot Module Contract Adapter
Adapts outputs from the frozen FootModule (EfficientNet-B3 / Vector Scaling / MC Dropout Option B)
into the unified immutable ModalityOutput contract.
"""

from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from PIL import Image
import numpy as np

from src.fusion.contracts.modality_output import (
    ModalityOutput,
    project_foot_risk,
    ModalityContractValidationError,
)


class FootAdapter:
    """
    Adapter converting raw Foot diagnostic outputs or running frozen FootModule inference
    to produce a verified ModalityOutput 8-tuple.
    """

    from src.fusion.reliability.global_reliability import FROZEN_FOOT_RELIABILITY
    DEFAULT_VERSION = "foot_efficientnet_b3_v1.0"
    DEFAULT_RELIABILITY = FROZEN_FOOT_RELIABILITY

    def __init__(
        self,
        foot_module: Optional[Any] = None,
        frozen_reliability: float = DEFAULT_RELIABILITY,
        model_version: str = DEFAULT_VERSION,
    ):
        if not isinstance(frozen_reliability, (int, float)) or isinstance(frozen_reliability, bool) or not np.isfinite(frozen_reliability) or not (0.0 <= frozen_reliability <= 1.0):
            raise ValueError(f"frozen_reliability must be a finite float in [0.0, 1.0], got {frozen_reliability!r}")
        self.foot_module = foot_module
        self.frozen_reliability = float(frozen_reliability)
        self.model_version = model_version

    def adapt_prediction_dict(
        self,
        pred_dict: Optional[Dict[str, Any]],
        quality: Optional[float] = None,
        reliability: Optional[float] = None,
    ) -> ModalityOutput:
        """
        Adapts a pre-computed FootModule prediction dictionary into a ModalityOutput.

        Args:
            pred_dict: Output dictionary from FootModule.predict(). If None, returns unavailable.
            quality: Optional wound image quality metric in [0.0, 1.0].
            reliability: Optional override for frozen reliability.

        Returns:
            ModalityOutput: Immutable 8-tuple contract.
        """
        if reliability is not None:
            if not isinstance(reliability, (int, float)) or isinstance(reliability, bool) or not np.isfinite(reliability) or not (0.0 <= reliability <= 1.0):
                raise ModalityContractValidationError(f"reliability override must be a finite float in [0.0, 1.0], got {reliability!r}")
            rel = float(reliability)
        else:
            rel = self.frozen_reliability

        if pred_dict is None:
            return ModalityOutput.unavailable(
                model_version=self.model_version,
                reliability=rel,
            )

        calib_probs = pred_dict.get("calibrated_probabilities")
        if calib_probs is None:
            calib_probs = pred_dict.get("calib_probabilities")

        from src.fusion.contracts.evidence_report import validate_probability_vector
        try:
            validate_probability_vector(calib_probs, 4)
        except Exception as e:
            raise ModalityContractValidationError(f"Invalid raw calibrated probabilities: {e}") from e

        norm_probs = [float(p) for p in calib_probs]
        probs_np = np.asarray(norm_probs, dtype=np.float64)

        # Deterministic Risk Projection: r_F = sum_{k=0}^3 (k/3) * p_k
        risk = project_foot_risk(norm_probs)

        # Confidence: Maximum calibrated class probability max_k p_k in [0.0, 1.0]
        confidence = float(np.clip(np.max(probs_np), 0.0, 1.0))

        # Uncertainty: Normalized predictive entropy from MC Dropout passes
        entropy_norm = pred_dict.get("mc_predictive_entropy_norm")
        if entropy_norm is None:
            entropy_norm = pred_dict.get("calib_entropy_norm")
        if entropy_norm is None:
            raise ModalityContractValidationError(
                "Foot prediction dict must contain 'mc_predictive_entropy_norm' or 'calib_entropy_norm' for uncertainty estimation."
            )
        if not isinstance(entropy_norm, (int, float)) or isinstance(entropy_norm, bool) or not np.isfinite(entropy_norm) or not (0.0 <= entropy_norm <= 1.0):
            raise ModalityContractValidationError(
                f"Foot normalized predictive entropy must be a finite float in [0.0, 1.0], got {entropy_norm!r}."
            )
        norm_unc = float(entropy_norm)

        # Quality: Default 0.5 neutral fallback for unmeasured quality
        if quality is not None:
            if not isinstance(quality, (int, float)) or isinstance(quality, bool) or not np.isfinite(quality) or not (0.0 <= quality <= 1.0):
                raise ModalityContractValidationError(f"quality must be a finite float in [0.0, 1.0], got {quality!r}")
            q_val = float(quality)
        else:
            q_val = 0.5

        return ModalityOutput(
            risk=risk,
            calibrated_probability=norm_probs,
            confidence=confidence,
            uncertainty=norm_unc,
            quality=q_val,
            availability=True,
            reliability=rel,
            model_version=self.model_version,
        )

    def adapt_to_semantic_prediction(
        self,
        pred_dict: Optional[Dict[str, Any]],
        quality: Optional[float] = None,
        reliability: Optional[float] = None,
    ):
        """
        Adapts Foot diagnostic output into the semantic ModalityPrediction schema.
        """
        from src.fusion.contracts.evidence_report import ModalityPrediction, FOOT_TARGET
        from src.fusion.reliability.global_reliability import FROZEN_FOOT_RELIABILITY

        if reliability is not None:
            if not isinstance(reliability, (int, float)) or isinstance(reliability, bool) or not np.isfinite(reliability) or not (0.0 <= reliability <= 1.0):
                raise ModalityContractValidationError(f"reliability override must be a finite float in [0.0, 1.0], got {reliability!r}")
            rel = float(reliability)
        else:
            rel = self.frozen_reliability if self.frozen_reliability is not None else FROZEN_FOOT_RELIABILITY

        if pred_dict is None:
            return ModalityPrediction(
                modality="foot",
                task_id=FOOT_TARGET.name,
                target=FOOT_TARGET,
                prediction={"class": None, "probabilities": [], "calibration_method": "vector_scaling"},
                uncertainty={"method": "mc_dropout_option_b_normalized_entropy", "value": 1.0, "interpretation": "unavailable"},
                quality=0.0,
                available=False,
                reliability=rel,
                model_version=self.model_version,
                legacy_projected_scalar=0.0,
            )

        calib_probs = pred_dict.get("calibrated_probabilities")
        if calib_probs is None:
            calib_probs = pred_dict.get("calib_probabilities")

        from src.fusion.contracts.evidence_report import validate_probability_vector
        try:
            validate_probability_vector(calib_probs, 4)
        except Exception as e:
            raise ModalityContractValidationError(f"Invalid raw calibrated probabilities: {e}") from e

        norm_probs = [float(p) for p in calib_probs]
        probs_np = np.asarray(norm_probs, dtype=np.float64)
        risk = project_foot_risk(norm_probs)
        pred_class = int(np.argmax(probs_np))

        # Canonical uncertainty from MC Dropout Option B: mc_predictive_entropy_norm, with explicit fallback to calib_entropy_norm
        if "mc_predictive_entropy_norm" in pred_dict and pred_dict["mc_predictive_entropy_norm"] is not None:
            entropy_norm = pred_dict["mc_predictive_entropy_norm"]
            unc_method = "mc_dropout_option_b_normalized_entropy"
            unc_source = "mc_predictive_entropy_norm"
            unc_interp = "stochastic_mc_predictive_entropy"
        elif "calib_entropy_norm" in pred_dict and pred_dict["calib_entropy_norm"] is not None:
            entropy_norm = pred_dict["calib_entropy_norm"]
            unc_method = "calibrated_predictive_entropy"
            unc_source = "calib_entropy_norm"
            unc_interp = "calibrated_distribution_entropy"
        else:
            raise ModalityContractValidationError(
                "Foot prediction dict must contain 'mc_predictive_entropy_norm' or 'calib_entropy_norm' for uncertainty estimation."
            )

        if not isinstance(entropy_norm, (int, float)) or isinstance(entropy_norm, bool) or not np.isfinite(entropy_norm) or not (0.0 <= entropy_norm <= 1.0):
            raise ModalityContractValidationError(
                f"Foot normalized predictive entropy must be a finite float in [0.0, 1.0], got {entropy_norm!r}."
            )
        norm_unc = float(entropy_norm)

        # Traceable quality fallback
        if quality is not None:
            if not isinstance(quality, (int, float)) or isinstance(quality, bool) or not np.isfinite(quality) or not (0.0 <= quality <= 1.0):
                raise ModalityContractValidationError(f"quality must be a finite float in [0.0, 1.0], got {quality!r}")
            q_val = float(quality)
            q_prov = "measured"
        else:
            q_val = 0.5  # Neutral fallback for unmeasured wound image quality
            q_prov = "unmeasured_neutral_fallback"

        return ModalityPrediction(
            modality="foot",
            task_id=FOOT_TARGET.name,
            target=FOOT_TARGET,
            prediction={
                "class": pred_class,
                "probabilities": norm_probs,
                "calibration_method": "vector_scaling",
                "quality_provenance": q_prov,
            },
            uncertainty={
                "method": unc_method,
                "source_field": unc_source,
                "value": norm_unc,
                "interpretation": unc_interp,
            },
            quality=q_val,
            available=True,
            reliability=rel,
            model_version=self.model_version,
            legacy_projected_scalar=risk,
        )

    def adapt_image(
        self,
        image: Optional[Union[str, Path, Image.Image]],
        mc_passes: int = 10,
        quality: Optional[float] = None,
    ) -> ModalityOutput:
        """
        Runs inference on the frozen FootModule and adapts output into ModalityOutput.

        Args:
            image: Image path or PIL Image. If None, emits unavailable contract.
            mc_passes: Number of stochastic passes (default 10).
            quality: Image quality index.

        Returns:
            ModalityOutput: Immutable 8-tuple contract.
        """
        if image is None:
            return ModalityOutput.unavailable(
                model_version=self.model_version,
                reliability=self.frozen_reliability,
            )

        if self.foot_module is None:
            raise RuntimeError("FootModule instance required for direct image adaptation.")

        pred_dict = self.foot_module.predict(image=image, mc_passes=mc_passes, generate_cam=False)
        return self.adapt_prediction_dict(pred_dict, quality=quality)
