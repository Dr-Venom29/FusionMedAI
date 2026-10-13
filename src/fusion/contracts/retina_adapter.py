"""
FusionMedAI - Phase C11.1: Retina Module Contract Adapter
Adapts outputs from the frozen RetinaModule (EfficientNet-B3 / Temperature Scaling / MC Dropout)
into the unified immutable ModalityOutput contract.
"""

from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from PIL import Image
import numpy as np

from src.fusion.contracts.modality_output import (
    ModalityOutput,
    project_retina_risk,
    ModalityContractValidationError,
)


class RetinaAdapter:
    """
    Adapter converting raw Retina diagnostic outputs or running frozen RetinaModule inference
    to produce a verified ModalityOutput 8-tuple.
    """

    from src.fusion.reliability.global_reliability import FROZEN_RETINA_RELIABILITY
    DEFAULT_VERSION = "retina_efficientnet_b3_v1.0"
    DEFAULT_RELIABILITY = FROZEN_RETINA_RELIABILITY
    THEORETICAL_MAX_VARIANCE = 0.16
    VARIANCE_FLOAT_TOLERANCE = 1e-6

    def __init__(
        self,
        retina_module: Optional[Any] = None,
        frozen_reliability: float = DEFAULT_RELIABILITY,
        model_version: str = DEFAULT_VERSION,
    ):
        if not isinstance(frozen_reliability, (int, float)) or isinstance(frozen_reliability, bool) or not np.isfinite(frozen_reliability) or not (0.0 <= frozen_reliability <= 1.0):
            raise ValueError(f"frozen_reliability must be a finite float in [0.0, 1.0], got {frozen_reliability!r}")
        self.retina_module = retina_module
        self.frozen_reliability = float(frozen_reliability)
        self.model_version = model_version

    def adapt_prediction_dict(
        self,
        pred_dict: Optional[Dict[str, Any]],
        quality: Optional[float] = None,
        reliability: Optional[float] = None,
    ) -> ModalityOutput:
        """
        Adapts a pre-computed RetinaModule prediction dictionary into a ModalityOutput.

        Args:
            pred_dict: Output dictionary from RetinaModule.predict(). If None, returns unavailable.
            quality: Optional image quality metric in [0.0, 1.0].
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

        calib_probs = pred_dict.get("calib_probabilities")
        from src.fusion.contracts.evidence_report import validate_probability_vector
        try:
            validate_probability_vector(calib_probs, 5)
        except Exception as e:
            raise ModalityContractValidationError(f"Invalid raw calib_probabilities: {e}") from e

        norm_probs = [float(p) for p in calib_probs]
        probs_np = np.asarray(norm_probs, dtype=np.float64)

        # Deterministic Risk Projection: r_R = sum_{k=0}^4 (k/4) * p_k
        risk = project_retina_risk(norm_probs)

        # Confidence: Normalized margin between top two predicted probabilities in [0.0, 1.0]
        sorted_probs = np.sort(probs_np)
        confidence = float(np.clip(sorted_probs[-1] - sorted_probs[-2], 0.0, 1.0))

        # Uncertainty: Normalized predictive variance from MC Dropout passes.
        # In RetinaModule.predict() (src/retina/retina_module.py:184), mc_predictive_variance is defined as
        # float(np.mean(np.var(mc_probs, axis=0))) for 5-class probability vectors on the simplex.
        # Since sum_k E[P_k^2] <= 1 and sum_k (E[P_k])^2 >= 1/5 = 0.20 by Cauchy-Schwarz,
        # sum_k Var(P_k) <= 0.80, so the mean across 5 classes is strictly bounded by 0.80 / 5 = 0.16.
        # Values in (0.16, 0.16 + 1e-6] are accepted to accommodate IEEE-754 floating-point rounding and clipped to 1.0;
        # values strictly exceeding 0.16 + 1e-6 are rejected as invalid upstream metrics.
        if "mc_predictive_variance" not in pred_dict or pred_dict["mc_predictive_variance"] is None:
            raise ModalityContractValidationError(
                "Retina prediction dict must contain 'mc_predictive_variance' for uncertainty estimation."
            )
        mc_var = pred_dict["mc_predictive_variance"]
        if (
            not isinstance(mc_var, (int, float))
            or isinstance(mc_var, bool)
            or not np.isfinite(mc_var)
            or mc_var < 0.0
            or mc_var > self.THEORETICAL_MAX_VARIANCE + self.VARIANCE_FLOAT_TOLERANCE
        ):
            raise ModalityContractValidationError(
                f"'mc_predictive_variance' must be a finite float in [0.0, 0.16] "
                f"(theoretical upper bound for 5-class simplex mean variance ± 1e-6 float tolerance), got {mc_var!r}."
            )
        norm_unc = float(np.clip(mc_var / self.THEORETICAL_MAX_VARIANCE, 0.0, 1.0))

        # Quality: If not specified, default to neutral 0.5 fallback
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
        Adapts Retina diagnostic output into the semantic ModalityPrediction schema.
        """
        from src.fusion.contracts.evidence_report import ModalityPrediction, RETINA_TARGET
        from src.fusion.reliability.global_reliability import FROZEN_RETINA_RELIABILITY

        if reliability is not None:
            if not isinstance(reliability, (int, float)) or isinstance(reliability, bool) or not np.isfinite(reliability) or not (0.0 <= reliability <= 1.0):
                raise ModalityContractValidationError(f"reliability override must be a finite float in [0.0, 1.0], got {reliability!r}")
            rel = float(reliability)
        else:
            rel = self.frozen_reliability if self.frozen_reliability is not None else FROZEN_RETINA_RELIABILITY

        if pred_dict is None:
            return ModalityPrediction(
                modality="retina",
                task_id=RETINA_TARGET.name,
                target=RETINA_TARGET,
                prediction={"class": None, "probabilities": [], "calibration_method": "temperature_scaling"},
                uncertainty={"method": "mc_dropout_predictive_variance", "value": 1.0, "interpretation": "unavailable"},
                quality=0.0,
                available=False,
                reliability=rel,
                model_version=self.model_version,
                legacy_projected_scalar=0.0,
            )

        calib_probs = pred_dict.get("calib_probabilities")
        from src.fusion.contracts.evidence_report import validate_probability_vector
        try:
            validate_probability_vector(calib_probs, 5)
        except Exception as e:
            raise ModalityContractValidationError(f"Invalid raw calib_probabilities: {e}") from e

        norm_probs = [float(p) for p in calib_probs]
        probs_np = np.asarray(norm_probs, dtype=np.float64)
        risk = project_retina_risk(norm_probs)
        pred_class = int(np.argmax(probs_np))

        if "mc_predictive_variance" not in pred_dict or pred_dict["mc_predictive_variance"] is None:
            raise ModalityContractValidationError(
                "Retina prediction dict must contain 'mc_predictive_variance' for uncertainty estimation."
            )
        mc_var = pred_dict["mc_predictive_variance"]
        if (
            not isinstance(mc_var, (int, float))
            or isinstance(mc_var, bool)
            or not np.isfinite(mc_var)
            or mc_var < 0.0
            or mc_var > self.THEORETICAL_MAX_VARIANCE + self.VARIANCE_FLOAT_TOLERANCE
        ):
            raise ModalityContractValidationError(
                f"'mc_predictive_variance' must be a finite float in [0.0, 0.16] "
                f"(theoretical upper bound for 5-class simplex mean variance ± 1e-6 float tolerance), got {mc_var!r}."
            )
        norm_unc = float(np.clip(mc_var / self.THEORETICAL_MAX_VARIANCE, 0.0, 1.0))

        if quality is not None:
            if not isinstance(quality, (int, float)) or isinstance(quality, bool) or not np.isfinite(quality) or not (0.0 <= quality <= 1.0):
                raise ModalityContractValidationError(f"quality must be a finite float in [0.0, 1.0], got {quality!r}")
            q_val = float(quality)
            q_prov = "measured"
        else:
            q_val = 0.5  # Neutral fallback for unmeasured image quality
            q_prov = "unmeasured_neutral_fallback"

        return ModalityPrediction(
            modality="retina",
            task_id=RETINA_TARGET.name,
            target=RETINA_TARGET,
            prediction={
                "class": pred_class,
                "probabilities": norm_probs,
                "calibration_method": "temperature_scaling",
                "quality_provenance": q_prov,
            },
            uncertainty={
                "method": "mc_dropout_predictive_variance",
                "value": norm_unc,
                "raw_metric": float(mc_var),
                "normalization_scale": 0.16,
                "interpretation": "epistemic_retinal_feature_uncertainty",
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
        mc_passes: int = 25,
        quality: Optional[float] = None,
    ) -> ModalityOutput:
        """
        Runs inference on the frozen RetinaModule and adapts output into ModalityOutput.

        Args:
            image: Image path or PIL Image. If None, emits unavailable contract.
            mc_passes: Number of stochastic passes (default 25).
            quality: Image quality index.

        Returns:
            ModalityOutput: Immutable 8-tuple contract.
        """
        if image is None:
            return ModalityOutput.unavailable(
                model_version=self.model_version,
                reliability=self.frozen_reliability,
            )

        if self.retina_module is None:
            raise RuntimeError("RetinaModule instance required for direct image adaptation.")

        pred_dict = self.retina_module.predict(image=image, mc_passes=mc_passes, generate_cam=False)
        return self.adapt_prediction_dict(pred_dict, quality=quality)
