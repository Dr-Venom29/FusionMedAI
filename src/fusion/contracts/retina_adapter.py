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

    DEFAULT_VERSION = "retina_efficientnet_b3_v1.0"
    DEFAULT_RELIABILITY = 0.885  # Placeholder prior to Phase C11.3 Global Reliability anchor

    def __init__(
        self,
        retina_module: Optional[Any] = None,
        frozen_reliability: float = DEFAULT_RELIABILITY,
        model_version: str = DEFAULT_VERSION,
    ):
        self.retina_module = retina_module
        self.frozen_reliability = float(np.clip(frozen_reliability, 0.0, 1.0))
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
            quality: Optional image quality metric in [0.0, 1.0]. Defaults to 1.0 if not provided.
            reliability: Optional override for frozen reliability.

        Returns:
            ModalityOutput: Immutable 8-tuple contract.
        """
        rel = reliability if reliability is not None else self.frozen_reliability

        if pred_dict is None:
            return ModalityOutput.unavailable(
                model_version=self.model_version,
                reliability=rel,
            )

        calib_probs = pred_dict.get("calib_probabilities")
        if calib_probs is None or len(calib_probs) != 5:
            raise ModalityContractValidationError(
                f"Retina prediction dict must contain 5-class 'calib_probabilities', got {calib_probs}"
            )

        # Normalize probability sum to exactly 1.0 within tolerance
        probs_np = np.asarray(calib_probs, dtype=np.float64)
        probs_np = probs_np / np.sum(probs_np)
        norm_probs = [float(p) for p in probs_np]

        # Deterministic Risk Projection: r_R = sum_{k=0}^4 (k/4) * p_k
        risk = project_retina_risk(norm_probs)

        # Confidence: Normalized margin between top two predicted probabilities in [0.0, 1.0]
        sorted_probs = np.sort(probs_np)
        confidence = float(np.clip(sorted_probs[-1] - sorted_probs[-2], 0.0, 1.0))

        # Uncertainty: Normalized predictive variance from MC Dropout passes
        mc_var = pred_dict.get("mc_predictive_variance", 0.0)
        # Scaled to [0, 1] interval based on theoretical max variance for 5 classes
        norm_unc = float(np.clip(mc_var / 0.16, 0.0, 1.0)) if mc_var > 0 else 0.0

        # Quality: If not specified, default to high quality for valid processed inputs
        q_val = float(np.clip(quality if quality is not None else 1.0, 0.0, 1.0))

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
