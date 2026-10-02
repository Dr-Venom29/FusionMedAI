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

    DEFAULT_VERSION = "foot_efficientnet_b3_v1.0"
    DEFAULT_RELIABILITY = 0.850  # Placeholder prior to Phase C11.3 Global Reliability anchor

    def __init__(
        self,
        foot_module: Optional[Any] = None,
        frozen_reliability: float = DEFAULT_RELIABILITY,
        model_version: str = DEFAULT_VERSION,
    ):
        self.foot_module = foot_module
        self.frozen_reliability = float(np.clip(frozen_reliability, 0.0, 1.0))
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
            quality: Optional wound image quality metric in [0.0, 1.0]. Defaults to 1.0.
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

        calib_probs = pred_dict.get("calibrated_probabilities")
        if calib_probs is None:
            # Fallback check for alternate key
            calib_probs = pred_dict.get("calib_probabilities")

        if calib_probs is None or len(calib_probs) != 4:
            raise ModalityContractValidationError(
                f"Foot prediction dict must contain 4-class calibrated probabilities, got {calib_probs}"
            )

        # Normalize probability sum to exactly 1.0 within tolerance
        probs_np = np.asarray(calib_probs, dtype=np.float64)
        probs_np = probs_np / np.sum(probs_np)
        norm_probs = [float(p) for p in probs_np]

        # Deterministic Risk Projection: r_F = sum_{k=0}^3 (k/3) * p_k
        risk = project_foot_risk(norm_probs)

        # Confidence: Maximum calibrated class probability max_k p_k in [0.0, 1.0]
        confidence = float(np.clip(np.max(probs_np), 0.0, 1.0))

        # Uncertainty: Normalized predictive entropy from MC Dropout passes
        entropy_norm = pred_dict.get("mc_predictive_entropy_norm")
        if entropy_norm is None:
            entropy_norm = pred_dict.get("calib_entropy_norm", 0.0)
        norm_unc = float(np.clip(entropy_norm, 0.0, 1.0))

        # Quality: Default 1.0 for valid input
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
