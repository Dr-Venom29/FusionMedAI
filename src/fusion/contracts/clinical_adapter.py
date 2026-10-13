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

    from src.fusion.reliability.global_reliability import FROZEN_CLINICAL_RELIABILITY
    DEFAULT_VERSION = "clinical_catboost_hpo_v1.0"
    DEFAULT_RELIABILITY = FROZEN_CLINICAL_RELIABILITY

    def __init__(
        self,
        clinical_service: Optional[Any] = None,
        frozen_reliability: float = DEFAULT_RELIABILITY,
        model_version: str = DEFAULT_VERSION,
    ):
        if not isinstance(frozen_reliability, (int, float)) or isinstance(frozen_reliability, bool) or not np.isfinite(frozen_reliability) or not (0.0 <= frozen_reliability <= 1.0):
            raise ValueError(f"frozen_reliability must be a finite float in [0.0, 1.0], got {frozen_reliability!r}")
        self.clinical_service = clinical_service
        self.frozen_reliability = float(frozen_reliability)
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
        if reliability is not None:
            if not isinstance(reliability, (int, float)) or isinstance(reliability, bool) or not np.isfinite(reliability) or not (0.0 <= reliability <= 1.0):
                raise ModalityContractValidationError(f"reliability override must be a finite float in [0.0, 1.0], got {reliability!r}")
            rel = float(reliability)
        else:
            rel = self.frozen_reliability

        if clinical_output is None:
            return ModalityOutput.unavailable(
                model_version=self.model_version,
                reliability=rel,
            )

        # Extract calibrated probability p1 without silent clipping
        if hasattr(clinical_output, "calibrated_probability"):
            raw_p1 = clinical_output.calibrated_probability
        elif isinstance(clinical_output, dict):
            raw_p1 = clinical_output.get("calibrated_probability")
            if raw_p1 is None:
                raw_p1 = clinical_output.get("calibrated_p")
        else:
            raise ModalityContractValidationError(
                f"Unexpected clinical output format: {type(clinical_output).__name__}"
            )

        if raw_p1 is None or not isinstance(raw_p1, (int, float)) or isinstance(raw_p1, bool) or not np.isfinite(raw_p1) or raw_p1 < 0.0 or raw_p1 > 1.0:
            raise ModalityContractValidationError(
                f"calibrated readmission probability must be a finite float in [0.0, 1.0], got {raw_p1!r}"
            )

        p1 = float(raw_p1)
        calib_vector = [float(1.0 - p1), float(p1)]

        # Deterministic Risk: r_C = p1
        risk = project_clinical_risk(calib_vector)

        # Confidence: Decision boundary distance 2*|p1 - 0.5| in [0.0, 1.0]
        confidence = float(np.clip(2.0 * abs(p1 - 0.5), 0.0, 1.0))

        # Uncertainty: Normalized bootstrap standard deviation (max theoretical std is 0.5 for Bernoulli)
        if hasattr(clinical_output, "uncertainty") and hasattr(clinical_output.uncertainty, "std_probability"):
            std_prob = clinical_output.uncertainty.std_probability
        elif isinstance(clinical_output, dict) and "uncertainty" in clinical_output:
            unc = clinical_output["uncertainty"]
            std_prob = unc.get("std_probability") if isinstance(unc, dict) else unc
        else:
            std_prob = None

        if std_prob is None:
            raise ModalityContractValidationError(
                "Clinical output must contain 'std_probability' within uncertainty payload."
            )
        if not isinstance(std_prob, (int, float)) or isinstance(std_prob, bool) or not np.isfinite(std_prob) or std_prob < 0.0 or std_prob > 0.5:
            raise ModalityContractValidationError(
                f"Clinical bootstrap uncertainty std_probability must be a finite float in [0.0, 0.5], got {std_prob!r}"
            )

        norm_unc = float(std_prob / 0.5)

        # Quality: Tabular completeness ratio (1.0 - missingness_ratio)
        if quality is not None:
            if not isinstance(quality, (int, float)) or isinstance(quality, bool) or not np.isfinite(quality) or not (0.0 <= quality <= 1.0):
                raise ModalityContractValidationError(f"quality must be a finite float in [0.0, 1.0], got {quality!r}")
            q_val = float(quality)
        else:
            raw_m_ratio = None
            if hasattr(clinical_output, "shift_detection"):
                sd = clinical_output.shift_detection
                if hasattr(sd, "missingness_ratio"):
                    raw_m_ratio = sd.missingness_ratio
                elif isinstance(sd, dict):
                    raw_m_ratio = sd.get("missingness_ratio")
            elif isinstance(clinical_output, dict) and "shift_detection" in clinical_output:
                sd = clinical_output["shift_detection"]
                if isinstance(sd, dict):
                    raw_m_ratio = sd.get("missingness_ratio")

            raw_feat_comp = None
            if hasattr(clinical_output, "feature_completeness"):
                raw_feat_comp = clinical_output.feature_completeness
            elif isinstance(clinical_output, dict) and "feature_completeness" in clinical_output:
                raw_feat_comp = clinical_output["feature_completeness"]

            if raw_m_ratio is not None:
                if not isinstance(raw_m_ratio, (int, float)) or isinstance(raw_m_ratio, bool) or not np.isfinite(raw_m_ratio) or raw_m_ratio < 0.0 or raw_m_ratio > 1.0:
                    raise ModalityContractValidationError(
                        f"missingness_ratio must be a finite float in [0.0, 1.0], got {raw_m_ratio!r}"
                    )
                q_val = float(1.0 - raw_m_ratio)
            elif raw_feat_comp is not None:
                if not isinstance(raw_feat_comp, (int, float)) or isinstance(raw_feat_comp, bool) or not np.isfinite(raw_feat_comp) or raw_feat_comp < 0.0 or raw_feat_comp > 1.0:
                    raise ModalityContractValidationError(
                        f"feature_completeness must be a finite float in [0.0, 1.0], got {raw_feat_comp!r}"
                    )
                q_val = float(raw_feat_comp)
            else:
                q_val = 0.5  # Explicit neutral fallback for unmeasured quality

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

    def adapt_to_semantic_prediction(
        self,
        clinical_output: Optional[Any],
        quality: Optional[float] = None,
        reliability: Optional[float] = None,
    ):
        """
        Adapts Clinical diagnostic output into the semantic ModalityPrediction schema.
        """
        from src.fusion.contracts.evidence_report import ModalityPrediction, CLINICAL_TARGET
        from src.fusion.reliability.global_reliability import FROZEN_CLINICAL_RELIABILITY

        if reliability is not None:
            if not isinstance(reliability, (int, float)) or isinstance(reliability, bool) or not np.isfinite(reliability) or not (0.0 <= reliability <= 1.0):
                raise ModalityContractValidationError(f"reliability override must be a finite float in [0.0, 1.0], got {reliability!r}")
            rel = float(reliability)
        else:
            rel = self.frozen_reliability if self.frozen_reliability is not None else FROZEN_CLINICAL_RELIABILITY

        if clinical_output is None:
            return ModalityPrediction(
                modality="clinical",
                task_id=CLINICAL_TARGET.name,
                target=CLINICAL_TARGET,
                prediction={"class": None, "probabilities": [], "calibration_method": "isotonic_regression"},
                uncertainty={"method": "bootstrap_ensemble_dispersion", "value": 1.0, "interpretation": "unavailable"},
                quality=0.0,
                available=False,
                reliability=rel,
                model_version=self.model_version,
                legacy_projected_scalar=0.0,
            )

        # Support both ClinicalOutput schema and dictionary payload
        calib_p = None
        calib_method = "isotonic_regression"
        unc_val = None

        if isinstance(clinical_output, dict):
            calib_p = clinical_output.get("calibrated_probability")
            if calib_p is None:
                calib_p = clinical_output.get("calibrated_p")
            calib_method = clinical_output.get("calibration_method", "isotonic_regression")
            unc_obj = clinical_output.get("uncertainty")
            if isinstance(unc_obj, dict):
                unc_val = unc_obj.get("std_probability")
            elif isinstance(unc_obj, (int, float)):
                unc_val = float(unc_obj)
        else:
            calib_p = getattr(clinical_output, "calibrated_probability", None)
            if calib_p is None:
                calib_p = getattr(clinical_output, "calibrated_p", None)
            calib_method = getattr(clinical_output, "calibration_method", "isotonic_regression")
            unc_obj = getattr(clinical_output, "uncertainty", None)
            if unc_obj is not None:
                unc_val = getattr(unc_obj, "std_probability", None)
                if unc_val is None and isinstance(unc_obj, (int, float)):
                    unc_val = float(unc_obj)

        if calib_p is None:
            raise ModalityContractValidationError(
                "Clinical output must contain 'calibrated_probability' or 'calibrated_p'."
            )

        if not isinstance(calib_p, (int, float)) or isinstance(calib_p, bool) or not np.isfinite(calib_p) or calib_p < 0.0 or calib_p > 1.0:
            raise ModalityContractValidationError(
                f"calibrated readmission probability must be a finite float in [0.0, 1.0], got {calib_p!r}"
            )

        p1 = float(calib_p)
        p0 = float(1.0 - p1)
        norm_probs = [p0, p1]
        # Standard classification rule at nominal 0.5 boundary (diagnostic rule, not clinical action cutoff)
        pred_class = 1 if p1 >= 0.5 else 0

        # Require and validate uncertainty
        if unc_val is None:
            raise ModalityContractValidationError(
                "Clinical output must provide 'std_probability' within uncertainty payload."
            )
        if not isinstance(unc_val, (int, float)) or isinstance(unc_val, bool) or not np.isfinite(unc_val) or unc_val < 0.0 or unc_val > 0.5:
            raise ModalityContractValidationError(
                f"Clinical bootstrap uncertainty std_probability must be a finite float in [0.0, 0.5], got {unc_val!r}"
            )
        # 0.5 is the theoretical maximum standard deviation for a bounded Bernoulli probability
        norm_unc = float(unc_val / 0.5)

        # Traceable quality fallback
        if quality is not None:
            if not isinstance(quality, (int, float)) or isinstance(quality, bool) or not np.isfinite(quality) or not (0.0 <= quality <= 1.0):
                raise ModalityContractValidationError(f"quality must be a finite float in [0.0, 1.0], got {quality!r}")
            q_val = float(quality)
            q_prov = "measured"
        else:
            raw_m_ratio = None
            if hasattr(clinical_output, "shift_detection"):
                sd = clinical_output.shift_detection
                if hasattr(sd, "missingness_ratio"):
                    raw_m_ratio = sd.missingness_ratio
                elif isinstance(sd, dict):
                    raw_m_ratio = sd.get("missingness_ratio")
            elif isinstance(clinical_output, dict) and "shift_detection" in clinical_output:
                sd = clinical_output["shift_detection"]
                if isinstance(sd, dict):
                    raw_m_ratio = sd.get("missingness_ratio")

            raw_feat_comp = None
            if hasattr(clinical_output, "feature_completeness"):
                raw_feat_comp = clinical_output.feature_completeness
            elif isinstance(clinical_output, dict) and "feature_completeness" in clinical_output:
                raw_feat_comp = clinical_output["feature_completeness"]

            if raw_m_ratio is not None:
                if not isinstance(raw_m_ratio, (int, float)) or isinstance(raw_m_ratio, bool) or not np.isfinite(raw_m_ratio) or raw_m_ratio < 0.0 or raw_m_ratio > 1.0:
                    raise ModalityContractValidationError(
                        f"missingness_ratio must be a finite float in [0.0, 1.0], got {raw_m_ratio!r}"
                    )
                q_val = float(1.0 - raw_m_ratio)
                q_prov = "derived_from_shift_detection"
            elif raw_feat_comp is not None:
                if not isinstance(raw_feat_comp, (int, float)) or isinstance(raw_feat_comp, bool) or not np.isfinite(raw_feat_comp) or raw_feat_comp < 0.0 or raw_feat_comp > 1.0:
                    raise ModalityContractValidationError(
                        f"feature_completeness must be a finite float in [0.0, 1.0], got {raw_feat_comp!r}"
                    )
                q_val = float(raw_feat_comp)
                q_prov = "measured"
            else:
                q_val = 0.5  # Neutral fallback for unmeasured clinical quality
                q_prov = "unmeasured_neutral_fallback"

        return ModalityPrediction(
            modality="clinical",
            task_id=CLINICAL_TARGET.name,
            target=CLINICAL_TARGET,
            prediction={
                "class": pred_class,
                "probabilities": norm_probs,
                "calibration_method": calib_method,
                "quality_provenance": q_prov,
                "classification_note": "p1 >= 0.5 threshold represents diagnostic class assignment, not an action cutoff.",
            },
            uncertainty={
                "method": "bootstrap_ensemble_dispersion",
                "value": norm_unc,
                "raw_std_probability": float(unc_val),
                "theoretical_max_std": 0.5,
                "interpretation": "epistemic_readmission_dispersion",
            },
            quality=q_val,
            available=True,
            reliability=rel,
            model_version=self.model_version,
            legacy_projected_scalar=p1,
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

        if hasattr(self.clinical_service, "predict_encounter"):
            pred_output = self.clinical_service.predict_encounter(encounter_data)
        elif hasattr(self.clinical_service, "predict_single"):
            pred_output = self.clinical_service.predict_single(encounter_data)
        else:
            raise RuntimeError(
                "clinical_service must implement 'predict_encounter' or 'predict_single'."
            )
        return self.adapt_prediction_output(pred_output, quality=quality)
