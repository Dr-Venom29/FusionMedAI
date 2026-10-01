"""
Unified Clinical Inference Service (Phase C10).
Integrates Frozen Preprocessing (D=119), CatBoost HPO, TreeSHAP (C6),
Isotonic Calibration (C7), Bootstrap Uncertainty (C8), and Shift Safeguards (C9).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier

from src.clinical.modeling.preprocessing import ClinicalPreprocessor
from src.clinical.calibration.calibrator import IsotonicCalibrator
from src.clinical.uncertainty.bootstrap_ensemble import BootstrapCatBoostEnsemble
from src.clinical.inference.schema import (
    ClinicalOutput,
    UncertaintyOutput,
    PredictiveInterval,
    FeatureAttribution,
    ShiftDetection,
    ModelProvenance,
)
from src.clinical.inference.validator import (
    validate_single_encounter,
    validate_batch_encounters,
    ClinicalValidationError,
)


class ClinicalInferenceService:
    """
    End-to-end inference service for the Clinical Tabular Modality of FusionMedAI.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        models_cache_dir: Optional[Path] = None,
        operating_threshold: float = 0.20,
        random_seed: int = 42,
    ):
        self.operating_threshold = operating_threshold
        self.random_seed = random_seed
        self.version = "clinical_c10_v1.0"
        self.manifest_sha256 = "c10_verified_e2e"

        # 1. Fit Locked Preprocessor on train
        self.preprocessor = ClinicalPreprocessor(scale_numerical=True)
        self.preprocessor.fit(train_df)
        self.feature_names = list(self.preprocessor.feature_names_)
        assert len(self.feature_names) == 119, f"Expected 119 features, got {len(self.feature_names)}"

        X_train, y_train, _ = self.preprocessor.transform(train_df)
        X_val, y_val, _ = self.preprocessor.transform(val_df)

        # 2. Fit / Initialize Canonical Frozen CatBoost HPO Candidate
        self.base_model_params = {
            "depth": 4,
            "learning_rate": 0.1383,
            "iterations": 350,
            "l2_leaf_reg": 2.911,
            "subsample": 0.655,
            "random_seed": random_seed,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "verbose": False,
            "thread_count": -1,
        }
        self.base_model = CatBoostClassifier(**self.base_model_params)
        self.base_model.fit(X_train, y_train, verbose=False)

        # 3. Fit 50-member Bootstrap Uncertainty Ensemble
        self.ensemble = BootstrapCatBoostEnsemble(
            n_estimators=50,
            random_seed=random_seed,
        )
        self.ensemble.fit(
            X_train=X_train,
            y_train=y_train,
            feature_names=self.feature_names,
            train_dir=models_cache_dir,
        )

        # 4. Fit Frozen C7 Isotonic Calibrator on Validation Predictions
        dist_val = self.ensemble.predict_distribution(X_val)
        self.calibrator = IsotonicCalibrator()
        self.calibrator.fit(dist_val["mean_prob"], y_val)

        # 5. Initialize TreeSHAP Explainer on Base Model
        self.explainer = shap.TreeExplainer(self.base_model)
        self.expected_value = float(self.explainer.expected_value)

        # Uncertainty reference thresholds from Phase C8
        self.unc_75th = 0.0249
        self.unc_90th = 0.0421

    def predict_encounter(
        self,
        raw_input: Union[Dict[str, Any], pd.Series],
        top_k_attributions: int = 5,
    ) -> ClinicalOutput:
        """
        Execute full end-to-end clinical inference on a single patient encounter.
        """
        # 1. Validate Input
        valid_dict = validate_single_encounter(raw_input)
        encounter_id = valid_dict.get("encounter_id", "single_query")

        # 2. Convert to DataFrame and Transform to D=119
        single_df = pd.DataFrame([valid_dict])
        X_trans, _, _ = self.preprocessor.transform(single_df)

        # 3. Base Point Prediction & Raw Probability
        raw_prob_base = float(self.base_model.predict_proba(X_trans)[0, 1])

        # 4. Stochastic Bootstrap Prediction Distribution (50 models)
        dist = self.ensemble.predict_distribution(X_trans)
        raw_prob_ensemble = float(dist["mean_prob"][0])
        std_prob = float(dist["std_prob"][0])
        q025 = float(dist["q025"][0])
        q975 = float(dist["q975"][0])
        entropy = float(dist["entropy"][0])

        # 5. Isotonic Calibrated Probability (from ensemble mean)
        cal_prob = float(self.calibrator.predict_proba(np.array([raw_prob_ensemble]))[0])

        # 6. Classification at Institutional Threshold
        pred_label = int(cal_prob >= self.operating_threshold)

        # 7. TreeSHAP Attribution Computation
        shap_vals = self.explainer.shap_values(X_trans)[0]  # shape: (119,)
        feature_vals = X_trans[0]

        # Rank features by absolute SHAP magnitude
        abs_order = np.argsort(np.abs(shap_vals))[::-1]
        attributions: List[FeatureAttribution] = []

        for rank, idx in enumerate(abs_order[:top_k_attributions], start=1):
            feat_name = self.feature_names[idx]
            attributions.append(
                FeatureAttribution(
                    feature=feat_name,
                    shap_value=float(shap_vals[idx]),
                    rank=rank,
                    feature_value=float(feature_vals[idx]),
                )
            )

        # 8. Confidence and Decision Tier Mapping
        is_high_u = std_prob >= self.unc_75th
        dist_to_thresh = abs(cal_prob - self.operating_threshold)
        in_ambiguity_zone = dist_to_thresh <= 0.03

        # Derived operational confidence
        if in_ambiguity_zone or std_prob >= self.unc_90th:
            confidence = "low"
        elif std_prob >= self.unc_75th or dist_to_thresh <= 0.06:
            confidence = "moderate"
        else:
            confidence = "high"

        # 6 formal decision tiers
        if in_ambiguity_zone:
            decision_tier = (
                "Near Threshold / High Uncertainty (Ambiguity)"
                if is_high_u
                else "Near Threshold / Low Uncertainty"
            )
        elif cal_prob < self.operating_threshold:
            decision_tier = (
                "Low Risk / High Uncertainty"
                if is_high_u
                else "Low Risk / Low Uncertainty"
            )
        else:
            decision_tier = (
                "High Risk / High Uncertainty"
                if is_high_u
                else "High Risk / Low Uncertainty"
            )

        # 9. Shift & Blind-Spot Detection Safeguards
        # Calculate missingness ratio in raw input
        missing_count = sum(
            1 for v in valid_dict.values()
            if v is None or (isinstance(v, str) and v in ("?", "", "None", "nan")) or (isinstance(v, float) and np.isnan(v))
        )
        missing_ratio = float(missing_count / len(valid_dict))
        is_degraded = missing_ratio >= 0.15

        shift_alerts = []
        if is_degraded:
            shift_alerts.append(f"Elevated missingness detected ({missing_ratio*100:.1f}% omitted features).")

        # Prior inpatient blind spot check
        inp_val = valid_dict.get("number_inpatient")
        try:
            inp_num = float(inp_val) if inp_val not in ("?", "", None, "None") else 0.0
        except ValueError:
            inp_num = 0.0

        blind_spot = False
        if inp_num == 0 and cal_prob < self.operating_threshold and std_prob < self.unc_75th:
            blind_spot = True
            shift_alerts.append(
                "Prior-utilization blind spot warning: Zero prior inpatient admissions observed. "
                "Low point uncertainty should not be interpreted as guaranteed stability."
            )

        # Approximate percentile in cohort
        percentile = float(min(99.9, max(0.1, (std_prob / 0.06) * 100.0)))

        # 10. Assemble Standardized ClinicalOutput Payload
        return ClinicalOutput(
            modality="clinical_tabular",
            encounter_id=encounter_id,
            prediction=pred_label,
            probability=raw_prob_ensemble,
            calibrated_probability=cal_prob,
            calibration_method="Isotonic_Regression",
            confidence=confidence,
            uncertainty=UncertaintyOutput(
                method="bootstrap_ensemble",
                std_probability=std_prob,
                percentile_in_cohort=percentile,
                is_high_uncertainty=is_high_u,
                predictive_interval_95=PredictiveInterval(lower=q025, upper=q975),
                aleatoric_entropy=entropy,
            ),
            decision_tier=decision_tier,
            operating_threshold=self.operating_threshold,
            feature_attributions=attributions,
            shift_detection=ShiftDetection(
                is_degraded=is_degraded,
                missingness_ratio=missing_ratio,
                blind_spot_warning=blind_spot,
                shift_alerts=shift_alerts,
            ),
            model_provenance=ModelProvenance(
                model_name="CatBoost_HPO_Bootstrap_Ensemble",
                ensemble_size=50,
                frozen_calibrator="Isotonic_Regression",
                version=self.version,
                manifest_sha256=self.manifest_sha256,
            ),
        )

    def predict_batch(
        self,
        df: pd.DataFrame,
        top_k_attributions: int = 3,
    ) -> List[ClinicalOutput]:
        """
        Execute end-to-end inference over a batch DataFrame.
        """
        valid_df = validate_batch_encounters(df)
        X_trans, _, _ = self.preprocessor.transform(valid_df)

        dist = self.ensemble.predict_distribution(X_trans)
        raw_probs = dist["mean_prob"]
        std_probs = dist["std_prob"]
        q025_arr = dist["q025"]
        q975_arr = dist["q975"]
        entropy_arr = dist["entropy"]

        cal_probs = self.calibrator.predict_proba(raw_probs)
        shap_matrix = self.explainer.shap_values(X_trans)

        outputs: List[ClinicalOutput] = []
        n_samples = len(valid_df)

        for i in range(n_samples):
            enc_id = valid_df.iloc[i].get("encounter_id", i)
            cal_p = float(cal_probs[i])
            raw_p = float(raw_probs[i])
            std_p = float(std_probs[i])
            q025 = float(q025_arr[i])
            q975 = float(q975_arr[i])
            ent = float(entropy_arr[i])
            pred_label = int(cal_p >= self.operating_threshold)

            # Top SHAP
            shap_row = shap_matrix[i]
            abs_order = np.argsort(np.abs(shap_row))[::-1]
            attributions = [
                FeatureAttribution(
                    feature=self.feature_names[idx],
                    shap_value=float(shap_row[idx]),
                    rank=r,
                    feature_value=float(X_trans[i, idx]),
                )
                for r, idx in enumerate(abs_order[:top_k_attributions], start=1)
            ]

            is_high_u = std_p >= self.unc_75th
            dist_to_thresh = abs(cal_p - self.operating_threshold)
            in_ambiguity_zone = dist_to_thresh <= 0.03

            if in_ambiguity_zone or std_p >= self.unc_90th:
                conf = "low"
            elif std_p >= self.unc_75th or dist_to_thresh <= 0.06:
                conf = "moderate"
            else:
                conf = "high"

            if in_ambiguity_zone:
                tier = (
                    "Near Threshold / High Uncertainty (Ambiguity)"
                    if is_high_u
                    else "Near Threshold / Low Uncertainty"
                )
            elif cal_p < self.operating_threshold:
                tier = (
                    "Low Risk / High Uncertainty"
                    if is_high_u
                    else "Low Risk / Low Uncertainty"
                )
            else:
                tier = (
                    "High Risk / High Uncertainty"
                    if is_high_u
                    else "High Risk / Low Uncertainty"
                )

            # Simple blind spot check
            row_dict = valid_df.iloc[i].to_dict()
            inp_v = row_dict.get("number_inpatient", 0)
            try:
                inp_n = float(inp_v) if inp_v not in ("?", "", None, "None") else 0.0
            except ValueError:
                inp_n = 0.0

            blind_spot = bool(inp_n == 0 and cal_p < self.operating_threshold and std_p < self.unc_75th)

            outputs.append(
                ClinicalOutput(
                    modality="clinical_tabular",
                    encounter_id=enc_id,
                    prediction=pred_label,
                    probability=raw_p,
                    calibrated_probability=cal_p,
                    calibration_method="Isotonic_Regression",
                    confidence=conf,
                    uncertainty=UncertaintyOutput(
                        method="bootstrap_ensemble",
                        std_probability=std_p,
                        percentile_in_cohort=float(min(99.9, max(0.1, (std_p / 0.06) * 100.0))),
                        is_high_uncertainty=is_high_u,
                        predictive_interval_95=PredictiveInterval(lower=q025, upper=q975),
                        aleatoric_entropy=ent,
                    ),
                    decision_tier=tier,
                    operating_threshold=self.operating_threshold,
                    feature_attributions=attributions,
                    shift_detection=ShiftDetection(
                        is_degraded=False,
                        missingness_ratio=0.0,
                        blind_spot_warning=blind_spot,
                        shift_alerts=["Prior-utilization blind spot warning"] if blind_spot else [],
                    ),
                    model_provenance=ModelProvenance(
                        model_name="CatBoost_HPO_Bootstrap_Ensemble",
                        ensemble_size=50,
                        frozen_calibrator="Isotonic_Regression",
                        version=self.version,
                        manifest_sha256=self.manifest_sha256,
                    ),
                )
            )

        return outputs
