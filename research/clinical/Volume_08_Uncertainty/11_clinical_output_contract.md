# Document 11: Multimodal ClinicalOutput Schema & Interface Contract

## 1. Architectural Role

The final multimodal architecture of **FusionMedAI** requires the Clinical Tabular Modality to output a rich, standardized payload (`ClinicalOutput`) consumed by downstream multimodal fusion blocks (ACARA-U).

Phase C8 formalizes the exact schema, cleanly separating **derived operational confidence** from **quantitative bootstrap uncertainty dispersion**.

---

## 2. Standardized JSON Output Contract

```json
{
  "modality": "clinical_tabular",
  "encounter_id": 11556,
  "prediction": 1,
  "probability": 0.7913,
  "calibrated_probability": 0.7333,
  "calibration_method": "Isotonic_Regression",
  "confidence": "moderate",
  "uncertainty": {
    "method": "bootstrap_ensemble",
    "std_probability": 0.1138,
    "percentile_in_cohort": 98.4,
    "is_high_uncertainty": true,
    "predictive_interval_95": {
      "lower": 0.5384,
      "upper": 0.9575
    },
    "aleatoric_entropy": 0.7382
  },
  "decision_tier": "High Risk / High Uncertainty",
  "operating_threshold": 0.20,
  "feature_attributions": [
    {"feature": "number_inpatient", "shap_value": 0.8412, "rank": 1},
    {"feature": "age_ordinal", "shap_value": 0.1420, "rank": 2},
    {"feature": "time_in_hospital", "shap_value": 0.0954, "rank": 3}
  ],
  "model_provenance": {
    "model_name": "CatBoost_HPO_Bootstrap_Ensemble",
    "ensemble_size": 50,
    "frozen_calibrator": "Isotonic_Regression",
    "version": "clinical_c8_v1.0",
    "manifest_sha256": "4b9e18f29ac..."
  }
}
```

---

## 3. Downstream Interface Fields Description

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `prediction` | `int (0/1)` | Binary risk classification at institutional threshold $\theta$. |
| `probability` | `float [0, 1]` | Raw ensemble mean readmission risk. |
| `calibrated_probability` | `float [0, 1]` | Post-hoc calibrated probability via Phase C7 Isotonic calibrator. |
| `confidence` | `string` | **Derived operational tier** (`"high"`, `"moderate"`, `"low"`), mapped from risk level, operating threshold distance, and uncertainty threshold. |
| `uncertainty.method` | `string` | Method used to quantify dispersion (`"bootstrap_ensemble"`). |
| `uncertainty.std_probability` | `float >= 0` | **Quantitative dispersion**: Standard deviation of predictions across bootstrap models ($\sigma_p$). |
| `uncertainty.predictive_interval_95` | `object` | 2.5th and 97.5th percentiles of ensemble prediction distribution. |
| `decision_tier` | `string` | Ambiguity and CDS protocol category (e.g., "Near Threshold / High Uncertainty"). |
| `feature_attributions` | `array` | Top local TreeSHAP attributions connecting patient factors to predicted risk. |
| `model_provenance` | `object` | Ensemble size, frozen calibrator, model version, and cryptographic manifest checksum. |
