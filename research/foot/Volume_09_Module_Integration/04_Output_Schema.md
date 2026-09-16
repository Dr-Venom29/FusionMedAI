# Chapter 04 — Unified Output Schema & Contract Specification

## 1. Output Schema Definition

The return dictionary of `FootModule.predict()` adheres to the unified FusionMedAI modality contract:

```json
{
  "modality": "foot",
  "prediction": 1,
  "prediction_label": "Grade 2",
  "raw_confidence": 0.6241,
  "calib_confidence": 0.6889,
  "raw_probabilities": [0.2152, 0.6241, 0.1251, 0.0356],
  "calib_probabilities": [0.1928, 0.6889, 0.0915, 0.0268],
  "calib_entropy_norm": 0.6690,
  "calib_margin": 0.4961,
  "mc_passes_N": 10,
  "mc_predictive_entropy": 0.9275,
  "mc_predictive_entropy_norm": 0.6690,
  "mc_expected_entropy": 0.9244,
  "mc_expected_entropy_norm": 0.6668,
  "mc_predictive_variance": 0.000435,
  "mc_mutual_information": 0.003083,
  "cam_overlay": "<ndarray uint8 (224, 224, 3)>",
  "cam_heatmap": "<ndarray uint8 (224, 224, 3)>",
  "cam_mean_intensity": 0.2422,
  "latency_ms": 687.81,
  "model_info": {
    "name": "EfficientNet-B3",
    "calibration": "vector_scaling",
    "stochastic_passes_N": 10
  }
}
```

---

## 2. Field Descriptions & Types

| Field Name | Type | Value Range / Units | Description |
| :--- | :---: | :---: | :--- |
| `modality` | `str` | `"foot"` | Modality identifier for fusion layer routing |
| `prediction` | `int` | `0, 1, 2, 3` | Predicted Wagner grade class index |
| `prediction_label` | `str` | `"Grade 1" ... "Grade 4"` | Human-readable Wagner grade label |
| `raw_confidence` | `float` | $[0.0, 1.0]$ | Maximum uncalibrated softmax probability |
| `calib_confidence` | `float` | $[0.0, 1.0]$ | Maximum Vector Scaled probability |
| `raw_probabilities` | `List[float]` | Length 4, $\sum = 1.0$ | Uncalibrated class probability vector |
| `calib_probabilities` | `List[float]` | Length 4, $\sum = 1.0$ | Calibrated class probability vector |
| `calib_entropy_norm` | `float` | $[0.0, 1.0]$ | Calibrated entropy normalized by $\log(4)$ |
| `calib_margin` | `float` | $[0.0, 1.0]$ | Difference between top 2 calibrated probabilities |
| `mc_passes_N` | `int` | $\ge 0$ | Number of stochastic passes executed ($N^*=10$) |
| `mc_predictive_entropy` | `float` | $\ge 0.0$ nats | Total predictive entropy $H(\bar{p})$ |
| `mc_predictive_entropy_norm`| `float` | $[0.0, 1.0]$ | Normalized total entropy $H(\bar{p}) / \log(4)$ |
| `mc_expected_entropy` | `float` | $\ge 0.0$ nats | Aleatoric expected entropy $\mathbb{E}[H(p)]$ |
| `mc_expected_entropy_norm` | `float` | $[0.0, 1.0]$ | Normalized aleatoric expected entropy |
| `mc_predictive_variance` | `float` | $\ge 0.0$ | Mean probability variance across passes |
| `mc_mutual_information` | `float` | $\ge 0.0$ nats | Epistemic mutual information $MI$ |
| `cam_overlay` | `ndarray` / `None` | `(224, 224, 3) uint8` | Blended RGB image and heatmap overlay |
| `cam_heatmap` | `ndarray` / `None` | `(224, 224, 3) uint8` | Pure JET color attribution heatmap |
| `cam_mean_intensity` | `float` | $[0.0, 1.0]$ | Average spatial attribution intensity |
| `latency_ms` | `float` | $> 0.0$ ms | Total end-to-end execution time in ms |
| `model_info` | `Dict` | — | Model name, calibrator, and pass metadata |
