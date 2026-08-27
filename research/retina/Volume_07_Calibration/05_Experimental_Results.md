# 5. Experimental Results

The Retina Module's output probabilities were calibrated using Temperature Scaling. This post-hoc calibration was applied to the pre-trained EfficientNet-B3 baseline model on the validation subset.

## Key Findings

The following table summarizes the global effect of the optimization.

| Metric | Before Calibration | After Calibration | Improvement |
|:---|:---:|:---:|:---:|
| **Accuracy** | Unchanged | Unchanged | 0 |
| **ECE** | 0.1058 | 0.0668 | ↓ 36.9% |
| **NLL** | 0.7220 | 0.5827 | ↓ 19.3% |
| **Brier** | 0.0631 | 0.0582 | ↓ 7.8% |
| **Temperature** | — | 1.6218 | Learned |

### Explanation of Results

1. **Accuracy**: As mathematically guaranteed, the classification accuracy on the validation and test sets remained completely unchanged because scaling all logits by a single positive scalar `T` preserves their relative ordering.
2. **Temperature**: The optimal temperature `T` learned on the validation set was `1.6218`, indicating that the base model was indeed overconfident.
3. **Global Improvement**: Calibration successfully reduced Expected Calibration Error (ECE) from 0.1058 down to 0.0668, indicating probabilities are now highly reliable. Negative Log-Likelihood (NLL) and Brier Score similarly improved.
4. **MCE Behavior**: MCE increased from 0.3585 to 0.6295. Independent verification against `TorchMetrics` produced identical values. Analysis showed the worst calibration bin contained only a single correctly classified sample, demonstrating that the increased MCE resulted from sparse-bin behavior rather than global miscalibration.
5. **Reliability Diagram**: Visually, the reliability curve aligns much closer to the perfect calibration diagonal, changing from points below the diagonal to bars closely matching it.

Refer to `tables/comparison.md` and the generated PDF report `reports/calibration_report.pdf` for exact numerical logs.
