# 8. Conclusion

This volume presented an empirical study characterizing the prediction uncertainty of a calibrated EfficientNet-B3 classifier on the APTOS 2019 test split, comparing deterministic uncertainty proxies against stochastic ensembling via Monte Carlo (MC) Dropout.

---

## 8.1 Summary of Key Findings

1. **Selective Prediction is Viable**:
   - The risk-coverage curves demonstrate a clear downward trend: as the most uncertain samples are rejected, the error rate (risk) on the remaining retained samples decreases. This validates the use of uncertainty thresholds for clinical referral workflows.
2. **Deterministic Proxies are Competitive**:
   - Calibrated deterministic proxies (specifically calibrated confidence and normalized calibrated entropy) are highly effective at detecting model errors. Post-hoc temperature scaling ($T \approx 1.6218$) successfully aligns raw confidence with empirical accuracy, making single-pass inference highly reliable for failure detection.
3. **Stochastic Decompositions Provide Clinical Value**:
   - MC Dropout successfully decomposes uncertainty into aleatoric (data ambiguity) and epistemic (parameter disagreement) components. Mutual Information (MI) identifies borderline grade-confusion cases (adjacent Diabetic Retinopathy stages) and provides a safety layer for detecting silent failures.
4. **Computational Latency is the Primary Constraint**:
   - MC Dropout requires $N=25$ forward passes, introducing a 25-fold latency penalty. For clinical screening deployments, deterministic calibrated confidence remains the preferred operational baseline due to its $O(1)$ efficiency. However, MC Dropout serves as a powerful offline auditing tool for identifying clinical ambiguity.

---

## 8.2 Future Research Directions

To build upon the findings of Volume 8, future work should focus on:
- **Out-of-Distribution (OOD) Audits**: Testing the epistemic uncertainty (Mutual Information) metrics against verified OOD datasets (e.g., non-retinal images or scans containing other macular pathologies).
- **Stochastic Depth Backbones**: Instantiating models with internal dropout or stochastic depth layers embedded within the convolutional backbone to enable deep stochastic representation perturbations.
- **Dynamic Compute Pathways**: Developing cascade screening pipelines that perform single-pass deterministic inference first, and only trigger multi-pass stochastic ensembling if the deterministic confidence is borderline.
