# Volume 08: Prediction Uncertainty Estimation

This volume details the characterization of prediction uncertainty for the calibrated EfficientNet-B3 Diabetic Retinopathy classifier on the APTOS 2019 Blindness Detection dataset.

---

## 1. Volume Index

1. **[01_Theory.md](01_Theory.md)**: Theoretical foundations of predictive uncertainty, expected entropy, mutual information, and predictive variance.
2. **[02_Uncertainty_Methods.md](02_Uncertainty_Methods.md)**: A detailed comparison of deterministic proxies vs. stochastic ensembling (MC Dropout).
3. **[03_Monte_Carlo_Dropout.md](03_Monte_Carlo_Dropout.md)**: Inspection of the EfficientNet-B3 dropout structure and selective dropout validation routines.
4. **[04_Uncertainty_Metrics.md](04_Uncertainty_Metrics.md)**: Detailed formulations and python implementations of uncertainty and risk-coverage metrics.
5. **[05_Experimental_Results.md](05_Experimental_Results.md)**: Quantitative results, error-detection performance tables, and convergence analysis.
6. **[06_Discussion.md](06_Discussion.md)**: Clinical implications of uncertainty-based selective prediction and grade confusion.
7. **[07_Limitations.md](07_Limitations.md)**: Limits on Out-of-Distribution validation, computational cost, and dropout placement.
8. **[08_Conclusion.md](08_Conclusion.md)**: Executive summary of findings.
9. **[09_Reproducibility.md](09_Reproducibility.md)**: Hardware, software, and data checksum logs for perfect replication.

---

## 2. Executive Summary of Objectives

The primary research question for this study is:

> **Does stochastic uncertainty estimation reliably identify incorrect predictions and enable clinical selective prediction?**

We run a comparative analysis to determine whether multi-pass stochastic uncertainty ensembling (MC Dropout) provides superior error-detection (AUROC/AUPRC) and selective prediction (AURC/E-AURC) compared to single-pass calibrated deterministic proxies.

**Out-of-Distribution (OOD) Scope**:
OOD infrastructure is implemented, but formal external OOD evaluation was not performed because no verified independent OOD dataset was available.

---

## 3. Verified Experimental Results

The Step 8 uncertainty estimation pipeline has been executed and verified:
- **Test Samples**: 367 retinal scans
- **Learned Temperature (restored from Step 7)**: 1.6218
- **MC Dropout Passes ($N$)**: 25
- **Probability Tensor Dimension**: 367 samples × 25 passes × 5 classes
- **MC Stochasticity Sanity Check**: Passed (non-zero variance verified across passes)
- **Primary Failure Detection Metric**: 
  - MC Predictive Variance achieved the highest AUROC of **0.8443** (and lowest AURC of **0.0406**).
  - Calibrated Confidence ($1 - C$) achieved a highly competitive baseline AUROC of **0.8381** (with AURC of **0.0418**).
- **Milestone Rejections**: At 50% coverage, calibrated confidence reduced remaining prediction risk (error rate) from **15.80%** down to **1.63%**.
- **Report Assets**: Includes Figures 1–10 (saved locally as PNGs), 6 tabular reports (saved locally as Markdown files), representative cases selection, and Grad-CAM mean intensity correlation values.

---

## 4. How to Run the Pipeline

To execute the uncertainty estimation pipeline, run the following command from the project root:

```powershell
python src/uncertainty.py --checkpoint experiments/efficientnet_b3/checkpoints/best_model.pt --model efficientnet_b3 --mc-passes 25 --save-plots
```

The script performs:
1. Dynamic calibration verification ($T = 1.6218$ loaded from Step 7).
2. Model weight and dataset freezing.
3. Running 5 stochastic sanity-check passes to verify active dropout.
4. Deterministic and 25-pass stochastic inference on the test split.
5. Computing error-detection metrics (AUROC/AUPRC) and selective prediction risk-coverage curves.
6. Plotting analysis figures in `results/uncertainty/figures/`.
7. Outputting structured summary tables in `results/uncertainty/tables/`.
