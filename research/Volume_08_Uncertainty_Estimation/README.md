# Volume 08: Prediction Uncertainty Estimation

This volume details the characterization of prediction uncertainty for the calibrated EfficientNet-B3 Diabetic Retinopathy classifier on the APTOS 2019 Blindness Detection dataset.

---

## 1. Volume Index

1. **[01_Theory.md](file:///d:/FusionMedAI/research/Volume_08_Uncertainty_Estimation/01_Theory.md)**: Theoretical foundations of predictive uncertainty, expected entropy, mutual information, and predictive variance.
2. **[02_Uncertainty_Methods.md](file:///d:/FusionMedAI/research/Volume_08_Uncertainty_Estimation/02_Uncertainty_Methods.md)**: A detailed comparison of deterministic proxies vs. stochastic ensembling (MC Dropout).
3. **[03_Monte_Carlo_Dropout.md](file:///d:/FusionMedAI/research/Volume_08_Uncertainty_Estimation/03_Monte_Carlo_Dropout.md)**: Inspection of the EfficientNet-B3 dropout structure and selective dropout validation routines.
4. **[04_Uncertainty_Metrics.md](file:///d:/FusionMedAI/research/Volume_08_Uncertainty_Estimation/04_Uncertainty_Metrics.md)**: Detailed formulations and python implementations of uncertainty and risk-coverage metrics.
5. **[05_Experimental_Results.md](file:///d:/FusionMedAI/research/Volume_08_Uncertainty_Estimation/05_Experimental_Results.md)**: Quantitative results, error-detection performance tables, and convergence analysis.
6. **[06_Discussion.md](file:///d:/FusionMedAI/research/Volume_08_Uncertainty_Estimation/06_Discussion.md)**: Clinical implications of uncertainty-based selective prediction and grade confusion.
7. **[07_Limitations.md](file:///d:/FusionMedAI/research/Volume_08_Uncertainty_Estimation/07_Limitations.md)**: Limits on Out-of-Distribution validation, computational cost, and dropout placement.
8. **[08_Conclusion.md](file:///d:/FusionMedAI/research/Volume_08_Uncertainty_Estimation/08_Conclusion.md)**: Executive summary of findings.
9. **[09_Reproducibility.md](file:///d:/FusionMedAI/research/Volume_08_Uncertainty_Estimation/09_Reproducibility.md)**: Hardware, software, and data checksum logs for perfect replication.

---

## 2. Executive Summary of Objectives

The primary research question for this study is:

> **Does stochastic uncertainty estimation reliably identify incorrect predictions and enable clinical selective prediction?**

We run a comparative analysis to determine whether multi-pass stochastic uncertainty ensembling (MC Dropout) provides superior error-detection (AUROC/AUPRC) and selective prediction (AURC/E-AURC) compared to single-pass calibrated deterministic proxies.

---

## 3. How to Run the Pipeline

To execute the uncertainty estimation pipeline, run the following command from the project root:

```powershell
python src/uncertainty.py --checkpoint experiments/efficientnet_b3/checkpoints/best_model.pt --model efficientnet_b3 --mc-passes 25 --save-plots
```

The script performs:
1. Dynamic calibration verification ($T = 1.6218$ loaded from Step 7).
2. Model weight and dataset freezing.
3. Spawning 5 stochastic check passes to verify active dropout layers.
4. Deterministic and 25-pass stochastic inference on the test split.
5. Computing error-detection metrics (AUROC/AUPRC) and selective prediction risk-coverage curves.
6. Plotting analysis figures in `results/uncertainty/figures/`.
7. Outputting structured summary tables in `results/uncertainty/tables/`.
