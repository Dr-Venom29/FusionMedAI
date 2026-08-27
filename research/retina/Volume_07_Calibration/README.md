# Volume 07: Retinal Model Calibration

This directory contains the complete documentation, experimental results, and artifacts for **Step 7** of the FusionMedAI project: **Post-Hoc Probability Calibration**.

## Overview

Deep neural networks, including the selected **EfficientNet-B3** retinal backbone, are often overconfident in their predictions. Although classification accuracy may be high, the raw softmax probabilities frequently fail to represent the true likelihood of correctness.

In this phase, we implemented **Temperature Scaling**, a post-hoc calibration technique that learns a single scalar parameter (T) using the validation set to calibrate the model's confidence estimates. The underlying EfficientNet-B3 weights remain completely frozen throughout this process, ensuring that predictive accuracy is preserved while improving probability calibration.

## Directory Structure

- **`01_Theory.md`**: Theoretical background on model calibration and prediction overconfidence.
- **`02_Metrics.md`**: Explanation of Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), Brier Score, and Maximum Calibration Error (MCE).
- **`03_Temperature_Scaling.md`**: Mathematical formulation and implementation details of Temperature Scaling.
- **`04_Reliability_Diagrams.md`**: Reliability diagrams, calibration curves, and confidence distribution visualizations before and after calibration.
- **`05_Experimental_Results.md`**: Quantitative evaluation and comparison of calibration performance.
- **`06_Discussion.md`**: Discussion of the results, design decisions, and relevance to medical AI.
- **`07_Limitations.md`**: Limitations of Temperature Scaling and future improvements.
- **`08_Conclusion.md`**: Summary of the calibration phase and transition to Step 8.
- **`09_Reproducibility.md`**: Experiment tracking, generated artifacts, and reproducibility documentation.
- **`figures/`**: Reliability diagrams, calibration curves, confidence histograms, and supporting visualizations.
- **`tables/`**: CSV and Markdown tables containing calibration metrics and quantitative comparisons.
- **`reports/`**: Automatically generated publication-ready PDF reports summarizing the calibration experiments.

## Current Status

**Volume 07 – Probability Calibration: ✅ Complete**

Probability calibration is complete and frozen for the Retina Module. The learned Temperature parameter ($T \approx 1.6218$) was integrated into the unified Retina inference engine to scale logits inside each MC pass before softmax.

The next development phase is the **Foot Ulcer Module**, which will calibrate its own classification scores using temperature scaling.