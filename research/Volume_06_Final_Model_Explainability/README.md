# Volume 06: Final Model Explainability (XAI)

This volume documents the implementation, evaluation, and results of the Explainable AI (XAI) pipeline developed for the final **EfficientNet-B3 Retina Module**.

## Purpose

The primary objective of this phase is to improve the transparency and interpretability of the trained retinal classification model by investigating the visual evidence supporting its predictions. Rather than treating the network as a "black box," the XAI pipeline verifies that the model primarily attends to clinically meaningful retinal regions instead of spurious image artifacts, borders, or background features. This improves model transparency and supports future clinical adoption.

## Pipeline

The explainability pipeline consists of the following automated stages:

1. **Inference** – Generate predictions for the complete test dataset.
2. **Grad-CAM Generation** – Produce visual attention maps from the final convolutional layer of the selected EfficientNet-B3 model.
3. **Clinical Interpretation** – Analyze the spatial distribution and intensity of activation maps using rule-based heuristics.
4. **Representative Case Selection** – Perform reproducible diversified sampling to select representative correct predictions, failure cases, high-confidence predictions, low-confidence predictions, and challenging examples for qualitative analysis.
5. **Automated Report Generation** – Generate publication-ready visual reports and supporting metadata.

## Outputs

This phase generates:

- Grad-CAM heatmaps and overlay visualizations.
- Representative explainability examples for qualitative evaluation.
- Comprehensive CSV files containing predictions, confidence scores, entropy, and spatial attention metrics.
- Metadata and reproducibility artifacts.
- Executive summary, XAI gallery, and failure analysis PDF reports.

## Folder Organization

- **`figures/`** – Representative Grad-CAM visualizations, failure cases, and summary statistics.
- **`reports/`** – Automatically generated publication-ready PDF reports.
- **`01_...` → `08_...`** – Detailed documentation covering methodology, Grad-CAM implementation, representative selection, clinical interpretation, experimental results, limitations, and conclusions.

## Next Steps

The explainability outputs generated in this volume provide qualitative validation of the Retina Module and complement the probability calibration stage (Volume 07). Together, these components establish the foundation for **Volume 08: Uncertainty Estimation**, where predictive confidence and uncertainty will be quantified before final Retina Module integration.