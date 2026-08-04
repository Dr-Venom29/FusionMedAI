# 01 XAI Architecture

The Explainable AI (XAI) pipeline in FusionMedAI is designed to extract, interpret, and report on the internal activations of the final selected retinal model (`EfficientNet-B3`). This pipeline is built to meet strict scientific standards, ensuring that model predictions are transparent and clinically interpretable before proceeding to the multimodal fusion stage (ACARA-U).

## Design Principles

1.  **Modularity**: The XAI pipeline is completely decoupled from the main training and benchmarking modules. All XAI-related code resides in the `src/xai/` package, allowing independent evolution.
2.  **Architecture Agnosticism**: Through the `get_target_layer()` utility, the pipeline dynamically resolves the appropriate convolutional feature map or transformer token representation based on the model architecture (`efficientnet`, `convnext`, `swin`).
3.  **No External Bloat**: To maximize control over the raw activation arrays and eliminate unnecessary dependencies (e.g., `pytorch-grad-cam`), the pipeline uses custom, from-scratch implementations of standard CAM algorithms.
4.  **Scientific Rigor**: The pipeline avoids overclaiming diagnostic capability (e.g., "The model identified microaneurysms") and instead strictly reports on spatial attention heuristics (e.g., "Attention spans multiple distinct focal regions").

## Architecture Flow

```text
Image
  ↓
EfficientNet-B3
  ↓
Inference
  ↓
GradCAM
  ↓
Clinical Interpreter
  ↓
Representative Selector
  ↓
PDF Reports
```

## Pipeline Components

*   `src/xai.py`: The main CLI entrypoint orchestrating the pipeline.
*   `src/xai/cam.py`: The abstract base class handling hook registration and heatmap normalization.
*   `src/xai/inference.py`: Executes batched inference, capturing confidence, Shannon Entropy, and Top-2 Margin.
*   `src/xai/selector.py`: A non-deterministic, multi-criteria sampling module for representative case selection.
*   `src/xai/clinical_interpreter.py`: The heuristic engine mapping pixel activations to clinical observations.
*   `src/xai/visualization.py`: Generates the composite visual panels.
*   `src/xai/report_generator.py`: Compiles the generated assets into publication-quality PDFs using `reportlab`.

The output is a complete suite of CSV predictions, raw Numpy arrays, image overlays, and PDF reports designed for direct inclusion in a dissertation or clinical review.

## Validation Status

The complete XAI architecture has been executed end-to-end on the finalized EfficientNet-B3 checkpoint.

All pipeline components successfully generated their expected outputs without manual intervention.
