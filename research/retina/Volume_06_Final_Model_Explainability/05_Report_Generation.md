# 05 Report Generation

To finalize the qualitative analysis of the retinal module, FusionMedAI features an automated PDF generation pipeline built on `reportlab`, ensuring all outputs are immediately publication-ready.

## Metadata and Provenance

Every execution of the XAI pipeline produces an extensive `manifest.json`. This ensures absolute reproducibility by tracking:
*   Dataset version and checksums
*   Git commit hashes
*   CLI execution arguments
*   PyTorch and CUDA environment states
*   Model checkpoint SHA256 signatures

## Automated PDF Reports

The `src/xai/report_generator.py` script aggregates the metadata, the clinical observations, and the visual panels (Original, Heatmap, Overlay) to generate three specific reports:

| Report | Purpose |
|:---|:---|
| `summary.pdf` | Executive summary |
| `xai_gallery.pdf` | Complete gallery |
| `failure_analysis.pdf` | Incorrect predictions |

These reports are designed to serve as self-contained clinical reviews.

## Experimental Validation

The report generation pipeline was successfully validated on the complete APTOS 2019 test set.

Generated artifacts include:

*   `summary.pdf`
*   `xai_gallery.pdf`
*   `failure_analysis.pdf`

The reports collectively document representative predictions, model failures, metadata, reproducibility information, and qualitative Grad-CAM analysis.
