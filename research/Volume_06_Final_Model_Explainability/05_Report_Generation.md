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

1.  **`summary.pdf`**: An executive overview containing the core `manifest.json` metadata, high-level metrics (e.g., average entropy, average confidence), and a curated selection of the Top 10 representative images.
2.  **`xai_gallery.pdf`**: A complete, comprehensive gallery documenting every single selected representative case, complete with auto-scaling figures, page numbers, and captions.
3.  **`failure_analysis.pdf`**: A subset report specifically filtering for cases where the model predicted incorrectly. This is a crucial document for understanding model blind spots (e.g., severe confusion across grades) before integrating the retinal module into the ACARA-U multimodal fusion engine.

## Experimental Validation

The report generation pipeline was successfully validated on the complete APTOS 2019 test set.

Generated artifacts include:

*   `summary.pdf`
*   `xai_gallery.pdf`
*   `failure_analysis.pdf`

The reports collectively document representative predictions, model failures, metadata, reproducibility information, and qualitative Grad-CAM analysis.
