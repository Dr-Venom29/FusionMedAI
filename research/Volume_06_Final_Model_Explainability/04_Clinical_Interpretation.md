# 04 Clinical Interpretation

A common pitfall in Deep Learning medical imaging is "overclaiming" based on visual attention. A Grad-CAM heatmap highlighting a red spot does not necessarily mean the model understands it is a microaneurysm. 

FusionMedAI addresses this limitation directly by ensuring the automated interpretation of CAMs remains strictly objective and mathematically grounded in spatial heuristics.

## Spatial Heuristics Engine

The `src/xai/clinical_interpreter.py` module evaluates the continuous CAM output by binarizing it at a 50% max intensity threshold and calculating specific morphological features:

1.  **Connected Components**: Measures the number of distinct focal regions the model's attention is distributed across.
    *   *Interpretation*: "Attention is highly localized to a single region" vs. "Attention spans multiple distinct focal regions."
2.  **Relative Centroids**: Computes image moments to find the center of mass of the model's attention relative to the image dimensions.
    *   *Interpretation*: "Attention concentrated near the central (macular) region" vs. "Attention overlaps peripheral retina."
3.  **Area Dispersion Ratio**: Calculates the percentage of the image covered by the primary attention mask.
    *   *Interpretation*: "Attention is widely dispersed across retinal tissue."
4.  **Average Activation Intensity**: Extracts the mean value of the attention mask to determine peak confidence within the localized area.
    *   *Interpretation*: "Extremely high activation intensity within focal regions."

This strictly heuristic approach ensures that all generated reports are scientifically defensible for dissertation review, avoiding unfounded claims about internal model reasoning.

## Execution Result

The heuristic interpreter successfully generated observations for every representative case included in the XAI reports.
