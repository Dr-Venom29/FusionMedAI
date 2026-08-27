# 07 Limitations

While the Explainable AI (XAI) pipeline successfully illuminates the internal attention mechanisms of the EfficientNet-B3 model, it has several inherent limitations that must be acknowledged:

1. **Grad-CAM is Qualitative**: Class Activation Mapping provides qualitative visual evidence of spatial attention. It does not provide rigorous, quantitative bounds on feature importance.
2. **Attention ≠ Explanation**: Just because a model attends to a specific pixel does not mean it "understands" the physiological significance of that pixel. The model is merely exploiting statistical correlations.
3. **Heuristic Observations**: The Clinical Interpreter relies on geometric and intensity-based heuristics (e.g., centroid calculation, thresholding). These heuristics are robust proxies but cannot replace true semantic segmentation or clinical diagnosis.
4. **Single Model**: The XAI pipeline was executed exclusively on the final EfficientNet-B3 model. Comparative explainability across the other tested architectures (ConvNeXt, Swin) is not provided.
5. **No Clinician Validation**: The generated spatial heuristics and visual overlays have not been validated by a board-certified ophthalmologist in a clinical trial setting.
6. **No Pixel-Level Segmentation**: The model identifies broad regions of interest but does not produce pixel-perfect masks of individual microaneurysms or hemorrhages.
