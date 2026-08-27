# Chapter 9: Limitations

This chapter documents the limitations of the current dataset preparation phase and explains why certain steps are deferred to later stages of the project.

---

## Documented Limitations

### 1. Only APTOS 2019 Integrated
- **Limitation**: The dataset preparation pipeline has been validated only on the APTOS 2019 dataset. Statistical observations are limited to the APTOS 2019 dataset, and dataset-specific preprocessing recommendations may require recalibration when applied to external retinal datasets.
- **Reason for Deferral**: Focusing on a single dataset first allows us to build, test, and refine the verification and metadata pipelines before extending them to other clinical modalities.

### 2. No External Validation Dataset
- **Limitation**: No external dataset (such as IDRiD) has been integrated for cross-dataset validation.
- **Reason for Deferral**: External validation represents a subsequent evaluation milestone. First, a reliable internal training, validation, and testing partition scheme must be established.

### 3. Advanced Retinal Image Enhancement Not Yet Applied
- **Limitation**: Advanced clinical enhancements (such as Ben Graham's processing, circular cropping, contrast-limited adaptive histogram equalization (CLAHE), or illumination correction) have not yet been applied to the fundus photographs.
- **Reason for Deferral**: While basic resizing and normalization are applied, advanced enhancement will be evaluated as an experimental variable during model optimization to determine its impact on class separation.

### 4. Image Augmentation Parameters Not Yet Optimized
- **Limitation**: The project includes a configurable augmentation pipeline. However, augmentation parameters have not yet been optimized experimentally.
- **Reason for Deferral**: Augmentation policies (rotation angles, jitter scales) must be tuned during model training rather than dataset ingestion, ensuring they can be evaluated against model convergence rates.

### 5. ImageNet Normalization Statistics Used
- **Limitation**: Image normalization currently uses ImageNet channel mean ($\mu=[0.485, 0.456, 0.406]$) and standard deviation ($\sigma=[0.229, 0.224, 0.225]$) parameters instead of dataset-specific retinal statistics.
- **Reason for Deferral**: Pretrained weights (ImageNet-pretrained backbones) expect ImageNet-normalized inputs during early model training. Computing custom retinal statistics is deferred to later optimization stages.

### 6. No Patient-Level Split Guarantee
- **Limitation**: Because the APTOS 2019 dataset does not include explicit patient identifiers, patient-level separation cannot be guaranteed. Splitting is performed at the image level, introducing a risk of intra-patient data leakage if a single patient has images in both training and validation sets.
- **Reason for Deferral**: This is a structural limitation of the public source metadata. Future work using dataset sources with patient IDs will enforce patient-level disjointness to prevent potential intra-patient data leakage.

### 7. Advanced Explainability Methods
- **Limitation**: Standard Grad-CAM and Grad-CAM++ were implemented in Volume 6. More advanced explainability techniques (Integrated Gradients, SHAP, Score-CAM, Layer-CAM) remain future work.
- **Reason for Deferral**: Explainability infrastructure has been prepared through feature extraction support in the model wrapper class, but more advanced visualization techniques remain future work.

### 8. Uncertainty Estimation
- **Limitation**: Uncertainty estimation was not part of the original dataset preparation phase.
- **Resolution in Later Volumes**: Prediction uncertainty was subsequently implemented and evaluated in Volume 8 using MC Dropout, predictive entropy, expected entropy, mutual information, predictive variance, error-detection analysis, risk-coverage analysis, AURC/E-AURC, and Monte Carlo convergence analysis.
- **Remaining Limitation**: The current uncertainty evaluation remains specific to the finalized Retina Module and does not yet establish cross-dataset uncertainty robustness.

### 9. No Out-of-Distribution (OOD) Detection
- **Limitation**: Images are assumed to originate from the APTOS distribution. External validation and out-of-distribution robustness remain future research directions.
- **Reason for Deferral**: OOD detection and cross-dataset shift checks are deferred to subsequent development volumes to isolate baseline model development from robustness testing.

### 10. Clinical Annotation Uncertainty and Label Noise
- **Limitation**: The annotations in public clinical datasets are subject to grader-dependent label noise and inter-grader variability. Automated verification validates label boundaries but cannot identify or correct clinical misclassifications.
- **Reason for Deferral**: Clinical validation is outside the scope of automated dataset pipelines. We assume the annotations in the public dataset are correct and focus on ensuring their structural integrity.

### 11. Camera and Acquisition Device Variability
- **Limitation**: Images in the dataset originate from different acquisition devices, clinical environments, and camera models, resulting in high variability in contrast, illumination, resolution, and field of view.
- **Reason for Deferral**: Preprocessing (resizing, normalization) helps mitigate this variability, but advanced domain standardization techniques are deferred to model optimization.

### 12. Cross-Dataset Domain Shift
- **Limitation**: APTOS images represent a different clinical cohort than IDRiD or other external datasets, introducing domain-shift scenarios.
- **Reason for Deferral**: Cross-dataset shift and out-of-distribution (OOD) checks are deferred to subsequent development volumes to isolate baseline model development from robustness testing.

---

## Conclusion
The limitations identified in this chapter were historically deferred to maintain modularity during the dataset preparation phase. Subsequent volumes successfully resolved many of these limitations—including model benchmarking, explainability, calibration, predictive uncertainty estimation, and Retina Module integration—while cross-dataset robustness, OOD detection models, and multi-modal aggregation represent upcoming fusion engineering goals.
