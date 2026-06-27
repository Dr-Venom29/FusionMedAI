# Chapter 8: Limitations

## Spatial Downsampling and Information Loss
- **Limitation**: High-resolution fundus images (up to $4,288 \times 2,848$ pixels) are commonly downscaled during model training to satisfy computational and memory constraints. Small lesions such as microaneurysms may occupy only a few pixels in the original image. When the image is downscaled, these lesions may become indistinguishable after interpolation.
- **Mitigation**: Multi-scale training benchmarks (spanning $224 \times 224$, $384 \times 384$, and $512 \times 512$ pixel dimensions) will be evaluated during the model development and training phase to identify the optimal tradeoff between spatial fidelity and training throughput.

---

## Single-Channel Grayscale Focus Estimation
- **Limitation**: Sharpness is estimated using the Variance of the Laplacian on single-channel grayscale conversions of the fundus images. Grayscale conversion ignores color-specific focus anomalies (such as chromatic aberrations). Furthermore, Laplacian variance measures image sharpness globally and does not distinguish whether diagnostically important retinal regions (such as the optic disc or macula) remain in focus.
- **Mitigation**: Future preprocessing modifications could calculate channel-wise Laplacian variances and incorporate spatial weightings to prioritize the sharpness of central retinal regions.

---

## Limitations of Exact Duplicate Detection
- **Limitation**: Visual duplicate detection is performed by computing SHA-256 hashes of the raw image pixel arrays. This method is highly sensitive to minor pixel variations. If the same eye image is present twice but with a 1-pixel shift, a different crop, or minor contrast variations, the SHA-256 hashes will differ, and the duplicate will not be flagged.
- **Mitigation**: Exact duplicates were successfully identified and logged. Near-duplicate detection could instead employ perceptual hashing, Structural Similarity Index Measure (SSIM) comparisons, local feature descriptors (SIFT/ORB), or learned embedding similarity.

---

## Quality Score Simplification
- **Limitation**: The proposed quality score ($Q$) is based on brightness, sharpness, and spatial resolution. It does not explicitly evaluate clinically relevant factors such as optic disc visibility, macular visibility, retinal field coverage, media opacity, or imaging artifacts. Consequently, a technically high-quality image may still have limited diagnostic value.
- **Mitigation**: Future work may integrate retinal quality assessment networks or no-reference image quality assessment models trained specifically for fundus photography to filter or weight training inputs.

---

## Statistical Inference
- **Limitation**: This exploratory phase focuses on descriptive statistics and visualization rather than hypothesis testing. Observed differences between classes should therefore be interpreted as exploratory findings rather than statistically validated effects.
- **Mitigation**: Future work may incorporate formal statistical testing (such as ANOVA, Kruskal-Wallis, or Mann-Whitney tests) to establish statistical significance for differences in brightness, sharpness, and RGB channels across disease stages.

---

## External Domain Generalization
- **Limitation**: The statistics and normalizations calculated in this phase are derived entirely from the APTOS 2019 training cohort. Differences in imaging devices, illumination conditions, acquisition protocols, and patient populations may alter image distributions. Applying these parameters to other cohorts (such as Messidor or IDRiD) without adjustment may lead to domain shifts that impair model generalization.
- **Mitigation**: Domain adaptation techniques or standard ImageNet normalizations will be evaluated to ensure generalization to external datasets.
