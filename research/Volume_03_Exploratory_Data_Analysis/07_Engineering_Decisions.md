# Chapter 7: Engineering Analysis and Design Decisions

## Preprocessing Design Decisions & Trade-Offs
The Exploratory Data Analysis reveals that the APTOS 2019 dataset has high spatial and colorimetric variances. To ensure robust model development, several key engineering trade-offs were evaluated:

| Preprocessing Choice | Selected Option | Alternative Evaluated | Trade-Off Justification |
| :--- | :--- | :--- | :--- |
| **Retinal Scaling** | Aspect-Ratio-Preserving Padding | Raw Stretching (Squashing) | Squashing distorts circular fundus structures into ellipses, altering lesion features. Padding preserves spatial geometry. |
| **Image Normalization** | Candidate benchmark of ImageNet vs. Dataset stats | Fixed ImageNet Normalization | Dataset-specific color statistics ($[0.421, 0.224, 0.072]$) differ substantially from ImageNet stats ($[0.485, 0.456, 0.406]$). Empirical benchmarking is required. |
| **Augmentation Design** | Configurable augmentation parameter ranges | Hardcoded Fixed Augmentations | Later phases will experimentally optimize rotations and flip settings; hardcoding augmentations prevents tuning. |
| **Quality Assessment** | Weighted Quality Score ($Q$) column | Binary pass/fail exclusion filter | Excluding all flagged outliers could substantially reduce the effective training set and disproportionately affect minority classes. Gradient weighting preserves data. |
| **Duplicate Auditing** | Pixel-level SHA-256 hashing | Filename-based deduplication | Patients are sometimes imaged multiple times, resulting in duplicate pixel arrays under different filenames. Pixel-level SHA-256 hashing was used to identify exact duplicate image content independent of filenames. |
| **Analysis Resolution** | 512 px in-memory | Native resolution | Preserves global statistics while reducing computation time and memory consumption during EDA profiling. |

---

## Preprocessing Candidate Specifications
The parameters exported to `datasets/metadata/preprocessing_recommendations.json` are:

- **Image Sizes**: `[224, 384, 512]` pixels. Candidate resolutions span low-, medium-, and high-resolution operating points to evaluate the trade-off between computational cost and preservation of fine retinal lesions.
- **Augmentation Bounds**:
  - Horizontal Flips: `[True, False]`
  - Rotation Range: `[0, 15, 30]` degrees
  - Color Jitter: `[True, False]`
  - Random Crop: `[True, False]`
- **Quality Outliers**:
  - Dark Threshold: $< 37.782003$
  - Bright Threshold: $> 92.419779$
  - Sharpness Threshold: $< 24.368078$

---

## Quality-Weighted Loss Function
As a candidate training strategy, the continuous quality score $Q_i$ may be incorporated into the optimization objective as a quality-weighted loss function:

$$\mathcal{L}_Q = \frac{1}{\sum_{i=1}^N Q_i} \sum_{i=1}^N Q_i \cdot \mathcal{L}(y_i, \hat{y}_i)$$

Where:
- $\mathcal{L}(y_i, \hat{y}_i)$ is the cross-entropy loss for sample $i$.
- $Q_i \in [0, 1]$ acts as a sample-wise learning weight, down-weighting the loss of poor quality, dark, or blurry images during backpropagation. This formulation represents a candidate training strategy to be evaluated in ablation studies.

---

## Computational Complexity Analysis
The feature extraction pipeline (`extract_stats.py`) is designed for high-throughput execution:

- **Time Complexity**: 
  - Sequential execution: $\mathcal{O}(N \times H \times W)$
  - Practical parallel runtime: $\mathcal{O}\left(\frac{N \times H \times W}{P}\right)$ where $N$ is the number of images ($3,662$), $H \times W$ is the spatial dimensions, and $P$ is the number of worker threads ($16$). Downscaling to $512$ pixels in-memory limits the constant factor, achieving high throughput ($\approx 47.4$ images/sec).
- **Space Complexity**: $\mathcal{O}(P \times H_{\text{resized}} \times W_{\text{resized}})$ auxiliary space (ignoring the output metadata table, which occupies $\mathcal{O}(N)$ memory storage). Because images are loaded and downscaled dynamically, RAM usage is bounded by the number of active workers rather than the size of the dataset on disk, ensuring safety on memory-constrained hardware.

---

## References
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *CVPR*, 248-255.
- Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
- Martin, R. C. (2017). *Clean Architecture: A Craftsman's Guide to Software Structure and Design*. Prentice Hall.
- Pech-Pacheco, J. L., Cristóbal, G., Chamorro-Martinez, J., & Fernández-Valdivia, J. (2000). Diatom autofocusing in brightfield microscopy: a comparative study. *ICPR*, 561-564.
- Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data Augmentation for Deep Learning. *Journal of Big Data*, 6(1), 1-48.
