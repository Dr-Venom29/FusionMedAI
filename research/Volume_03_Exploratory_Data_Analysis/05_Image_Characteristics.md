# Chapter 5: Image Characteristics

## Spatial Resolution and Aspect Ratios
The APTOS 2019 training cohort exhibits high spatial dimension variations due to different camera makes and cropping methods:

- **Minimum Resolution**: $474 \times 358$ pixels (low-resolution screening images)
- **Maximum Resolution**: $4,288 \times 2,848$ pixels (high-resolution camera sensors)
- **Average Width**: $2,015.18$ pixels
- **Average Height**: $1,526.83$ pixels
- **Average Aspect Ratio**: $1.2831$ (approximately consistent with a $4:3$ imaging aspect ratio commonly produced by retinal fundus cameras)
- **Orientation Consistency**: The dataset contains $0$ portrait-oriented images (width < height); all $3,662$ files are landscape format.

---

## Spatial Scaling Engineering Implications
These spatial properties present critical challenges for the image preprocessing pipeline in Phase 4:

1. **Aspect Ratio Preservation**:
   Standard image scaling that forces images into a square (e.g., $224 \times 224$) causes spatial distortion. Isotropic anatomical structures such as the retinal field may become geometrically distorted.
   - *Engineering Recommendation*: Resizing must implement aspect-ratio-preserving constant-value padding (black padding) or center-cropping to maintain circular retinal geometry.

2. **Resolution Reduction and Feature Loss**:
   Downscaling high-resolution images ($4,288 \times 2,848$) to $224 \times 224$ represents a significant reduction:
   $$\text{Pixel Reduction} = \frac{4288 \times 2848 - 224 \times 224}{4288 \times 2848} \times 100 \approx 99.59\%$$
   Extremely small features, such as microaneurysms (which may be only $5\text{--}15$ pixels wide in raw images), may become indistinguishable after interpolation.
   - *Engineering Recommendation*: Multi-scale training benchmarks using different target resolutions should be evaluated to measure accuracy-speed trade-offs:

| Candidate Resolution | Advantages | Disadvantages |
| :---: | :--- | :--- |
| **$224 \times 224$** | Fastest training, lowest memory footprint | Loss of fine localized lesions |
| **$384 \times 384$** | Better lesion preservation, moderate speed | Increased GPU memory requirements |
| **$512 \times 512$** | Highest spatial fidelity, minimal feature loss | Slowest training, highest GPU memory usage |

---

## Global and Class-wise RGB Channel Analysis
The global channel means and standard deviations (normalized to the $[0, 1]$ range) are:
- **Global RGB Mean**: $\mu_{\text{global}} = [0.421598, 0.223786, 0.072202]$
- **Global RGB Std**: $\sigma_{\text{global}} = [0.276549, 0.150057, 0.080490]$

The observed color distribution is consistent with retinal fundus photography, where vascular tissue, retinal pigmentation, illumination characteristics, and camera sensor responses collectively produce a dominant red channel and comparatively lower green and blue channel intensities. These statistics will serve as candidate normalization parameters during preprocessing and will be compared against ImageNet normalization in controlled ablation experiments.

### Class-wise RGB Breakdown
To evaluate whether disease progression causes colorimetric shifts, channel statistics were computed across the five classes:

| Class Staging | Red Mean | Green Mean | Blue Mean | Red Std Dev | Green Std Dev | Blue Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Class 0** (No DR) | $0.416368$ | $0.222873$ | $0.068884$ | $0.292754$ | $0.159137$ | $0.087143$ |
| **Class 1** (Mild NPDR) | $0.452164$ | $0.222137$ | $0.059349$ | $0.249443$ | $0.129467$ | $0.056948$ |
| **Class 2** (Moderate NPDR) | $0.429409$ | $0.227916$ | $0.081488$ | $0.260751$ | $0.141538$ | $0.075325$ |
| **Class 3** (Severe NPDR) | $0.397003$ | $0.219236$ | $0.081376$ | $0.246883$ | $0.138934$ | $0.071396$ |
| **Class 4** (Proliferative DR) | $0.411181$ | $0.221699$ | $0.075962$ | $0.250298$ | $0.139111$ | $0.072566$ |

The class-wise color analysis shows that only modest variations in channel statistics are observed across disease stages, suggesting that diabetic retinopathy progression is associated more strongly with localized pathological lesions than with global color distribution changes.

Refer to the figures copied to the research volume:
- [Resolution Histogram Analysis](images/Fig_03_03_resolution_histogram.png): Figures 3.3–3.6 demonstrate the large variability in spatial dimensions.
- [Resolution Scatter Plot](images/Fig_03_04_resolution_scatter.png)
- [Aspect Ratio Boxplot](images/Fig_03_05_aspect_ratio_histogram.png)
- [File Size Histogram](images/Fig_03_06_filesize_histogram.png)
- [Overlay of RGB Channel Histograms](images/Fig_03_09_rgb_histogram.png): Figure 3.9 confirms the stable RGB distribution across the dataset.
- [Class-wise Resolution and File Size Metrics](images/Fig_03_12_classwise_resolution.png)
