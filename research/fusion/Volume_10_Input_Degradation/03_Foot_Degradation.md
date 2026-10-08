# Diabetic Foot Ulcer Input Degradation Operators

## 1. Modality Degradation Taxonomy

The Diabetic Foot Ulcer photograph channel ($\mathcal{M}_{\text{Foot}}$) is evaluated under four deterministic degradation operators reflecting controlled photography challenges in wound assessment:

1. **D-F1: Gaussian Blur (`OP_FOOT_BLUR`)**  
   Simulates camera motion blur or close-up macro focusing failure, attenuating Sobel boundary gradient magnitude and edge clarity.
   
2. **D-F2: Contrast Attenuation (`OP_FOOT_CONTRAST`)**  
   Simulates poor ambient lighting or low-contrast skin tones by compressing dynamic range, directly degrading unsupervised Otsu Contrast-to-Noise Ratio (CNR).
   
3. **D-F3: Illumination Shift (`OP_FOOT_ILLUMINATION`)**  
   Simulates underexposure and uneven illumination across wound tissue via dynamic range scaling.
   
4. **D-F4: Synthetic Image Artifacts (`OP_FOOT_ARTIFACT`)**  
   Simulates dressing tape occlusion, specular flash reflections, or optical marker artifacts across the image via circular opaque masks.

---

## 2. Parameterization Across the 4-Level Severity Grid

| Severity Level | State Description | D-F1 Blur ($\sigma$, Kernel) | D-F2 Contrast Factor | D-F3 Illum Factor (`lum_factor`) | D-F4 Artifact Radius (px) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **D0** | Clean Baseline | $\sigma=0.0, k=1$ | $1.00$ | $1.00$ | $0\text{ px}$ |
| **D1** | Mild Degradation | $\sigma=1.5, k=5$ | $0.70$ | $0.75$ | $40\text{ px}$ |
| **D2** | Moderate Degradation | $\sigma=3.5, k=11$ | $0.45$ | $0.50$ | $70\text{ px}$ |
| **D3** | Severe Degradation | $\sigma=9.0, k=25$ | $0.20$ | $0.25$ | $100\text{ px}$ |

---

## 3. Signal Quality Layer Coupling

Degraded foot ulcer inputs $x_F^{(d)} = D_d(x_F)$ are evaluated directly by the frozen unsupervised quality engine:

$$
Q_F(x_F^{(d)}) = \frac{1}{2} \cdot \left[ q_{\text{cnr}}(x_F^{(d)}) + q_{\text{boundary}}(x_F^{(d)}) \right]
$$

where:
- $q_{\text{cnr}}$ calculates unsupervised Otsu foreground-background separation in $[1.8, 13.0]$.
- $q_{\text{boundary}}$ calculates Sobel edge gradient magnitude in $[15.0, 60.0]$.

