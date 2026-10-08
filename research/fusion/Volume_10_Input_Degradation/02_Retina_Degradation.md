# Retinal Fundus Input Degradation Operators

## 1. Modality Degradation Taxonomy

The retinal fundus image channel ($\mathcal{M}_{\text{Retina}}$) is subjected to four deterministic degradation operators acting as controlled experimental proxies for acquisition-related image quality loss:

1. **D-R1: Gaussian Blur (`OP_RETINA_BLUR`)**  
   Serves as a controlled proxy for acquisition-related loss of high-frequency image information, including blur-like degradation. Attenuates spatial gradients and raw Laplacian variance $s_{\text{raw}}$.
   
2. **D-R2: Contrast Attenuation (`OP_RETINA_CONTRAST`)**  
   Serves as a controlled proxy for reduced dynamic range and hazy optics by compressing pixel luminance around the image mean.
   
3. **D-R3: Illumination Shift (`OP_RETINA_ILLUMINATION`)**  
   Serves as a controlled proxy for underexposure by shifting mean luminance into the non-optimal dark penalty zone ($\mu_i < 35.0$).
   
4. **D-R4: Synthetic Occlusion Artifacts (`OP_RETINA_ARTIFACT`)**  
   Serves as a controlled proxy for optical path obstructions and sensor occlusions via deterministic opacity patches.

---

## 2. Parameterization Across the 4-Level Severity Grid

| Severity Level | State Description | D-R1 Blur ($\sigma$, Kernel) | D-R2 Contrast Factor | D-R3 Illum Shift ($\Delta \mu$) | D-R4 Artifact Radius |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **D0** | Clean Baseline | $\sigma=0.0, k=1$ | $1.00$ | $0$ | $r=0\text{ px}$ |
| **D1** | Mild Degradation | $\sigma=1.5, k=5$ | $0.70$ | $-25$ | $r=45\text{ px}$ |
| **D2** | Moderate Degradation | $\sigma=3.5, k=11$ | $0.45$ | $-45$ | $r=75\text{ px}$ |
| **D3** | Severe Degradation | $\sigma=7.0, k=21$ | $0.20$ | $-60$ | $r=110\text{ px}$ |


---

## 3. Signal Quality Layer Coupling

Degraded images $x_R^{(d)} = D_d(x_R)$ are evaluated directly by the frozen unsupervised quality engine:

$$
Q_R(x_R^{(d)}) = \frac{1}{2} \cdot \left[ q_{\text{sharp}}(x_R^{(d)}) + q_{\text{illum}}(x_R^{(d)}) \right]
$$

where:
- $q_{\text{sharp}}$ measures normalized Laplacian variance in $[4.0, 55.0]$.
- $q_{\text{illum}}$ measures composite mean luminance and dynamic range standard deviation.

The quality layer operates strictly without disease labels or model predictions.
