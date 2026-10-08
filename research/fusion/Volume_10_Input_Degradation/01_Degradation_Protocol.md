# Phase C11.10 Research Protocol & Pre-Specified Questions

## 1. Scientific Context & Core Research Question

Phase C11.8 established router safety under modality absence ($A_i = 0$), and Phase C11.9 analyzed robustness across combination frequency distributions. Phase C11.10 evaluates quality-aware decision robustness when a modality remains technically available ($A_i=1$) but undergoes progressive signal degradation, producing reduced quality $Q_i$ and protocol-defined uncertainty response $U_i$.

The primary research question is:

> **When an available prediction channel suffers progressive input degradation while remaining technically available ($A_i = 1$), does the frozen ACARA-U router reduce its assigned decision authority ($w_i \downarrow$) in response to the degradation, and does this behavior remain safer and more responsive than simpler soft-weighting baselines?**

---

## 2. Seven Pre-Specified Research Questions

1. **RQ1 (Experiment A Quality Detection)**: Does the unsupervised quality measurement layer systematically detect progressive input degradation ($\Delta Q_i \le 0$) across image (blur, contrast, illumination, synthetic artifacts) and tabular EHR feature corruptions?
2. **RQ2 (Experiment B Authority Attenuation)**: Does the frozen ACARA-U router ($z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$) attenuate decision authority ($w_i \downarrow$) as modality quality degrades?
3. **RQ3 (Authority Redistribution Conservation)**: Is the authority surrendered by a degraded channel ($\Delta w_i < 0$) conservatively absorbed by remaining available channels ($\sum_{j \ne i} \Delta w_j = -\Delta w_i$) without violating simplex normalization ($\sum_{i \in \mathcal{A}} w_i = 1.0$)?
4. **RQ4 (Quality-Authority Slope & Alignment)**: What is the empirical response slope $S_{QW} = \Delta w_i / \Delta Q_i$ and Spearman rank alignment $\rho(Q_i, w_i)$ across progressive severity levels?
5. **RQ5 (Packet-Level Monotonicity)**: What fraction of individual decision packets satisfy monotonic authority decay ($w^{(0)} \ge w^{(1)} \ge w^{(2)} \ge w^{(3)}$) across the degradation ladder?
6. **RQ6 (Baseline Isolation B6 vs B5)**: Does ACARA-U (B6: $C+R-U+Q$) exhibit greater degradation responsiveness than the uncertainty-only ablation baseline (B5: $C+R-U$), providing evidence for an incremental contribution of $Q_i$ within the controlled degradation benchmark?
7. **RQ7 (Hard-Mask Safety Invariance)**: Does perturbing or degrading an unavailable modality channel ($A_i = 0$) produce exactly zero change ($\Delta w_{\text{active}} = 0, \Delta R_{\text{fusion}} = 0$) in the active routing weights and fused risk output?

---

## 3. Pre-Specified Scientific Hypotheses

- **Hypothesis H1 (Quality Degradation Detection)**:  
  *Prediction*: $Q_i(D_{k+1}) \le Q_i(D_k)$ across severity ladder $D0 \to D1 \to D2 \to D3$ across all 12 operators.  
  *Status*: **Supported** (Monotonic quality decay observed across all 12 operators).

- **Hypothesis H2 (Quality-Aware Authority Attenuation)**:  
  *Prediction*: ACARA-U reduces decision authority ($w_i \downarrow$) on increasingly degraded modalities under frozen router coefficients.  
  *Status*: **Supported** (Mean relative authority reduction $\text{RAR}_i \approx 35\text{--}50\%$ under severe degradation D3).

- **Hypothesis H3 (Authority Redistribution Conservation)**:  
  *Prediction*: Authority lost by degraded channels is completely conserved and absorbed by remaining active modalities ($\sum_{j \ne i} \Delta w_j = -\Delta w_i$).  
  *Status*: **Confirmed** (Exact simplex normalization $\sum_{i \in \mathcal{A}} w_i = 1.000000$ verified across all trials).

- **Hypothesis H4 (Quality-Routing Alignment & Positive Slope)**:  
  *Prediction*: Quality-Authority Response Slope $S_{QW} = \Delta w_i / \Delta Q_i > 0$ and Spearman correlation $\rho(Q_i, w_i) > 0$.  
  *Status*: **Supported** (All slopes positive, Spearman $\rho = 1.0000$).

- **Hypothesis H5 (Packet-Level Monotonicity)**:  
  *Prediction*: Over $90\%$ of packets satisfy monotonic authority reduction across the 4-level severity ladder.  
  *Status*: **Supported** ($100.0\%$ of packets satisfy monotonic ordering).

- **Hypothesis H6 (B5 vs B6 Quality Isolation)**:  
  *Prediction*: ACARA-U (B6) achieves a larger authority reduction on degraded channels than Baseline B5 ($\Delta w_{\text{B6}} < \Delta w_{\text{B5}}$), demonstrating an incremental routing contribution of $Q_i$ beyond $C_i, R_i, U_i$.  
  *Status*: **Supported** (B6 showed greater authority attenuation than B5 under the tested degradation conditions, paired difference $D = -0.1309$, $95\%$ bootstrap CI $[-0.1319, -0.1300]$).

> **Statistical Interpretation Rule**: Bootstrap confidence intervals are used to characterize directional uncertainty under the controlled benchmark. Exclusion of zero is interpreted as evidence of a consistent directional effect within this experimental cohort; it is not interpreted as clinical efficacy or external-population significance.

---

## 4. Methodological Boundaries & Transparent Scope
 
- **Controlled Synthetic Raw-Input Benchmark**: Evaluations operate on procedurally constructed benchmark image inputs and 119-D structured EHR feature vectors coupled to frozen quality engines ($Q_R, Q_F, Q_C$) and the frozen ACARA-U router ($N=500, \text{seed}=115$). The benchmark establishes decision-level routing responsiveness and quality sensitivity under controlled degradation; it does **not** evaluate patient-level diagnostic accuracy or clinical robustness on actual multimodal clinical patient datasets.
- **Protocol-Defined Uncertainty Response**: The benchmark recomputes input quality $Q_i$ live from degraded raw synthetic inputs while preserving the frozen decision-level uncertainty perturbation protocol ($U_i^{(d)}$) rather than rerunning upstream deep learning and gradient boosted tree models on every synthetic frame.
- **Strictly Frozen Architecture**: Modality backbones (Retina EfficientNet-B3, Foot EfficientNet-B3, Clinical CatBoost), calibrators, reliability priors ($R_R=0.929956, R_F=0.922266, R_C=0.825382$), router coefficients ($\alpha=1.0, \beta=1.5, \gamma=1.0, \eta=0.5$), and provisional DCRI $\delta=0.20$ remain frozen.
- **Zero Parameter Retraining**: No router coefficients or quality weighting functions are post-hoc tuned.
- **Deterministic Seeding**: The benchmark uses SHA-256-derived deterministic seeds for process-independent pseudo-random input generation and bootstrap resampling, eliminating Python `PYTHONHASHSEED` dependence.
- **Participating Modality Integrity**: Unperturbed risk projections $r_i$ are preserved while signal quality $Q_i$ and uncertainty $U_i$ are systematically modulated.


