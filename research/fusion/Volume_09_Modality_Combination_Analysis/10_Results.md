# Phase C11.9 Comprehensive Experimental Results

## 1. Executive Summary of Scientific Findings

Phase C11.9 evaluated whether the frequency distribution of modality combinations impacts the behavior, numerical stability, and derived risk-index behavior of ACARA-U. Key findings across 1,500 controlled decision packet trials ($N=500$ across distributions D1, D2, D3) include:

1. **Simplex Invariant Preservation (100.0%)**: Across all head, middle, and tail combinations, active routing weights strictly normalized to unity ($\sum w_i = 1.000000$), while unavailable channels received exactly zero authority ($w_i = 0.000000$).
2. **Global Metric Stability Under Distribution Shift**:
   - Population Mean Risk $\overline{R_{\text{fusion}}}$: $0.2841$ (D1 Balanced) $\to 0.2909$ (D2 Moderate Tail) $\to 0.3005$ (D3 Strong Tail).
   - Population Mean DCRI $\overline{\text{DCRI}}$: $0.2096$ (D1) $\to 0.1953$ (D2) $\to 0.1952$ (D3).
   - Global Shannon Entropy $\overline{H(w)}$: $0.4154$ (D1) $\to 0.6662$ (D2) $\to 0.7737$ (D3).
3. **Point-Estimate Tail Sensitivity Across Soft Baselines**:
   - ACARA-U achieved the lowest observed point-estimate tail sensitivity ($D_{\text{tail}} = 0.1883$ in D2, $0.1876$ in D3) among the evaluated soft-weighting baselines (Uniform Averaging $0.1925 / 0.1945$, Confidence Weighting $0.1959 / 0.1974$, Reliability+Confidence $0.1967 / 0.1973$, Conf+Rel-Uncertainty $0.1889 / 0.1889$). The margin over B5 was small ($+0.0006$ in D2, $+0.0012$ in D3; paired bootstrap CIs crossing zero).
   - Winner-take-all baseline B1 achieved lower nominal tail sensitivity by exclusively selecting Retina, but suffered higher head-tier risk dispersion ($\sigma = 0.2755$ in D2 vs $0.1880$ for ACARA-U).
4. **Uncertainty Aggregation by Modality Cardinality**: Total uncertainty generally increased with the number of available channels because $U_{\text{sum}}$ is additive, although individual combinations were not strictly ordered by cardinality.
5. **Deterministic Fail-Closed Boundary Safety**: Zero-modality inputs ($\text{EMPTY}$) unconditionally return `NO_MODALITY_AVAILABLE` with $R_{\text{fusion}} = 0.0$ and $\text{DCRI} = 0.0$.

---

## 2. Cross-Distribution Summary Table

| Distribution Name | Cohort Head / Mid / Tail | Mean Risk $\overline{R_{\text{fusion}}}$ | Mean $\overline{\text{DCRI}}$ | Mean Entropy $\overline{H(w)}$ | Mean $U_{\text{sum}}$ | Tail Sensitivity $D_{\text{tail}}$ | Tail Risk Std $\sigma(R_{\text{tail}})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D1_BALANCED** | $500$ / $0$ / $0$ | $0.2841 \pm 0.2232$ | $0.2096 \pm 0.2202$ | $0.4154 \pm 0.3808$ | $0.3725 \pm 0.3515$ | $0.0000$ | $0.0000$ |
| **D2_MODERATE_HEAD_TAIL** | $300$ / $125$ / $75$ | $0.2909 \pm 0.1948$ | $0.1953 \pm 0.2065$ | $0.6662 \pm 0.3340$ | $0.4784 \pm 0.3168$ | $0.1883 \pm 0.1423$ | $0.2508$ |
| **D3_STRONG_LONG_TAIL** | $375$ / $90$ / $35$ | $0.3005 \pm 0.2001$ | $0.1952 \pm 0.2119$ | $0.7737 \pm 0.3045$ | $0.5264 \pm 0.2974$ | $0.1876 \pm 0.1551$ | $0.2784$ |

---

## 3. Conclusions on Scientific Viability

Across the controlled D1–D3 decision-packet distributions, ACARA-U preserved the active-authority simplex, maintained zero authority for unavailable modalities, and exhibited stable routing behavior under increasing combination skew. Tail configurations showed greater observed risk dispersion and sensitivity to modality reduction, while head-tier mean risk remained similar between the D2 and D3 controlled distributions. These findings characterize decision-level robustness under controlled modality-frequency shift; they do not establish clinical robustness or patient-level multimodal validity.
