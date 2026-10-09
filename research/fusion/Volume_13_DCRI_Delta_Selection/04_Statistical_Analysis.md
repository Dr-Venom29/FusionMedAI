# Chapter 04 — Regime Stratification & Bootstrap Statistics

## 1. Modality Regime Stratification Analysis

To ensure that the uncertainty penalty multiplier $\delta$ behaves stably regardless of whether an encounter presents with single, dual, or triple modalities, we evaluate all 11 candidate $\delta$ values across all 7 active modality availability regimes:
- **Single-Modality Regimes ($M=1$)**: Retina Only (`R`), Foot Only (`F`), Clinical Only (`C`).
- **Dual-Modality Regimes ($M=2$)**: Retina-Foot (`RF`), Retina-Clinical (`RC`), Foot-Clinical (`FC`).
- **Triple-Modality Regime ($M=3$)**: Retina-Foot-Clinical (`RFC`).
- **Empty Edge Case ($M=0$)**: Fail-closed sentinel (`EMPTY`).

### Mean DCRI Across All 11 Candidates & Modality Regimes ($\overline{\text{DCRI}}$)

| Candidate $\delta$ | `R` ($M=1$) | `F` ($M=1$) | `C` ($M=1$) | `RF` ($M=2$) | `RC` ($M=2$) | `FC` ($M=2$) | `RFC` ($M=3$) | `EMPTY` ($M=0$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\delta = 0.00$** (`D00`) | $0.255037$ | $0.521907$ | $0.113468$ | $0.384661$ | $0.170669$ | $0.297771$ | $0.289900$ | `NO_MODALITY_AVAILABLE` |
| **$\delta = 0.05$** (`D05`) | $0.254968$ | $0.492211$ | $0.111536$ | $0.354898$ | $0.151322$ | $0.266070$ | $0.258203$ | `NO_MODALITY_AVAILABLE` |
| **$\delta = 0.10$** (`D10`, $\delta^*$) | **$0.254899$** | **$0.462515$** | **$0.109605$** | **$0.325135$** | **$0.131975$** | **$0.234370$** | **$0.226506$** | `NO_MODALITY_AVAILABLE` |
| **$\delta = 0.15$** (`D15`) | $0.254831$ | $0.432820$ | $0.107673$ | $0.295371$ | $0.112628$ | $0.202669$ | $0.194809$ | `NO_MODALITY_AVAILABLE` |
| **$\delta = 0.20$** (`D20`) | $0.254762$ | $0.403124$ | $0.105741$ | $0.265608$ | $0.093281$ | $0.170969$ | $0.163113$ | `NO_MODALITY_AVAILABLE` |
| **$\delta = 0.25$** (`D25`) | $0.254693$ | $0.373428$ | $0.103810$ | $0.235845$ | $0.073934$ | $0.139268$ | $0.131416$ | `NO_MODALITY_AVAILABLE` |
| **$\delta = 0.30$** (`D30`) | $0.254624$ | $0.343732$ | $0.101878$ | $0.206081$ | $0.054587$ | $0.107567$ | $0.099719$ | `NO_MODALITY_AVAILABLE` |
| **$\delta = 0.40$** (`D40`) | $0.254487$ | $0.284344$ | $0.098014$ | $0.146554$ | $0.015894$ | $0.044165$ | $0.036325$ | `NO_MODALITY_AVAILABLE` |
| **$\delta = 0.50$** (`D50`) | $0.254350$ | $0.224948$ | $0.094150$ | $0.087027$ | $-0.022800$ | $-0.019236$ | $-0.027068$ | `NO_MODALITY_AVAILABLE` |
| **$\delta = 0.75$** (`D75`) | $0.254006$ | $0.076467$ | $0.084491$ | $-0.061791$ | $-0.119534$ | $-0.177740$ | $-0.185552$ | `NO_MODALITY_AVAILABLE` |
| **$\delta = 1.00$** (`D100`) | $0.253662$ | $-0.072011$ | $0.074832$ | $-0.210609$ | $-0.216269$ | $-0.336244$ | $-0.344036$ | `NO_MODALITY_AVAILABLE` |

---

### Negative DCRI Proportion Across All 11 Candidates & Modality Regimes ($P(\text{DCRI} < 0)$)

| Candidate $\delta$ | `R` ($M=1$) | `F` ($M=1$) | `C` ($M=1$) | `RF` ($M=2$) | `RC` ($M=2$) | `FC` ($M=2$) | `RFC` ($M=3$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\delta = 0.00$** (`D00`) | $0.0\%$ | $0.0\%$ | $0.0\%$ | $0.0\%$ | $0.0\%$ | $0.0\%$ | $0.0\%$ |
| **$\delta = 0.05$** (`D05`) | $0.0\%$ | $0.0\%$ | $0.0\%$ | $0.0\%$ | $0.0\%$ | $0.0\%$ | $0.0\%$ |
| **$\delta = 0.10$** (`D10`, $\delta^*$) | **$0.0\%$** | **$2.2\%$** | **$0.0\%$** | **$4.4\%$** | **$7.6\%$** | **$8.2\%$** | **$7.4\%$** |
| **$\delta = 0.15$** (`D15`) | $0.0\%$ | $5.4\%$ | $0.0\%$ | $11.6\%$ | $18.4\%$ | $19.2\%$ | $18.0\%$ |
| **$\delta = 0.20$** (`D20`) | $0.0\%$ | $8.0\%$ | $0.0\%$ | $17.6\%$ | $25.2\%$ | $26.8\%$ | $24.4\%$ |
| **$\delta = 0.25$** (`D25`) | $0.0\%$ | $11.2\%$ | $0.0\%$ | $23.4\%$ | $31.4\%$ | $34.0\%$ | $28.8\%$ |
| **$\delta = 0.30$** (`D30`) | $0.0\%$ | $13.6\%$ | $0.0\%$ | $28.0\%$ | $38.4\%$ | $40.8\%$ | $35.0\%$ |
| **$\delta = 0.40$** (`D40`) | $0.0\%$ | $18.2\%$ | $0.0\%$ | $37.8\%$ | $50.8\%$ | $54.0\%$ | $46.8\%$ |
| **$\delta = 0.50$** (`D50`) | $0.0\%$ | $25.0\%$ | $0.0\%$ | $47.2\%$ | $61.0\%$ | $64.4\%$ | $61.6\%$ |
| **$\delta = 0.75$** (`D75`) | $0.0\%$ | $39.4\%$ | $0.0\%$ | $67.0\%$ | $74.2\%$ | $78.4\%$ | $77.2\%$ |
| **$\delta = 1.00$** (`D100`) | $0.0\%$ | $54.2\%$ | $0.0\%$ | $77.8\%$ | $81.6\%$ | $86.8\%$ | $82.6\%$ |

---

## 2. Paired Bootstrap Inference ($B=1000$)

To establish non-parametric confidence intervals for differences between candidate penalty values, $B=1000$ paired bootstrap resamples were evaluated across the frozen cohort ($N=500$, seed 115):

### Difference vs Zero-Penalty Baseline ($\Delta_{\text{DCRI}} = \text{DCRI}_0 - \text{DCRI}_\delta$)

| Comparison | Observed Paired Difference $\Delta_{\text{obs}}$ | Bootstrap Mean $\overline{\Delta_{\text{boot}}}$ | Bootstrap Std Error | 95% Bootstrap Percentile CI | Analytical Slope Agreement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $\text{DCRI}_0$ vs $\text{DCRI}_{0.05}$ | $+0.031697$ | $+0.031718$ | $0.000550$ | $[+0.030646, +0.032792]$ | Exact match ($0.05 \times \overline{U_{\text{sum}}}$) |
| **$\text{DCRI}_0$ vs $\text{DCRI}_{0.10}$ ($\delta^*$)** | **$+0.063394$** | **$+0.063436$** | **$0.001101$** | **$[+0.061292, +0.065584]$** | **Exact match ($0.10 \times \overline{U_{\text{sum}}}$)** |
| $\text{DCRI}_0$ vs $\text{DCRI}_{0.15}$ | $+0.095090$ | $+0.095154$ | $0.001651$ | $[+0.091938, +0.098376]$ | Exact match ($0.15 \times \overline{U_{\text{sum}}}$) |
| $\text{DCRI}_0$ vs $\text{DCRI}_{0.20}$ | $+0.126787$ | $+0.126872$ | $0.002201$ | $[+0.122583, +0.131168]$ | Exact match ($0.20 \times \overline{U_{\text{sum}}}$) |
| $\text{DCRI}_0$ vs $\text{DCRI}_{0.25}$ | $+0.158484$ | $+0.158590$ | $0.002751$ | $[+0.153230, +0.163959]$ | Exact match ($0.25 \times \overline{U_{\text{sum}}}$) |
| $\text{DCRI}_0$ vs $\text{DCRI}_{0.30}$ | $+0.190181$ | $+0.190308$ | $0.003302$ | $[+0.183876, +0.196751]$ | Exact match ($0.30 \times \overline{U_{\text{sum}}}$) |
| $\text{DCRI}_0$ vs $\text{DCRI}_{0.40}$ | $+0.253574$ | $+0.253744$ | $0.004402$ | $[+0.245167, +0.262335]$ | Exact match ($0.40 \times \overline{U_{\text{sum}}}$) |
| $\text{DCRI}_0$ vs $\text{DCRI}_{0.50}$ | $+0.316968$ | $+0.317181$ | $0.005503$ | $[+0.306458, +0.327919]$ | Exact match ($0.50 \times \overline{U_{\text{sum}}}$) |
| $\text{DCRI}_0$ vs $\text{DCRI}_{0.75}$ | $+0.475452$ | $+0.475771$ | $0.008254$ | $[+0.459688, +0.491879]$ | Exact match ($0.75 \times \overline{U_{\text{sum}}}$) |
| $\text{DCRI}_0$ vs $\text{DCRI}_{1.00}$ | $+0.633936$ | $+0.634361$ | $0.011005$ | $[+0.612917, +0.655838]$ | Exact match ($1.00 \times \overline{U_{\text{sum}}}$) |

> [!NOTE]
> **Interpretation Scope**: The observed paired mean difference $\Delta_{\text{obs}} = \frac{1}{N}\sum (\text{DCRI}_0 - \text{DCRI}_\delta)$ strictly equals the cohort mean penalty $\delta \overline{U_{\text{sum}}}$. A strictly positive bootstrap confidence interval ($\text{CI}_{\text{lower}} > 0$) confirms the mathematical presence and non-degeneracy of the uncertainty discount across the evaluated cohort. It serves as a verification of numerical consistency, and is **not** evidence of improved clinical risk prediction, enhanced disease calibration, or superior patient-level clinical outcomes.

---

## 3. Rank Stability & Ordering Invariance Across All 11 Candidates

To verify how the application of $\delta$ affects relative patient encounter rankings, we evaluate Spearman rank correlation $\rho_s$ and Kendall's $\tau$ between each candidate $\text{DCRI}_\delta$ and the unpenalized fused risk $R_{\text{fusion}}$:

| Candidate ID | $\delta$ Value | Spearman $\rho_s$ | Kendall's $\tau$ | $p$-value | Rank Preservation Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **D00** | $0.00$ | $1.000000$ | $1.000000$ | $0.0$ | Identical Ordering |
| **D05** | $0.05$ | $0.997040$ | $0.955271$ | $< 10^{-220}$ | Near-Perfect Preservation |
| **D10** | **$0.10$** | **$0.989286$** | **$0.911327$** | **$< 10^{-200}$** | **Robust Rank Fidelity ($\rho_s > 0.98$)** |
| **D15** | $0.15$ | $0.978032$ | $0.871054$ | $< 10^{-180}$ | High Rank Fidelity |
| **D20** | $0.20$ | $0.964053$ | $0.833299$ | $< 10^{-170}$ | Moderate Reordering |
| **D25** | $0.25$ | $0.948895$ | $0.799471$ | $< 10^{-150}$ | Moderate Reordering |
| **D30** | $0.30$ | $0.931644$ | $0.768080$ | $< 10^{-140}$ | Pronounced Reordering |
| **D40** | $0.40$ | $0.893963$ | $0.710774$ | $< 10^{-120}$ | Moderate Distortion |
| **D50** | $0.50$ | $0.854866$ | $0.662894$ | $< 10^{-100}$ | Substantial Distortion |
| **D75** | $0.75$ | $0.760086$ | $0.569475$ | $< 10^{-80}$ | Heavy Reordering |
| **D100** | $1.00$ | $0.681638$ | $0.496353$ | $< 10^{-60}$ | Severe Rank Scrambling |

