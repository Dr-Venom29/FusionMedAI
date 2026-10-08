# Experiment A: Clean Fusion Benchmark Across Conditions B0–B5

## 1. Experimental Setup

Experiment A evaluates the decision-level fusion behavior on the clean baseline cohort ($N=500, \text{seed}=115, \text{D0}$) across six canonical experimental conditions:

- **B0**: Uncalibrated Uniform Fusion ($w_i = 1/M$)
- **B1**: Uncalibrated Reliability-Selected Fusion
- **B2**: Uncalibrated ACARA-U Dynamic Fusion ($z_i = \alpha C_i^{\text{raw}} + \beta R_i - \gamma U_i + \eta Q_i$)
- **B3**: Calibrated Uniform Fusion ($w_i = 1/M$)
- **B4**: Calibrated Reliability-Selected Fusion
- **B5**: Calibrated ACARA-U Dynamic Fusion ($z_i = \alpha C_i^{\text{cal}} + \beta R_i - \gamma U_i + \eta Q_i$)

---

## 2. Decision-Level Cohort Performance Summary

| Condition | Calibration State | Mean $w_R$ | Mean $w_F$ | Mean $w_C$ | Mean $R_{\text{fusion}}$ ($\sigma$) | Mean $\text{DCRI}$ ($\sigma$) | Mean $H(w)$ | Conflict Index | Dominant Modality (R / F / C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0: Uniform** | Uncalibrated | $0.3333$ | $0.3333$ | $0.3333$ | $0.2947$ ($0.1268$) | $0.1309$ ($0.1680$) | $1.0986$ | $0.5368$ | $100\% \text{ Tied}$ |
| **B1: Reliability** | Uncalibrated | $1.0000$ | $0.0000$ | $0.0000$ | $0.2437$ ($0.2768$) | $0.1309$ ($0.1680$) | $0.0000$ | $0.5368$ | $100.0\% / 0\% / 0\%$ |
| **B2: ACARA-U** | Uncalibrated | $0.4529$ | $0.1968$ | $0.3503$ | $0.2563$ ($0.1471$) | $0.1309$ ($0.1680$) | $1.0288$ | $0.5368$ | $88.0\% / 1.4\% / 10.6\%$ |
| **B3: Uniform** | Calibrated | $0.3333$ | $0.3333$ | $0.3333$ | $0.2968$ ($0.1227$) | $0.1333$ ($0.1619$) | $1.0986$ | $0.5203$ | $100\% \text{ Tied}$ |
| **B4: Reliability** | Calibrated | $1.0000$ | $0.0000$ | $0.0000$ | $0.2550$ ($0.2672$) | $0.1333$ ($0.1619$) | $0.0000$ | $0.5203$ | $100.0\% / 0\% / 0\%$ |
| **B5: ACARA-U** | Calibrated | **$0.4373$** | **$0.2024$** | **$0.3604$** | **$0.2588$** ($0.1397$) | **$0.1333$** ($0.1619$) | **$1.0346$** | **$0.5203$** | **$80.4\% / 2.2\% / 17.4\%$** |

---

## 3. Key Observations

1. **Retinal Authority Calibration Adjustment**: Calibrating probabilities softens peak retinal fundus confidence scores, causing ACARA-U to reduce retinal authority from $0.4529$ (B2) to $0.4373$ (B5) ($\Delta w_R = -0.0156$).
2. **Authority Concentration Reduction**: Retinal dominance decreases from $88.0\%$ to $80.4\%$, with clinical modality dominance increasing from $10.6\%$ to $17.4\%$, yielding a less concentrated routing authority distribution across channels.
3. **Conflict Reduction**: Average pairwise modality conflict decreases from $0.5368$ to $0.5203$ under calibrated risk projections.
4. **Fused Risk Stability**: Population-level fused risk shifts by $+0.0025$ ($0.2563 \to 0.2588$), remaining stable and well within bounded operating intervals.
