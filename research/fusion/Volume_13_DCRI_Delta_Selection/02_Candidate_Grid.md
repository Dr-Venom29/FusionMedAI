# Chapter 02 — Candidate Grid Specification & Domain Analysis

## 1. Mathematical Domain of DCRI

Let the active modality set be $\mathcal{A} \subseteq \{1, \dots, M\}$ with cardinality $M = \lvert \mathcal{A} \rvert \le 3$. Since:

$$R_{\text{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i \in [0.0, 1.0]$$

and each normalized modality uncertainty satisfies $U_i \in [0.0, 1.0]$, the sum of uncertainties satisfies:

$$0.0 \le U_{\text{sum}} = \sum_{i \in \mathcal{A}} U_i \le M \le 3.0$$

Consequently, for any candidate penalty multiplier $\delta \in [0.0, 1.0]$, the theoretical range of $\text{DCRI}_\delta$ is:

$$\text{DCRI}_\delta \in [-\delta M, 1.0] \subseteq [-3.0, 1.0]$$

Negative values occur whenever:

$$\delta U_{\text{sum}} > R_{\text{fusion}}$$

This occurs when the aggregate epistemic uncertainty of the reporting models exceeds the positive evidence of risk. Under the FusionMedAI fail-safe design philosophy, negative values are intentionally left unclamped to preserve signal regarding low-confidence, high-uncertainty encounters.

---

## 2. Pre-Specified Candidate Grid Taxonomy

The parameter grid comprises 11 pre-specified evaluation points chosen to capture fine gradations in the realistic operating domain ($0.00$ to $0.30$), intermediate penalties ($0.40$ to $0.50$), and extreme asymptotic boundary conditions ($0.75$ to $1.00$).

| Candidate ID | Multiplier $\delta$ | Tier / Category | Scientific Rationale |
| :--- | :---: | :--- | :--- |
| **D00** | $0.00$ | Zero Penalty Baseline | Represents pure router-level uncertainty awareness ($w_i \propto e^{-\gamma U_i}$) without second-stage additive penalty. $\text{DCRI}_0 \equiv R_{\text{fusion}}$. |
| **D05** | $0.05$ | Conservative Fine Grid | Minimal uncertainty attenuation ($\sim 5\%$ penalty relative to base risk). |
| **D10** | $0.10$ | Conservative Fine Grid | Moderate uncertainty attenuation ($\sim 22\%$ penalty relative to base risk); balanced threshold. |
| **D15** | $0.15$ | Conservative Fine Grid | Intermediate conservative attenuation ($\sim 33\%$ penalty relative to base risk). |
| **D20** | $0.20$ | Provisional Reference | Historical operating point inherited from initial C11.6 exploration. |
| **D25** | $0.25$ | Moderate Penalty | Upper boundary of fine candidate region. |
| **D30** | $0.30$ | Moderate Penalty | Transition into aggressive uncertainty discounting. |
| **D40** | $0.40$ | Moderate Penalty | Substantial uncertainty penalty exceeding $80\%$ of mean base risk. |
| **D50** | $0.50$ | Substantial Penalty | Severe uncertainty discounting driving median DCRI negative. |
| **D75** | $0.75$ | Strong Boundary | Extreme sensitivity probe evaluating tail collapse. |
| **D100** | $1.00$ | Full Boundary | Maximum theoretical penalty ($P = U_{\text{sum}}$). |

---

## 3. Upstream Component Freezing

All input variables entering the DCRI selection pipeline are frozen upstream:

```
[Frozen Upstream State]
├── Router Kernel: ACARA-U v2
│   └── Coefficients: Theta_0 = (alpha=1.0, beta=1.5, gamma=1.0, eta=0.5)
├── Modality Prior Reliabilities (Phase C11.3 Locked):
│   ├── Retina (EfficientNet-B3): R_retina = 0.929956
│   ├── Foot Ulcer (ResNet-50):   R_foot   = 0.922266
│   └── Clinical (CatBoost):      R_clin   = 0.825382
├── Calibration: Temperature & Isotonic Scaled (discrete clinical evaluation ECE = 0.000000 within controlled tabular benchmark cohort; does not imply external or universal calibration validity)
├── Quality Layer: Standardized quality metric scoring framework Q_i in [0.0, 1.0] (evaluated according to ISO/IEC 25010 data quality guidelines)
└── Cohort: N=500 Controlled Decision Packets (Seed 115)
```
