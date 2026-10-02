# 04 Exact Empirical Reliability Results — Phase C11.3

## 1. Measured Validation Values

The exact evaluation of discrimination ($\text{AUC}_i$) and calibration quality ($1 - \text{ECE}_i$) across the locked validation splits yields the following empirical metrics:

```text
====================================================================================================
Modality        Task Type         Val N    Validation AUC    Validation ECE    1 - ECE    Global R_i
====================================================================================================
Retina (R)      5-Class DR          366        0.911149          0.051237     0.948763      0.929956
Foot (F)        4-Class Wagner    1,006        0.884435          0.039903     0.960097      0.922266
Clinical (C)    Binary Readmit   14,911        0.650764          0.000000     1.000000      0.825382
====================================================================================================
```

---

## 2. Detailed Breakdown by Modality

### 2.1 Retinal Imaging Modality ($R_R$)
- **Validation Macro OvR AUC**: $0.911149$
- **10-Bin Equal-Frequency ECE**: $0.051237$
- **Calibration Quality Score ($1 - \text{ECE}_R$)**: $0.948763$
- **Calculation**:
  $$R_R = \frac{1}{2} \cdot [0.911149 + 0.948763] = \mathbf{0.929956}$$

### 2.2 Diabetic Foot Ulcer Modality ($R_F$)
- **Validation Macro OvR AUC**: $0.884435$
- **10-Bin Equal-Frequency ECE**: $0.039903$
- **Calibration Quality Score ($1 - \text{ECE}_F$)**: $0.960097$
- **Calculation**:
  $$R_F = \frac{1}{2} \cdot [0.884435 + 0.960097] = \mathbf{0.922266}$$

### 2.3 Structured Clinical EHR Modality ($R_C$)
- **Validation Binary ROC-AUC**: $0.650764$
- **10-Bin Equal-Frequency ECE**: $0.000000$ (Isotonic mapping on validation set aligns empirical bins exactly)
- **Calibration Quality Score ($1 - \text{ECE}_C$)**: $1.000000$
- **Calculation**:
  $$R_C = \frac{1}{2} \cdot [0.650764 + 1.000000] = \mathbf{0.825382}$$

---

## 3. Directionality & Sensitivity Verification

1. **AUC Sensitivity**: Holding ECE constant, $\frac{\partial R_i}{\partial \text{AUC}_i} = +0.5 > 0$ (improving discrimination increases reliability prior).
2. **ECE Sensitivity**: Holding AUC constant, $\frac{\partial R_i}{\partial \text{ECE}_i} = -0.5 < 0$ (poor calibration decreases reliability prior).
3. **No Ranking Artifacts**: The computed values accurately reflect both image models possessing high joint discrimination/calibration ($R_R \approx 0.930, R_F \approx 0.922$) while tabular EHR reflects high calibration with moderate retrospective discrimination ($R_C \approx 0.825$).

---

## 4. Scientific Disclosure: Clinical In-Sample Isotonic Calibration Artifact

The calculated Clinical validation $\text{ECE}_C = 0.000000$ is disclosed under the following methodological audit:
- **Mechanism**: Isotonic regression was fitted directly on the raw CatBoost validation predictions $p_{\text{raw}}^{\text{val}}$. Because the calibrator was evaluated on the same observations used for fitting, the resulting validation probabilities are subject to in-sample calibration optimism. Under the present validation data and the 10-bin equal-frequency ECE definition, this produces an observed ECE of $0.000000$. This value should therefore not be interpreted as an out-of-sample calibration estimate.
- **Parametric / Out-of-Sample Reference**:
  - Parametric Platt Scaling validation $\text{ECE} = 0.0030 \implies R_C = 0.823882$.
  - Parametric Beta Calibration validation $\text{ECE} = 0.0040 \implies R_C = 0.823382$.
  - Raw uncalibrated CatBoost validation $\text{ECE} = 0.0048 \implies R_C = 0.822982$.
  - Held-out locked test set isotonic $\text{ECE} = 0.0062$.
- **Impact on Reliability**: The maximum variation across calibration definitions is $\Delta R_C \le 0.0024$ ($<0.3\%$). Locking $R_C = 0.825382$ strictly follows the pre-registered protocol (using the selected isotonic calibrator on the validation split) while transparently acknowledging this in-sample property.

---

## 5. Independent Live Verification

The frozen AUC, ECE, and reliability values are not verified solely against static serialized constants. The C11.3 deep verification gate ([`verify_c11_3_reliability.py`](../../verification/fusion/reliability/verify_c11_3_reliability.py)) reconstructs calibrated validation probabilities directly from the stored validation artifacts on disk and recomputes AUC, ECE, and $R_i$ live from scratch.

The live recomputations match the frozen constants within $10^{-5}$ tolerance across Retina, Foot, and Clinical before the C11.3 freeze is accepted:
- **Retina Live**: $\text{AUC}=0.911149, \text{ECE}=0.051237 \implies R_R=0.929956$
- **Foot Live**: $\text{AUC}=0.884435, \text{ECE}=0.039903 \implies R_F=0.922266$
- **Clinical Live**: $\text{AUC}=0.650764, \text{ECE}=0.000000 \implies R_C=0.825382$
