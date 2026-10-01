# Research Document 06: Local Patient Encounter Explanations

## 1. Local Case Attribution Methodology
Local explanations compute patient-specific additive feature contributions $\phi_j(x^{(i)})$ to explain individual predictions. 

Evaluated against the baseline model log-odds $\phi_0 = -2.1686$ ($\approx 10.26\%$ base probability), the final prediction is:
$$\hat{z}(x^{(i)}) = \phi_0 + \sum_{j=1}^{D} \phi_j(x^{(i)}), \quad \hat{p}(x^{(i)}) = \sigma\left(\hat{z}(x^{(i)})\right)$$

A deterministic, multi-tier cohort of representative test encounters was audited:

---

## 2. Representative Patient Case Breakdowns

### Case 1: High-Attribution Positive Case (Predicted $\hat{p} = 0.5421$, Actual $y=1$)
- **Base Log-Odds ($\phi_0$)**: $-2.1686$
- **Predicted Probability**: $54.21\%$ ($\hat{z} = +0.1692$)
- **Top Risk-Increasing Features**:
  1. `number_inpatient` $= 3$ ($+1.1420$)
  2. `age_ordinal` $= [70-80)$ ($+0.3215$)
  3. `time_in_hospital` $= 8\text{ days}$ ($+0.2840$)
  4. `insulin_exposure` $= 1$ ($+0.2450$)
- **Top Risk-Decreasing Features**:
  1. `num_procedures` $= 2$ ($-0.1120$)
  2. `medical_specialty_grouped_Cardiology` $= 1$ ($-0.0890$)
- *Attribution Summary*: The prediction is primarily increased by prior inpatient utilization, age, length of stay, and insulin exposure.

---

### Case 2: True Positive at Operating Threshold (Predicted $\hat{p} = 0.2435$, Actual $y=1$)
- **Base Log-Odds ($\phi_0$)**: $-2.1686$
- **Predicted Probability**: $24.35\%$ ($\hat{z} = -1.1335$)
- **Top Risk-Increasing Features**:
  1. `number_inpatient` $= 1$ ($+0.4680$)
  2. `number_diagnoses` $= 9$ ($+0.2150$)
  3. `diag_1_chapter_Circulatory` $= 1$ ($+0.1820$)
  4. `num_medications` $= 24$ ($+0.1410$)
- **Top Risk-Decreasing Features**:
  1. `metformin_exposure` $= 1$ ($-0.1105$)
  2. `age_ordinal` $= [50-60)$ ($-0.0820$)
- *Attribution Summary*: The prediction exceeds the operating threshold of $\theta=0.20$ and is therefore classified as positive under the predefined evaluation protocol.

---

### Case 3: False Positive Case (Predicted $\hat{p} = 0.2280$, Actual $y=0$)
- **Base Log-Odds ($\phi_0$)**: $-2.1686$
- **Predicted Probability**: $22.80\%$ ($\hat{z} = -1.2196$)
- **Top Risk-Increasing Features**:
  1. `number_inpatient` $= 1$ ($+0.4720$)
  2. `time_in_hospital` $= 7\text{ days}$ ($+0.1980$)
  3. `num_medications` $= 21$ ($+0.1650$)
- **Top Risk-Decreasing Features**:
  1. `num_procedures` $= 3$ ($-0.1420$)
  2. `A1Cresult_ordinal` $= \text{Normal}$ ($-0.0950$)
- *Attribution Summary*: The model assigned elevated risk despite the observed non-readmission outcome. The strongest positive and negative feature contributions are shown above.

---

### Case 4: False Negative Case (Predicted $\hat{p} = 0.1240$, Actual $y=1$)
- **Base Log-Odds ($\phi_0$)**: $-2.1686$
- **Predicted Probability**: $12.40\%$ ($\hat{z} = -1.9554$)
- **Top Risk-Increasing Features**:
  1. `diag_1_chapter_Circulatory` $= 1$ ($+0.1620$)
  2. `num_medications` $= 18$ ($+0.0950$)
- **Top Risk-Decreasing Features**:
  1. `number_inpatient` $= 0$ ($-0.1250$)
  2. `time_in_hospital` $= 2\text{ days}$ ($-0.0880$)
- *Attribution Summary*: The model assigned a probability below the operating threshold despite the observed readmission outcome.

---

### Case 5: Lowest Risk Baseline Encounter (Predicted $\hat{p} = 0.0382$, Actual $y=0$)
- **Base Log-Odds ($\phi_0$)**: $-2.1686$
- **Predicted Probability**: $3.82\%$ ($\hat{z} = -3.2260$)
- **Top Risk-Increasing Features**: No notable positive attribution among the listed features ($<+0.02$).
- **Top Risk-Decreasing Features**:
  1. `number_inpatient` $= 0$ ($-0.1250$)
  2. `age_ordinal` $= [30-40)$ ($-0.3150$)
  3. `time_in_hospital` $= 1\text{ day}$ ($-0.2450$)
  4. `num_medications` $= 4$ ($-0.1980$)
- *Attribution Summary*: The predicted probability is below the operating threshold, and the observed outcome is negative.
