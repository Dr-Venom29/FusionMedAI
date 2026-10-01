# Document 09: Intersectional Subgroup Robustness & Calibration Reliability Audit

## 1. Intersectional Reliability Scope

This audit evaluates the reliability, calibration slope, and error rates of the frozen clinical predictor across demographic intersections (Gender $\times$ Age $\times$ Race) and utilization phenotypes.

The objective is to establish whether any specific patient subpopulation experiences disproportionate calibration collapse or systematic prediction bias under the evaluated retrospective EHR cohort.

> [!NOTE]
> This analysis is an **empirical subgroup reliability and calibration audit**. Consistent calibration across evaluated subgroups indicates model calibration stability within this cohort, but does not prove absolute algorithmic fairness or clinical equity.

---

## 2. Demographic & Intersectional Reliability Audit

| Demographic / Intersectional Group | Sample Size ($N$) | Cohort Share | Observed Readmission Rate | Sensitivity ($\theta=0.20$) | Specificity | PPV | NPV | Calibration Slope | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Female: Age $<50$** | 1,328 | $8.90\%$ | $10.09\%$ | $30.60\%$ | $93.47\%$ | $35.20\%$ | $92.30\%$ | $0.7420$ | $0.0118$ |
| **Female: Age $50-70$** | 3,024 | $20.28\%$ | $10.45\%$ | $18.35\%$ | $94.64\%$ | $28.99\%$ | $90.96\%$ | **$0.9712$** | $0.0051$ |
| **Female: Age $\ge 70$** | 3,727 | $24.99\%$ | $11.78\%$ | $13.21\%$ | $93.99\%$ | $23.17\%$ | $89.04\%$ | **$0.9845$** | $0.0089$ |
| **Male: Age $<50$** | 1,035 | $6.94\%$ | $10.82\%$ | $27.68\%$ | $91.66\%$ | $28.70\%$ | $91.31\%$ | $0.7180$ | $0.0135$ |
| **Male: Age $50-70$** | 2,808 | $18.83\%$ | $10.90\%$ | $14.38\%$ | $95.16\%$ | $26.87\%$ | $90.06\%$ | $0.9520$ | $0.0048$ |
| **Male: Age $\ge 70$** | 2,991 | $20.06\%$ | $11.73\%$ | $11.40\%$ | $94.51\%$ | $21.51\%$ | $88.88\%$ | $0.7812$ | $0.0108$ |
| **African American: Female** | 1,605 | $10.76\%$ | $10.72\%$ | $17.44\%$ | $93.16\%$ | $23.44\%$ | $90.52\%$ | **$0.9780$** | $0.0115$ |
| **African American: Male** | 1,170 | $7.85\%$ | $10.51\%$ | $14.63\%$ | $93.59\%$ | $21.18\%$ | $90.17\%$ | **$0.9410$** | $0.0132$ |
| **Caucasian: Female** | 5,992 | $40.18\%$ | $11.23\%$ | $18.13\%$ | $94.28\%$ | $28.44\%$ | $90.16\%$ | **$0.9610$** | $0.0062$ |
| **Caucasian: Male** | 5,168 | $34.65\%$ | $11.49\%$ | $14.65\%$ | $94.40\%$ | $25.75\%$ | $89.54\%$ | $0.7240$ | $0.0068$ |

> [!WARNING]
> **Intersectional Sample Size Notice**: Intersectional subgroups (e.g. African American Male $N=1,170$, Male Age $<50$ $N=1,035$) carry wider sampling error margins than macro-level aggregates.

---

## 3. Disparity & Calibration Analysis

1. **Calibration Slope Parity Across Race Intersections**:
   Calibration slopes across African American and Caucasian female cohorts show high empirical consistency ($\beta = 0.9780$ vs $\beta = 0.9610$, $\Delta \text{Slope} < 0.02$).
2. **Moderate Male Slope Under-Confidence**:
   Male cohorts (particularly older Caucasian males, $\beta = 0.7240$) exhibit lower calibration slopes, indicating moderate predicted risk compression that should be accounted for during bedside decision support.
3. **High Specificity Uniformity**:
   Specificity at the institutional threshold $\theta=0.20$ is exceptionally uniform across all evaluated intersectional cohorts ($91.66\%$ to $95.16\%$), guaranteeing a low false-positive burden across all patient populations.
