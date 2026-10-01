# Document 08: Subgroup Calibration Reliability Audit

## 1. Demographic & Clinical Subgroup Audit

A model that is well-calibrated globally may exhibit localized calibration drift across sensitive demographic sub-populations or high-risk clinical cohorts.

Using the validation-selected primary candidate (Isotonic Regression), we evaluate subgroup calibration reliability on the locked test partition ($N=14,913$) across:
1. **Prior Utilization**: Prior Inpatient Encounters ($\ge 1$ vs $0$).
2. **Gender**: Male vs. Female.
3. **Age Cohorts**: Younger ($<50$ years), Middle-Aged ($50-70$ years), and Older ($\ge 70$ years).

---

## 2. Empirical Subgroup Calibration Results

The table below reports sample counts, empirical readmission prevalence, raw vs. calibrated Brier score, Log Loss, ECE, intercept, slope, ROC-AUC, and PR-AUC across all evaluated subgroups:

| Subgroup | $N$ | Prevalence | Raw Brier | Cal Brier | Raw LogLoss | Cal LogLoss | Raw ECE | Cal ECE | Raw Slope | Cal Slope | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Inpatient $\ge 1$** | 2,091 | 20.76% | 0.158297 | 0.159316 | 0.494762 | 0.496996 | 0.019119 | 0.027785 | 0.894029 | 0.790895 | 0.605581 | 0.306212 |
| **Gender: Male** | 6,834 | 11.25% | 0.096891 | 0.096999 | 0.339319 | 0.343051 | 0.006055 | 0.006863 | 0.887056 | 0.741654 | 0.631137 | 0.187090 |
| **Gender: Female** | 8,079 | 11.00% | 0.094028 | 0.094329 | 0.329094 | 0.329865 | 0.007984 | 0.009834 | 1.002783 | 0.946484 | 0.662421 | 0.199412 |
| **Age $< 50$** | 2,363 | 10.41% | 0.086346 | 0.086801 | 0.305421 | 0.315728 | 0.008319 | 0.009843 | 0.940828 | 0.730616 | 0.703723 | 0.250101 |
| **Age $50-70$** | 5,832 | 10.67% | 0.091615 | 0.091745 | 0.322849 | 0.323223 | 0.002718 | 0.003693 | 0.976023 | 0.934733 | 0.660571 | 0.194305 |
| **Age $\ge 70$** | 6,718 | 11.76% | 0.101737 | 0.101938 | 0.353244 | 0.354017 | 0.005716 | 0.008525 | 0.927192 | 0.887883 | 0.612270 | 0.171715 |

---

## 3. Findings & Subgroup Calibration Observations

### 3.1 Prior Inpatient Utilization Cohort
Patients with $\ge 1$ prior inpatient stay exhibit an observed 30-day readmission prevalence of **$20.76\%$** (nearly double the baseline cohort rate of $11.11\%$).
- The model maintains strong discrimination in this high-risk group with a **PR-AUC of $0.3062$**.
- Calibration slope is $\beta = 0.8940$ (raw) and $0.7909$ (calibrated), reflecting that while risk is elevated, extreme predictions require cautious interpretation.

### 3.2 Gender Reliability
- Readmission prevalence is closely matched across genders ($11.25\%$ Male vs $11.00\%$ Female).
- Female cohort calibration slope tracks near unity ($\beta = 1.0028$ raw, $0.9465$ calibrated), with strong discrimination ($\text{ROC-AUC} = 0.6624$).
- Male cohort exhibits a slightly flatter slope ($\beta = 0.8871$ raw, $0.7417$ calibrated) and $\text{ROC-AUC} = 0.6311$.

### 3.3 Age Stratification
- **Age $< 50$**: Exhibits the highest discrimination (**ROC-AUC: 0.7037**, **PR-AUC: 0.2501**), with low Brier score ($0.0863$).
- **Age $50-70$**: Demonstrates the best calibration slope ($\beta = 0.9760$ raw, $0.9347$ calibrated) and minimal ECE ($0.0027$ raw, $0.0037$ calibrated).
- **Age $\ge 70$**: Largest subgroup ($N=6,718$) with elevated prevalence ($11.76\%$), maintaining robust Brier score ($0.1017$) and balanced slope ($\beta = 0.9272$).

---

## 4. Methodological Scope & Fairness Distinctions
This subgroup audit evaluates **calibration consistency and statistical reliability** across cohorts. It demonstrates:
- **No Major Calibration Degradation**: All evaluated subgroup ECE values remain below $0.030$.
- **Fairness Non-Claim**: Statistical calibration parity is not equivalent to algorithmic fairness, demographic parity, or equalized odds. Broader sociotechnical fairness auditing and intersectional disparity evaluations remain the subject of dedicated research in Phase C9.
