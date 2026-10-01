# Research Document 08: Subgroup Fairness & Performance Equity Audit

## 1. Ethical & Clinical Motivation

Clinical artificial intelligence models must perform equitably across diverse patient demographics and clinical phenotypes. Biases embedded in healthcare administrative datasets (e.g., insurance coverage disparities, unequal access to primary care, systemic referral biases) can inadvertently cause models to perform suboptimally for vulnerable populations.

This document presents a comprehensive subgroup audit of the benchmarked **CatBoost** model on the locked test partition ($N=14,913$).

---

## 2. Demographic Subgroup Performance Audit

### A. Age Groups
Patients were stratified into three broad life stages: Younger ($<50\text{ years}$), Middle-Aged ($[50, 70)\text{ years}$), and Elderly ($\ge 70\text{ years}$):

| Subgroup | Encounter Count ($N$) | Cohort Share (\%) | Base Rate (\%) | ROC-AUC | PR-AUC | Sensitivity ($\theta=0.20$) | Specificity ($\theta=0.20$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Younger ($<50\text{ yrs}$)** | $1,972$ | $13.22\%$ | $9.84\%$ | $0.6385$ | $0.1812$ | $13.40\%$ | $95.16\%$ |
| **Middle-Aged ($[50, 70)\text{ yrs}$)** | $7,025$ | $47.11\%$ | $10.88\%$ | $0.6450$ | $0.1985$ | $15.45\%$ | $94.32\%$ |
| **Elderly ($\ge 70\text{ yrs}$)** | $5,916$ | $39.67\%$ | $11.93\%$ | $\mathbf{0.6518}$ | $\mathbf{0.2174}$ | $\mathbf{18.41\%}$ | $93.40\%$ |

*Insight*: Readmission predictability increases with age. Elderly patients have higher baseline readmission rates ($11.93\%$) and clearer multi-morbid clinical trajectories, yielding higher ROC-AUC ($0.6518$) and sensitivity ($18.41\%$).

---

### B. Racial & Ethnic Groups

| Racial / Ethnic Category | Encounter Count ($N$) | Cohort Share (\%) | Base Rate (\%) | ROC-AUC | PR-AUC | Sensitivity ($\theta=0.20$) | Specificity ($\theta=0.20$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Caucasian** | $11,180$ | $74.97\%$ | $11.39\%$ | $0.6482$ | $0.2071$ | $16.89\%$ | $93.92\%$ |
| **African American** | $2,834$ | $19.00\%$ | $10.87\%$ | $0.6438$ | $0.1942$ | $15.26\%$ | $94.46\%$ |
| **Hispanic** | $302$ | $2.03\%$ | $9.93\%$ | $0.6410$ | $0.1870$ | $13.33\%$ | $95.22\%$ |
| **Asian** | $94$ | $0.63\%$ | $8.51\%$ | $0.6320$ | $0.1650$ | $12.50\%$ | $96.51\%$ |
| **Other / Unknown** | $503$ | $3.37\%$ | $10.34\%$ | $0.6455$ | $0.1980$ | $15.38\%$ | $94.68\%$ |

*Insight*: Performance is stable across major racial cohorts (Caucasian ROC-AUC $0.6482$ vs African American $0.6438$, $\Delta \text{ROC-AUC} = 0.0044$). Small sample sizes in Asian ($N=94$) and Hispanic ($N=302$) sub-cohorts yield wider confidence intervals but show consistent specificities ($\ge 95.2\%$).

---

### C. Gender

| Gender | Encounter Count ($N$) | Cohort Share (\%) | Base Rate (\%) | ROC-AUC | PR-AUC | Sensitivity ($\theta=0.20$) | Specificity ($\theta=0.20$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Female** | $8,012$ | $53.72\%$ | $11.33\%$ | $0.6491$ | $0.2085$ | $16.96\%$ | $93.98\%$ |
| **Male** | $6,901$ | $46.28\%$ | $10.95\%$ | $0.6449$ | $0.1982$ | $15.87\%$ | $94.15\%$ |

*Insight*: Near-identical discrimination and threshold metrics between female and male cohorts, satisfying demographic parity criteria.

---

## 3. Clinical Phenotype & Utilization Subgroup Audit

### A. Prior Inpatient Hospitalizations (Strongest Stratifier)

| Utilization Subgroup | Encounter Count ($N$) | Cohort Share (\%) | Base Rate (\%) | ROC-AUC | PR-AUC | Sensitivity ($\theta=0.20$) | Specificity ($\theta=0.20$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Prior Inpatient ($0$)** | $10,134$ | $67.95\%$ | $7.62\%$ | $0.6124$ | $0.1345$ | $8.42\%$ | $97.10\%$ |
| **Prior Inpatient ($\ge 1$)** | $4,779$ | $32.05\%$ | $\mathbf{18.67\%}$ | $\mathbf{0.6380}$ | $\mathbf{0.3012}$ | $\mathbf{23.43\%}$ | $\mathbf{87.41\%}$ |

*Insight*: Prior inpatient admission is the most decisive risk multiplier. Patients with $\ge 1$ prior admission have a $2.45\times$ higher baseline readmission rate ($18.67\%$ vs $7.62\%$). The model captures $23.43\%$ of true readmissions in this high-risk sub-cohort at $\theta=0.20$.

---

### B. Primary ICD-9 Diagnosis Category

```mermaid
xychart-beta
    title "CatBoost ROC-AUC by Primary Diagnosis Category"
    x-axis ["Circulatory", "Respiratory", "Diabetes", "Digestive", "Injury", "Genitourinary", "Musculoskeletal", "Neoplasms"]
    y-axis "ROC-AUC" 0.55 0.70
    bar [0.654, 0.648, 0.658, 0.639, 0.641, 0.635, 0.628, 0.647]
```

| Primary ICD-9 Chapter | Encounter Count ($N$) | Base Rate (\%) | ROC-AUC | PR-AUC | Sensitivity ($\theta=0.20$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Circulatory (390-459)** | $4,512$ | $11.41\%$ | $0.6542$ | $0.2155$ | $17.51\%$ |
| **Diabetes (250.xx)** | $1,288$ | $12.89\%$ | $\mathbf{0.6580}$ | $\mathbf{0.2310}$ | $\mathbf{19.88\%}$ |
| **Respiratory (460-519)** | $2,098$ | $11.01\%$ | $0.6485$ | $0.2042$ | $16.45\%$ |
| **Digestive (520-579)** | $1,372$ | $10.28\%$ | $0.6392$ | $0.1874$ | $14.89\%$ |
| **Injury / Poisoning (800-999)** | $1,051$ | $10.18\%$ | $0.6411$ | $0.1895$ | $14.95\%$ |
| **Genitourinary (580-629)** | $748$ | $10.83\%$ | $0.6352$ | $0.1820$ | $14.81\%$ |
| **Musculoskeletal (710-739)** | $732$ | $8.61\%$ | $0.6281$ | $0.1582$ | $11.11\%$ |
| **Neoplasms (140-239)** | $491$ | $10.59\%$ | $0.6470$ | $0.1980$ | $15.38\%$ |

---

## 4. Algorithmic Fairness & Disparity Summary

1. **No Disparate Exclusion**: Specificity remains high ($\ge 93.4\%$) across all demographic subgroups, ensuring that false positive rates do not disproportionately burden any single group.
2. **Clinical Consistency**: Disparities in Sensitivity reflect genuine differences in base prevalence ($7.62\%$ in unadmitted vs $18.67\%$ in prior admitted), adhering to the principle of calibrated clinical utility rather than artificial statistical equalized odds.
