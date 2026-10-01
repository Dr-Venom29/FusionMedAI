# Document 07: Subgroup Uncertainty & Longitudinal Phenotype Audit

## 1. Subgroup Uncertainty Reliability Audit

To ensure that the uncertainty mechanism remains informative and reliable across diverse patient demographics and clinical phenotypes, we audit uncertainty distributions across:
1. **Prior Utilization Phenotype**: Prior Inpatient Encounters ($= 0$ vs $\ge 1$).
2. **Gender**: Male vs. Female.
3. **Age Brackets**: Younger ($<50$ years), Middle-Aged ($50-70$ years), and Older ($\ge 70$ years).

---

## 2. Empirical Subgroup Uncertainty Table

| Subgroup | $N$ | Prevalence | Mean Uncertainty ($\sigma_p$) | Median Uncertainty | IQR Uncertainty | Error Rate ($\theta=0.20$) | Error Detection AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prior Inpatient = 0** | 9,912 | 8.59% | **0.0160** | 0.0131 | $[0.0100, 0.0181]$ | 8.67% | 0.5871 |
| **Prior Inpatient $\ge 1$** | 5,001 | 16.14% | **0.0336** | 0.0255 | $[0.0186, 0.0384]$ | 27.05% | **0.7048** |
| **Gender: Male** | 6,834 | 11.25% | 0.0215 | 0.0158 | $[0.0112, 0.0247]$ | 14.97% | **0.7003** |
| **Gender: Female** | 8,079 | 11.00% | 0.0223 | 0.0165 | $[0.0115, 0.0250]$ | 14.72% | **0.7215** |
| **Age $< 50$** | 2,363 | 10.41% | 0.0240 | 0.0140 | $[0.0098, 0.0257]$ | 14.30% | **0.7669** |
| **Age $50-70$** | 5,832 | 10.67% | 0.0210 | 0.0151 | $[0.0106, 0.0245]$ | 13.80% | **0.7168** |
| **Age $\ge 70$** | 6,718 | 11.76% | 0.0219 | 0.0174 | $[0.0128, 0.0251]$ | 15.91% | **0.6826** |

---

## 3. Scientific Connection: C6 Attributions $\to$ C8 Uncertainty

### 3.1 Prior Inpatient Phenotype Divergence
In Phase C6, TreeSHAP established that `number_inpatient` is the single most dominant feature ($22.43\%$ attribution share). The uncertainty audit reveals the epistemic consequence of this feature:
- **Prior Inpatient $= 0$ ($N=9,912$)**: When historical inpatient utilization is absent, the model exhibits low prediction dispersion ($\mu_{\sigma} = 0.0160$). Most predictions default toward the lower base rate ($8.59\%$), resulting in a low error rate ($8.67\%$). Because misclassifications in this group are dominated by unobserved factors rather than feature ambiguity, error-detection AUROC is modest ($0.5871$).
- **Prior Inpatient $\ge 1$ ($N=5,001$)**: When utilization history is present, the model experiences higher epistemic variance ($\mu_{\sigma} = 0.0336$, more than $2\times$ higher). In this complex sub-cohort, uncertainty is highly informative, achieving an **Error Detection AUROC of $0.7048$**.

### 3.2 Demographic Subgroup Reliability
- **Gender**: Mean uncertainty is virtually identical between Male ($0.0215$) and Female ($0.0223$) cohorts, with strong error-detection AUROC across both ($0.7003$ Male, $0.7215$ Female).
- **Age**: Younger patients ($<50$) exhibit the highest error detection capability (**$\text{AUROC} = 0.7669$**), while older patients ($\ge 70$) maintain balanced dispersion ($\mu_{\sigma} = 0.0219$, $\text{AUROC} = 0.6826$).

---

## 4. Methodological Scope & Fairness Distinctions
This subgroup audit evaluates **uncertainty reliability and failure detection** across cohorts. It demonstrates:
- **No Subgroup Collapse**: The uncertainty mechanism remains informative across demographic and utilization strata.
- **Fairness Non-Claim**: Uncertainty reliability across subgroups is not equivalent to complete algorithmic fairness, demographic parity, or equalized opportunity. Comprehensive intersectional fairness and distribution shift auditing will be conducted in Phase C9.
