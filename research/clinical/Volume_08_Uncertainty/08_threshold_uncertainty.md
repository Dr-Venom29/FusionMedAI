# Document 08: Threshold Ambiguity & Clinical Decision Tiers

## 1. Decision Tiers around Operating Threshold $\theta = 0.20$

In clinical decision-support workflows, a point prediction close to the decision threshold (e.g., $\bar{p} = 0.201$) requires different management if the model is confident ($\sigma_p = 0.008$) versus highly uncertain ($\sigma_p = 0.055$).

We define an uncertainty threshold at the 75th percentile ($\sigma_{\text{thresh}} = 0.0249$) and a threshold ambiguity zone $[0.17, 0.23]$ around $\theta = 0.20$.

---

## 2. Decision Tiers Empirical Breakdown

| Decision Tier | Encounter Count ($N$) | Cohort Share (%) | Observed Prevalence | Mean Predicted Prob ($\bar{p}$) | Mean Uncertainty ($\sigma_p$) | Error Rate | Proposed Decision-Support Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Low Risk / Low Uncertainty** | **10,850** | **$72.76\%$** | 8.99% | 0.0890 | 0.0140 | 8.99% | **Low-Intensity CDS Protocol**: Confident low-risk trajectory, subject to clinical oversight. |
| **Low Risk / High Uncertainty** | **2,120** | **$14.22\%$** | 12.97% | 0.1227 | 0.0367 | 12.97% | **Secondary Review Candidate**: Sub-threshold but elevated parameter ambiguity. |
| **High Risk / Low Uncertainty** | **7** | **$0.05\%$** | 14.29% | 0.2483 | 0.0226 | 85.71% | **Clear Elevated Risk Flag**: High model confidence in elevated trajectory. |
| **High Risk / High Uncertainty** | **694** | **$4.65\%$** | 28.96% | 0.3146 | 0.0737 | 71.04% | **Intensive High-Risk Review**: Complex patient; high observed readmission rate ($29.0\%$). |
| **Near Threshold / High Uncertainty** | **915** | **$6.14\%$** | 17.81% | 0.1954 | 0.0429 | 39.89% | **Decision Boundary Ambiguity**: Unclear actionability; flagged for secondary review. |
| **Near Threshold / Low Uncertainty** | **327** | **$2.19\%$** | 13.15% | 0.1887 | 0.0210 | 29.97% | **Marginal Stable Case**: Confident near-threshold assessment. |

---

## 3. Operational Insights

1. **Low-Intensity CDS Coverage ($72.76\%$)**:
   Almost three-quarters of all hospital discharges fall into the **Low Risk / Low Uncertainty** tier, where the model exhibits high parameter stability and lower baseline readmission frequency ($8.99\%$).
2. **Actionable Isolation of Ambiguity ($6.14\%$)**:
   The **Near Threshold / High Uncertainty** tier isolates $915$ encounters where point predictions straddle the boundary and model variance is elevated. Rather than forcing a fragile binary alert, the proposed protocol tags these encounters as *Decision Ambiguity* for secondary review.
3. **High-Yield Triage for High-Risk Cohort ($4.65\%$)**:
   The **High Risk / High Uncertainty** tier contains patients with an observed readmission rate of **$28.96\%$** (nearly $3\times$ the population base rate).
