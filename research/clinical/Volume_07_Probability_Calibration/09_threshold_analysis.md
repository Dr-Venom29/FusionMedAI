# Document 09: Operating Threshold Analysis & Clinical Trade-offs

## 1. Decision Threshold Sweep

Because calibrated probabilities represent estimated risk levels, setting a decision threshold $\theta \in [0, 1]$ directly defines which patient encounters are flagged for targeted clinical transitional care interventions.

Using the validation-selected primary candidate (Isotonic Regression), we evaluate operating characteristics across a threshold grid $\theta \in [0.05, 0.50]$ on the locked test partition ($N=14,913$, $1,658$ positive readmission encounters).

---

## 2. Threshold Performance Table

| Threshold ($\theta$) | Sensitivity | Specificity | PPV (Precision) | NPV | F1 Score | TP | FP | TN | FN | Flagged Encounters | Flagged % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.05** | 96.44% | 10.62% | 11.89% | 95.98% | 0.2117 | 1,599 | 11,847 | 1,408 | 59 | 13,446 | 90.16% |
| **0.10** | 60.62% | 60.71% | 16.18% | 92.49% | 0.2554 | 1,005 | 5,208 | 8,047 | 653 | 6,213 | 41.66% |
| **0.15** | 40.05% | 78.42% | 18.84% | 91.27% | 0.2563 | 664 | 2,860 | 10,395 | 994 | 3,524 | 23.63% |
| **0.20** | 16.34% | 94.09% | 25.69% | 89.99% | 0.1998 | 271 | 784 | 12,471 | 1,387 | 1,055 | 7.07% |
| **0.25** | 7.90% | 98.14% | 34.75% | 89.50% | 0.1287 | 131 | 246 | 13,009 | 1,527 | 377 | 2.53% |
| **0.30** | 7.18% | 98.48% | 37.07% | 89.45% | 0.1203 | 119 | 202 | 13,053 | 1,539 | 321 | 2.15% |
| **0.35** | 4.64% | 99.23% | 43.02% | 89.27% | 0.0838 | 77 | 102 | 13,153 | 1,581 | 179 | 1.20% |
| **0.40** | 2.35% | 99.80% | 59.09% | 89.10% | 0.0452 | 39 | 27 | 13,228 | 1,619 | 66 | 0.44% |
| **0.45** | 2.35% | 99.80% | 59.09% | 89.10% | 0.0452 | 39 | 27 | 13,228 | 1,619 | 66 | 0.44% |
| **0.50** | 1.63% | 99.83% | 55.10% | 89.03% | 0.0316 | 27 | 22 | 13,233 | 1,631 | 49 | 0.33% |

---

## 3. Clinical Operational Regimes

Depending on institutional resource availability and intervention capacity, three distinct operational regimes emerge within the verified positive Net Benefit window ($\theta \in [0.05, 0.25]$):

### 3.1 Broad Screening Tier ($\theta = 0.10$)
- **Operating Profile**: Sensitivity = **$60.62\%$**, Specificity = **$60.71\%$**, NPV = **$92.49\%$**.
- **Workload**: Flags $41.66\%$ of discharged patients ($6,213$ encounters).
- **Target Use Case**: Low-cost, automated post-discharge outreach (e.g., automated SMS reminders, digital medication adherence check-ins).
- **Clinical Trade-off**: Captures $1,005$ of the $1,658$ readmissions, while filtering out $8,047$ true negatives.

### 3.2 Balanced Transitional Care Tier ($\theta = 0.15$)
- **Operating Profile**: Sensitivity = **$40.05\%$**, Specificity = **$78.42\%$**, PPV = **$18.84\%$**, NPV = **$91.27\%$**, F1 = **$0.2563$** (Peak F1).
- **Workload**: Flags $23.63\%$ of discharges ($3,524$ encounters).
- **Target Use Case**: Pharmacist-led discharge medication reconciliation and follow-up nurse phone calls within 48 hours.
- **Clinical Trade-off**: Identifies $664$ high-risk cases while keeping clinical workload below a quarter of total volume.

### 3.3 Intensive Resource-Constrained Tier ($\theta = 0.20$)
- **Operating Profile**: Sensitivity = **$16.34\%$**, Specificity = **$94.09\%$**, PPV = **$25.69\%$**, NPV = **$89.99\%$**.
- **Workload**: Flags $7.07\%$ of discharges ($1,055$ encounters).
- **Target Use Case**: High-intensity post-acute interventions (home health visits, multidisciplinary team case management).
- **Clinical Trade-off**: 1 in every 4 flagged patients experiences readmission ($\text{PPV} = 25.69\%$), minimizing alert fatigue and focusing high-cost staff resources.
