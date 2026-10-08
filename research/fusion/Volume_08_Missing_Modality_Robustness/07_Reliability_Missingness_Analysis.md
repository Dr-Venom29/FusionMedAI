# Reliability × Missingness Interaction Analysis

## 1. Frozen Reliability Hierarchy

From Phase C11.3 (Global Reliability Priors), the validation-derived reliability constants are:

$$R_R = 0.929956 > R_F = 0.922266 > R_C = 0.825382$$

---

## 2. Single-Modality Removal Hierarchy

| Removed Modality | Frozen Prior ($R_i$) | Remaining Pair | Mean $\Delta R$ | Mean $\Delta \text{DCRI}_{0.20}$ | 95% Bootstrap CI ($\Delta R$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Retina ($-R$)** | **0.929956** (Highest) | Foot + Clinical | **0.145687** | 0.145728 | $[0.137358, 0.154205]$ |
| **Foot ($-F$)** | **0.922266** (Second) | Retina + Clinical | **0.103242** | 0.123969 | $[0.095084, 0.111734]$ |
| **Clinical ($-C$)** | **0.825382** (Third) | Retina + Foot | **0.062082** | 0.067204 | $[0.057628, 0.066712]$ |

---

## 3. Structural Ordering Insights

1. **Observed Impact Ordering Under Frozen Reliability Priors**: In the evaluated C11.8 cohort, the ordering of mean single-modality risk sensitivity matched the ordering of the frozen reliability priors:

$$\overline{\Delta R}_{-R} (0.145687) > \overline{\Delta R}_{-F} (0.103242) > \overline{\Delta R}_{-C} (0.062082)$$

This is an empirical association under the frozen router configuration and should not be interpreted as evidence that reliability alone determines missing-modality sensitivity.
2. **Authority Buffer**: Because Clinical has the lowest reliability prior ($R_C = 0.825382$), its baseline authority is lowest ($\overline{w_C} = 0.239766$). When Clinical is missing, Retina and Foot absorb its authority with minimal overall risk shift.
3. **Impact of Highest-Authority Modality Removal**: Removing Retina removes the dominant channel ($\overline{w_R} = 0.503350$), forcing the system to rely entirely on Foot and Clinical, producing a larger decision trajectory shift under the evaluated cohort.
