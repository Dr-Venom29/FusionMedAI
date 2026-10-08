# Dynamic Authority Redistribution Analysis

## 1. Redistribution Mechanics

When a modality drops out, ACARA-U dynamically reallocates decision authority to the remaining channels via softmax logit normalization over the active mask $\mathcal{A}$.

$$\Delta w_j = w_j^{\text{subset}} - w_j^{\text{full}}$$

---

## 2. Bimodal Redistribution Matrix ($N=500$)

| Subset Regime | Removed Channel | $\Delta w_R$ (Mean) | $\Delta w_F$ (Mean) | $\Delta w_C$ (Mean) | Relinquished Weight ($w_k^{\text{full}}$) | Net Absorption ($\sum \Delta w_{\text{active}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`retina_foot` ($-C$)** | Clinical | $+0.158983$ | $+0.080783$ | $-0.239766$ | $0.239766$ | $+0.239766$ |
| **`retina_clinical` ($-F$)** | Foot | $+0.171575$ | $-0.256884$ | $+0.085309$ | $0.256884$ | $+0.256884$ |
| **`foot_clinical` ($-R$)** | Retina | $-0.503350$ | $+0.251600$ | $+0.251750$ | $0.503350$ | $+0.503350$ |

---

## 3. Authority Absorption Proportions

When Clinical drops out ($-C$, releasing $23.98\%$ authority):
- **Retina absorbs**: $\frac{0.158983}{0.239766} \approx 66.31\%$ of relinquished authority.
- **Foot absorbs**: $\frac{0.080783}{0.239766} \approx 33.69\%$ of relinquished authority.

When Foot drops out ($-F$, releasing $25.69\%$ authority):
- **Retina absorbs**: $\frac{0.171575}{0.256884} \approx 66.79\%$ of relinquished authority.
- **Clinical absorbs**: $\frac{0.085309}{0.256884} \approx 33.21\%$ of relinquished authority.

When Retina drops out ($-R$, releasing $50.34\%$ authority):
- **Foot absorbs**: $\frac{0.251600}{0.503350} \approx 49.98\%$ of relinquished authority.
- **Clinical absorbs**: $\frac{0.251750}{0.503350} \approx 50.02\%$ of relinquished authority.

---

## 4. Scientific Insights

1. **Retina-Dominant Authority Redistribution**: Under the frozen C11.8 router configuration, Retina receives the highest mean authority and absorbs approximately **two-thirds** ($66.3\% - 66.8\%$) of the authority released when either Foot or Clinical is removed. This behavior is consistent with the combined effects of the frozen reliability prior, confidence, uncertainty, and quality terms in the router.
2. **Equipartition upon Retina Loss**: Under the frozen router configuration, the remaining Foot and Clinical channels receive nearly equal mean authority after Retina removal ($49.98\% / 50.02\%$). This observed allocation is consistent with the competing effects of Foot's higher reliability prior ($R_F=0.922$) and Clinical's substantially lower predictive uncertainty ($\overline{U_C} \approx 0.04$ vs $\overline{U_F} \approx 0.59$).
