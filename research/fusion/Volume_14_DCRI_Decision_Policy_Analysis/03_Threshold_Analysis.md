# Chapter 03 — Decision Threshold & Sensitivity Analysis

## 1. Primary Policy Operating Comparison ($\tau_1 = 0.20, \tau_2 = 0.40$)

Evaluated across the frozen $N=500$ controlled decision cohort (`seed=115`) under reference router $\Theta_0$ and frozen uncertainty multiplier $\delta^* = 0.10$:

| Action Category | Policy B: Fused Risk $R_{\mathrm{fusion}}$ (Unpenalized) | Policy A: $\mathrm{DCRI}_{0.10}$ (Uncertainty-Discounted) | Absolute Difference | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Routine Review (Tier 0: $< 0.20$)** | $192$ ($38.4\%$) | **$262$ ($52.4\%$)** | $+70$ ($+14.0\%$) | $+36.46\%$ |
| **Additional Assessment (Tier 1: $[0.20, 0.40)$)** | $225$ ($45.0\%$) | **$181$ ($36.2\%$)** | $-44$ ($-8.8\%$) | $-19.56\%$ |
| **Escalation for Review (Tier 2: $\ge 0.40$)** | $83$ ($16.6\%$) | **$57$ ($11.4\%$)** | $-26$ ($-5.2\%$) | $-31.33\%$ |

### Figure 1: Action Allocation Comparison
![Figure 1: Policy Action Distributions](figures/fig1_policy_action_distributions.png)

---

## 2. Action Transition & Reclassification Dynamics

The $3 \times 3$ action transition matrix maps how packets migrate from Policy B ($R_{\mathrm{fusion}}$) to Policy A ($\mathrm{DCRI}_{0.10}$):

| Policy B Action \ Policy A Action | Routine Review (Tier 0) | Additional Assessment (Tier 1) | Escalation (Tier 2) | Total Policy B Packets |
| :--- | :---: | :---: | :---: | :---: |
| **Routine Review (Tier 0)** | **$192$ ($100.0\%$)** | $0$ ($0.0\%$) | $0$ ($0.0\%$) | $192$ ($38.4\%$) |
| **Additional Assessment (Tier 1)** | **$70$ ($31.11\%$)** | **$155$ ($68.89\%$)** | $0$ ($0.0\%$) | $225$ ($45.0\%$) |
| **Escalation (Tier 2)** | $0$ ($0.0\%$) | **$26$ ($31.33\%$)** | **$57$ ($68.67\%$)** | $83$ ($16.6\%$) |
| **Total Policy A Packets** | **$262$ ($52.4\%$)** | **$181$ ($36.2\%$)** | **$57$ ($11.4\%$)** | **$500$ ($100.0\%$)** |

### Key Reclassification Invariants:
1. **Total Reclassification Count**: $96/500$ packets ($19.2\%$).
2. **Strict Monotonic Downgrade (H1 Consistent with the evaluated results)**: All $96$ reclassified packets moved to a lower action tier ($100\%$ downgrades, $0\%$ upgrades).
3. **Escalation Allocation Shift (H2 Supported within this benchmark)**: Under Policy A, $26$ packets move from Escalation down to Additional Assessment, representing an absolute shift of $5.2$ percentage points of the total cohort and a **$31.33\%$ relative reduction** in the Escalation category compared to the Policy B reference baseline.

---

## 3. Pre-Specified Threshold Grid Sweep (25 Pairs)

| $\tau_1$ (Low) | $\tau_2$ (High) | Policy B Routine (%) | Policy A Routine (%) | Policy B Escalation (%) | Policy A Escalation (%) | Total Reclassified (%) | Escalation Reduction (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $0.10$ | $0.35$ | $10.8\%$ | $19.4\%$ | $32.4\%$ | $24.4\%$ | $16.6\%$ | $8.0\%$ |
| $0.10$ | $0.40$ | $10.8\%$ | $19.4\%$ | $16.6\%$ | $11.4\%$ | $13.8\%$ | $5.2\%$ |
| $0.10$ | $0.45$ | $10.8\%$ | $19.4\%$ | $16.0\%$ | $10.6\%$ | $14.0\%$ | $5.4\%$ |
| $0.10$ | $0.50$ | $10.8\%$ | $19.4\%$ | $12.0\%$ | $7.4\%$ | $13.2\%$ | $4.6\%$ |
| $0.10$ | $0.60$ | $10.8\%$ | $19.4\%$ | $4.4\%$ | $2.4\%$ | $10.6\%$ | $2.0\%$ |
| $0.15$ | $0.35$ | $23.0\%$ | $33.4\%$ | $32.4\%$ | $24.4\%$ | $18.4\%$ | $8.0\%$ |
| $0.15$ | $0.40$ | $23.0\%$ | $33.4\%$ | $16.6\%$ | $11.4\%$ | $15.6\%$ | $5.2\%$ |
| $0.15$ | $0.45$ | $23.0\%$ | $33.4\%$ | $16.0\%$ | $10.6\%$ | $15.8\%$ | $5.4\%$ |
| $0.15$ | $0.50$ | $23.0\%$ | $33.4\%$ | $12.0\%$ | $7.4\%$ | $15.0\%$ | $4.6\%$ |
| $0.15$ | $0.60$ | $23.0\%$ | $33.4\%$ | $4.4\%$ | $2.4\%$ | $12.4\%$ | $2.0\%$ |
| **$0.20$** | **$0.40$** | **$38.4\%$** | **$52.4\%$** | **$16.6\%$** | **$11.4\%$** | **$19.2\%$** | **$5.2\%$** |
| $0.20$ | $0.45$ | $38.4\%$ | $52.4\%$ | $16.0\%$ | $10.6\%$ | $19.4\%$ | $5.4\%$ |
| $0.20$ | $0.50$ | $38.4\%$ | $52.4\%$ | $12.0\%$ | $7.4\%$ | $18.6\%$ | $4.6\%$ |
| $0.20$ | $0.60$ | $38.4\%$ | $52.4\%$ | $4.4\%$ | $2.4\%$ | $16.0\%$ | $2.0\%$ |
| $0.25$ | $0.35$ | $46.0\%$ | $58.2\%$ | $32.4\%$ | $24.4\%$ | $20.2\%$ | $8.0\%$ |
| $0.25$ | $0.40$ | $46.0\%$ | $58.2\%$ | $16.6\%$ | $11.4\%$ | $17.4\%$ | $5.2\%$ |
| $0.25$ | $0.45$ | $46.0\%$ | $58.2\%$ | $16.0\%$ | $10.6\%$ | $17.6\%$ | $5.4\%$ |
| $0.25$ | $0.50$ | $46.0\%$ | $58.2\%$ | $12.0\%$ | $7.4\%$ | $16.8\%$ | $4.6\%$ |
| $0.25$ | $0.60$ | $46.0\%$ | $58.2\%$ | $4.4\%$ | $2.4\%$ | $14.2\%$ | $2.0\%$ |
| $0.30$ | $0.35$ | $56.0\%$ | $68.4\%$ | $32.4\%$ | $24.4\%$ | $20.4\%$ | $8.0\%$ |
| $0.30$ | $0.40$ | $56.0\%$ | $68.4\%$ | $16.6\%$ | $11.4\%$ | $17.6\%$ | $5.2\%$ |
| $0.30$ | $0.45$ | $56.0\%$ | $68.4\%$ | $16.0\%$ | $10.6\%$ | $17.8\%$ | $5.4\%$ |
| $0.30$ | $0.50$ | $56.0\%$ | $68.4\%$ | $12.0\%$ | $7.4\%$ | $17.0\%$ | $4.6\%$ |
| $0.30$ | $0.60$ | $56.0\%$ | $68.4\%$ | $4.4\%$ | $2.4\%$ | $14.4\%$ | $2.0\%$ |

### Figure 2: Reclassification Sensitivity Across Threshold Grid
![Figure 2: Threshold Sensitivity Heatmap](figures/fig2_threshold_sensitivity_heatmap.png)
