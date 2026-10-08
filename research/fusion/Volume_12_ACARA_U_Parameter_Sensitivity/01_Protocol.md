# Phase C11.12 Research Protocol & Pre-Specified Questions

## 1. Scientific Context & Core Research Question

Phase C11.11 characterized the impact of upstream probability calibration on decision-level fusion. Phase C11.12 examines the intrinsic parameter sensitivity of the frozen ACARA-U routing logit kernel:

$$z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$$

around the established reference configuration:

$$\Theta_0 = (\alpha=1.0, \beta=1.5, \gamma=1.0, \eta=0.5)$$

The primary research question is:

> **How sensitive are ACARA-U modality-routing decisions and derived decision indices to perturbations of the router weighting coefficients $\alpha, \beta, \gamma, \eta$ around the frozen reference configuration, and does the router occupy a locally stable behavioral neighborhood free from routing collapse or numerical discontinuities?**

---

## 2. Core Methodological Scope: Sensitivity Analysis vs Optimization

A foundational methodological boundary governs Phase C11.12:

- **Sensitivity Analysis, NOT Hyperparameter Optimization**: The goal is not to find a "better" set of coefficients, maximize an empirical index, or re-tune the router post-hoc.
- **Retaining Reference Configuration**: $\Theta_0 = (1.0, 1.5, 1.0, 0.5)$ remains the frozen reference architecture for subsequent phases.
- **Provisional $\delta = 0.20$**: Retained as the provisional DCRI uncertainty trade-off prior to dedicated selection in Phase C11.13.
- **Identical Packet Cohort**: Evaluated across exactly $N=500$ paired decision packets from the frozen cohort ($\text{seed}=115$).

---

## 3. Eight Pre-Specified Scientific Hypotheses

1. **Hypothesis H1 (Confidence Sensitivity $\alpha$)**:  
   *Prediction*: Increasing $\alpha$ increases the influence of per-packet model confidence differences on routing logits.  
   *Status*: **Supported** ($S_\alpha(w_R) = +0.062972$, $S_\alpha^{\text{norm}} = +0.125692$).

2. **Hypothesis H2 (Reliability Sensitivity $\beta$)**:  
   *Prediction*: Increasing $\beta$ systematically increases the decision authority allocated to modalities with higher validation reliability ($R_R=0.929956 > R_C=0.825382$).  
   *Status*: **Supported** ($S_\beta(w_R) = +0.013270$, $S_\beta(w_C) = -0.018214$).

3. **Hypothesis H3 (Uncertainty Sensitivity $\gamma$)**:  
   *Prediction*: Increasing $\gamma$ monotonically intensifies the routing penalty on modalities exhibiting higher predictive uncertainty without triggering numerical collapse.  
   *Status*: **Supported** (Monotonic logit penalty confirmed; mean entropy remains $\ge 0.9633\text{ nats}$).

4. **Hypothesis H4 (Quality Sensitivity $\eta$)**:  
   *Prediction*: Increasing $\eta$ linearly scales the quality bonus term in logit space according to $\Delta(z_i - z_j) = \Delta \eta (Q_i - Q_j)$.  
   *Status*: **Supported** (Max error $< 10^{-15}$ across all $N=500$ packets).

5. **Hypothesis H5 (Invariant Preservation)**:  
   *Prediction*: All 23 named configuration evaluations (representing 19 unique coefficient vectors) strictly preserve non-negativity ($w_i \ge 0$) and active simplex normalization ($\sum_{i \in \mathcal{A}} w_i = 1.000000$).  
   *Status*: **Confirmed** ($100.0\%$ packet compliance).

6. **Hypothesis H6 (Missing-Modality Invariance)**:  
   *Prediction*: Unavailable modalities ($A_i = 0$) retain $w_i = 0.0$ and exhibit zero cross-talk ($\Delta w_{\text{active}} = 0.000000$) under arbitrary masked-value perturbations.  
   *Status*: **Confirmed** (Max cross-talk deviation $= 0.0\times 10^0$).

7. **Hypothesis H7 (Reference Reproducibility)**:  
   *Prediction*: The frozen reference configuration $\Theta_0$ reproduces previously sealed ACARA-U routing behavior within numerical tolerance.  
   *Status*: **Confirmed** ($\bar{w}_R=0.501002, \bar{w}_F=0.260925, \bar{w}_C=0.238073, \bar{H}=1.011041$).

8. **Hypothesis H8 (Local Behavioral Stability)**:  
   *Prediction*: Perturbations within the tested reference neighborhood produce stable directional responses without routing collapse, invariant violations, or numerical instability.  
   *Status*: **Supported** (No routing collapse or numerical instability observed; entropy bounded in $[0.9633, 1.0460]\text{ nats}$ across 23 named evaluations / 19 unique coefficient vectors).

> **Statistical Interpretation Rule**: Paired bootstrap confidence intervals ($B=1,000$, $\text{seed}=115$) quantify directional uncertainty under controlled perturbations; exclusion of zero denotes a consistent directional parameter response within this experimental cohort, not population clinical significance.
