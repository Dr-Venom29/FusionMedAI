# Parameter Grid & Perturbation Taxonomy

## 1. Mathematical Structure of the Sensitivity Grid

The ACARA-U routing logit is parameterized by the 4-tuple $\Theta = (\alpha, \beta, \gamma, \eta) \in [0.0, 5.0]^4$:

$$z_i(\Theta) = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$$

To establish clear parameter attribution, Phase C11.12 implements a **One-Factor-At-A-Time (OFAT)** design sweeping each coefficient individually while holding all other terms locked at the reference values $\Theta_0 = (1.0, 1.5, 1.0, 0.5)$, complemented by a 3-level combined factorial test.

---

## 2. Complete 23-Evaluation Matrix (19 Unique Coefficient Vectors)

The grid comprises 23 named sensitivity evaluations representing 19 unique coefficient vectors across the 4 kernel terms and combined perturbations (with A3, B3, G3, Q3, and M all evaluating the frozen reference configuration $\Theta_0$):

| Config ID | Name | $\alpha$ | $\beta$ | $\gamma$ | $\eta$ | Sweep Category | Perturbation Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **A1** | $\alpha = 0.50$ | $0.50$ | $1.50$ | $1.00$ | $0.50$ | $\alpha$ sweep | Low confidence weighting ($-50\%$) |
| **A2** | $\alpha = 0.75$ | $0.75$ | $1.50$ | $1.00$ | $0.50$ | $\alpha$ sweep | Moderate-low confidence weighting ($-25\%$) |
| **A3** | $\alpha = 1.00$ (Ref) | $1.00$ | $1.50$ | $1.00$ | $0.50$ | $\alpha$ sweep | Frozen reference baseline ($0\%$) |
| **A4** | $\alpha = 1.25$ | $1.25$ | $1.50$ | $1.00$ | $0.50$ | $\alpha$ sweep | Moderate-high confidence weighting ($+25\%$) |
| **A5** | $\alpha = 1.50$ | $1.50$ | $1.50$ | $1.00$ | $0.50$ | $\alpha$ sweep | High confidence weighting ($+50\%$) |
| **B1** | $\beta = 1.00$ | $1.00$ | $1.00$ | $1.00$ | $0.50$ | $\beta$ sweep | Low reliability weighting ($-33\%$) |
| **B2** | $\beta = 1.25$ | $1.00$ | $1.25$ | $1.00$ | $0.50$ | $\beta$ sweep | Moderate-low reliability weighting ($-17\%$) |
| **B3** | $\beta = 1.50$ (Ref) | $1.00$ | $1.50$ | $1.00$ | $0.50$ | $\beta$ sweep | Frozen reference baseline ($0\%$) |
| **B4** | $\beta = 1.75$ | $1.00$ | $1.75$ | $1.00$ | $0.50$ | $\beta$ sweep | Moderate-high reliability weighting ($+17\%$) |
| **B5** | $\beta = 2.00$ | $1.00$ | $2.00$ | $1.00$ | $0.50$ | $\beta$ sweep | High reliability weighting ($+33\%$) |
| **G1** | $\gamma = 0.50$ | $1.00$ | $1.50$ | $0.50$ | $0.50$ | $\gamma$ sweep | Low uncertainty penalty ($-50\%$) |
| **G2** | $\gamma = 0.75$ | $1.00$ | $1.50$ | $0.75$ | $0.50$ | $\gamma$ sweep | Moderate-low uncertainty penalty ($-25\%$) |
| **G3** | $\gamma = 1.00$ (Ref) | $1.00$ | $1.50$ | $1.00$ | $0.50$ | $\gamma$ sweep | Frozen reference baseline ($0\%$) |
| **G4** | $\gamma = 1.25$ | $1.00$ | $1.50$ | $1.25$ | $0.50$ | $\gamma$ sweep | Moderate-high uncertainty penalty ($+25\%$) |
| **G5** | $\gamma = 1.50$ | $1.00$ | $1.50$ | $1.50$ | $0.50$ | $\gamma$ sweep | High uncertainty penalty ($+50\%$) |
| **Q1** | $\eta = 0.250$ | $1.00$ | $1.50$ | $1.00$ | $0.250$ | $\eta$ sweep | Low quality bonus ($-50\%$) |
| **Q2** | $\eta = 0.375$ | $1.00$ | $1.50$ | $1.00$ | $0.375$ | $\eta$ sweep | Moderate-low quality bonus ($-25\%$) |
| **Q3** | $\eta = 0.500$ (Ref) | $1.00$ | $1.50$ | $1.00$ | $0.500$ | $\eta$ sweep | Frozen reference baseline ($0\%$) |
| **Q4** | $\eta = 0.625$ | $1.00$ | $1.50$ | $1.00$ | $0.625$ | $\eta$ sweep | Moderate-high quality bonus ($+25\%$) |
| **Q5** | $\eta = 0.750$ | $1.00$ | $1.50$ | $1.00$ | $0.750$ | $\eta$ sweep | High quality bonus ($+50\%$) |
| **L** | Combined Low | $0.75$ | $1.25$ | $0.75$ | $0.375$ | Combined | Simultaneous moderate low scaling ($-25\%$) |
| **M** | Combined Ref | $1.00$ | $1.50$ | $1.00$ | $0.500$ | Combined | Simultaneous reference configuration ($\Theta_0$) |
| **H** | Combined High | $1.25$ | $1.75$ | $1.25$ | $0.625$ | Combined | Simultaneous moderate high scaling ($+25\%$) |

---

## 3. Analytical Logit Derivative Semantics

Because the ACARA-U logit is linear in the coefficients, the change in the relative logit between any two modalities $i$ and $j$ under a parameter change $\Delta \Theta$ is exact:

$$\Delta(z_i - z_j) = \Delta \alpha (C_i - C_j) + \Delta \beta (R_i - R_j) - \Delta \gamma (U_i - U_j) + \Delta \eta (Q_i - Q_j)$$

This exact derivative structure ensures that each hyperparameter strictly modulates its intended physical channel attribute:
- $\alpha$ scales the sensitivity to instantaneous prediction confidence differences $(C_i - C_j)$.
- $\beta$ scales the authority bias toward modalities with higher validation reliability $(R_i - R_j)$.
- $\gamma$ penalizes channels exhibiting higher predictive uncertainty $(U_i - U_j)$.
- $\eta$ awards authority bonuses to channels with cleaner input quality $(Q_i - Q_j)$.
