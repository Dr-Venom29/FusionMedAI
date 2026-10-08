# Routing Authority Sensitivity & Redistribution Under Calibration

## 1. Mathematical Formulation

Calibration affects the modality probability distribution vector $p_i$, which directly updates:
1. Continuous scalar risk projection: $r_i = f(p_i)$
2. Normalized router confidence: $C_i = \max_k(p_{i, k})$

The ACARA-U dynamic routing logit:

$$
z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i
$$

translates confidence softening ($\Delta C_i < 0$) into authority adjustment:

$$
\Delta w_i = w_i^{\text{cal}} - w_i^{\text{raw}}
$$

---

## 2. Authority Shift Across Active Modalities

Across the $N=500$ clean cohort (B5 Calibrated ACARA-U vs B2 Uncalibrated ACARA-U):

| Channel | Mean Uncalibrated Weight ($\overline{w_i^{\text{raw}}}$) | Mean Calibrated Weight ($\overline{w_i^{\text{cal}}}$) | Mean Authority Delta ($\overline{\Delta w_i}$) | $95\%$ Paired Bootstrap CI | Relative Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Retina ($\mathcal{M}_R$)** | $0.4529$ | $0.4373$ | $\mathbf{-0.0156}$ | $[-0.0168, -0.0144]$ | $-3.45\%$ |
| **Foot ($\mathcal{M}_F$)** | $0.1968$ | $0.2024$ | $\mathbf{+0.0055}$ | $[+0.0049, +0.0063]$ | $+2.82\%$ |
| **Clinical ($\mathcal{M}_C$)** | $0.3503$ | $0.3604$ | $\mathbf{+0.0101}$ | $[+0.0093, +0.0108]$ | $+2.88\%$ |

---

## 3. Simplex Conservation Invariant

The authority surrendered by the retinal channel is exactly conserved and absorbed by the foot and clinical channels:

$$
\sum_{i \in \mathcal{A}} \Delta w_i = (-0.015635) + (+0.005549) + (+0.010087) = 0.000001 \approx 0
$$

$$
\sum_{i \in \mathcal{A}} w_i = 1.000000 \quad (\text{Exact within } 10^{-6})
$$

### Routing Entropy Adjustment
The routing entropy $H(w)$ increases slightly from $1.0288$ to $1.0346$ ($\Delta H(w) = +0.0059$, $95\%$ CI: $[+0.0052, +0.0065]$), indicating a less concentrated routing authority distribution across available modalities.
