# Availability Regime Sensitivity & Hard-Mask Invariance

## 1. Cross-Regime Evaluation Structure

Every coefficient configuration in the parameter grid was evaluated across all seven valid availability regimes plus the empty modality set:
- **Unimodal Regimes**: Retina Only (R), Foot Only (F), Clinical Only (C)
- **Bimodal Regimes**: Retina + Foot (RF), Retina + Clinical (RC), Foot + Clinical (FC)
- **Trimodal Regime**: Retina + Foot + Clinical (RFC)
- **Empty Set**: Zero modalities available ($\emptyset$)

---

## 2. Cross-Regime Routing Invariance Matrix

### Table 12.3: Mean Modality Authority Across Availability Regimes Under Key Configurations

| Config | Regime | $\bar{w}_R$ | $\bar{w}_F$ | $\bar{w}_C$ | Mean $\bar{R}_{\text{fusion}}$ | Mean Entropy $\bar{H}$ | Active Simplex Sum |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1** ($\alpha=0.50$) | R | $1.000000$ | $0.000000$ | $0.000000$ | $0.252085$ | $0.000000$ | $1.000000$ |
| | RF | $0.651765$ | $0.348235$ | $0.000000$ | $0.288252$ | $0.639454$ | $1.000000$ |
| | RFC | $0.467300$ | $0.248055$ | $0.284645$ | $0.283040$ | $1.046031$ | $1.000000$ |
| **A3** ($\Theta_0$ Ref) | R | $1.000000$ | $0.000000$ | $0.000000$ | $0.252085$ | $0.000000$ | $1.000000$ |
| | F | $0.000000$ | $1.000000$ | $0.000000$ | $0.355938$ | $0.000000$ | $1.000000$ |
| | C | $0.000000$ | $0.000000$ | $1.000000$ | $0.261678$ | $0.000000$ | $1.000000$ |
| | RF | $0.660333$ | $0.339667$ | $0.000000$ | $0.287386$ | $0.635815$ | $1.000000$ |
| | RC | $0.677598$ | $0.000000$ | $0.322402$ | $0.255174$ | $0.624778$ | $1.000000$ |
| | FC | $0.000000$ | $0.521876$ | $0.478124$ | $0.310860$ | $0.686884$ | $1.000000$ |
| | RFC | $0.501002$ | $0.260925$ | $0.238073$ | $0.289900$ | $1.011041$ | $1.000000$ |
| **A5** ($\alpha=1.50$) | R | $1.000000$ | $0.000000$ | $0.000000$ | $0.252085$ | $0.000000$ | $1.000000$ |
| | RF | $0.662657$ | $0.337343$ | $0.000000$ | $0.287127$ | $0.630043$ | $1.000000$ |
| | RFC | $0.530272$ | $0.271724$ | $0.198003$ | $0.295378$ | $0.963261$ | $1.000000$ |
| **G1** ($\gamma=0.50$) | RFC | $0.462062$ | $0.315079$ | $0.222859$ | $0.304208$ | $1.034913$ | $1.000000$ |
| **G5** ($\gamma=1.50$) | RFC | $0.534345$ | $0.215668$ | $0.249987$ | $0.277888$ | $0.977226$ | $1.000000$ |

---

## 3. Mask Invariance and Edge Case Handling

1. **Strict Simplex Conservation**: For all 23 named evaluations (19 unique parameter vectors) across all 7 non-empty regimes, active weights sum identically to $1.000000 \pm 10^{-6}$.
2. **Strong Masked-Value Invariance**: When a modality channel is marked unavailable ($A_i = 0$), arbitrary variations in its confidence, reliability, uncertainty, quality, or continuous risk produce **exactly $0.000000$ change** in the active channels' weights and fused risk output across all 23 named evaluations (19 unique parameter vectors).
3. **Empty Modality Handling**: The zero-modality edge case safely returns `status = NO_MODALITY_AVAILABLE`, $w_i = 0.0$, and $H = 0.0$ without numerical faults across all parameter configurations.
