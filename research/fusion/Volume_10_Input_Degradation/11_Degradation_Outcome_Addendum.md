# Addendum: Outcome-Grounded Degradation Accuracy & Utility (B6 vs B5)

> **Evaluation Addendum to Volume 10**  
> **Status:** 🟢 SEALED & VERIFIED (8/8 Outcome Gates Passed)  
> **Evaluation Sample:** $N=5,000$ Confirmatory Packets (10 Independent Seeds `301`–`310`)  
> **Reference Parameters:** $\Theta_0 = (1.0, 1.5, 1.0, 0.5)$, $\delta^* = 0.10$  

---

## 1. Research Question & Rationale

Volume 10 established that under 12 controlled degradation operators, **ACARA-U (B6)** attenuates degraded-channel routing authority more aggressively than B5 ($\mathrm{RAR} = 35.2\%\text{--}50.5\%$, paired difference $\Delta w = -0.1309$, $95\%$ CI: $[-0.1319, -0.1300]$).

However, authority attenuation alone does not prove that the resulting fused estimate is closer to the true patient risk. This addendum evaluates:
> **Does quality-aware routing authority attenuation actually translate into lower prediction error (MAE) against a known oracle target $Y^*$, and what happens when quality sensors fail?**

---

## 2. Oracle Error Across Degradation Severities ($N=5,000$ Synthetic Packets)

| Degradation Scenario | Evaluated Packets | B5 MAE | B6 MAE (Full ACARA-U) | Paired $\Delta_{\mathrm{MAE}} (\mathrm{B6} - \mathrm{B5})$ | Relative MAE Reduction (%)<br>*(+ indicates improvement, − indicates increased error)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Inputs (D0, Reference)** | $3,022$ | $0.055285$ | $0.055483$ | $+0.000198$ | **$-0.36\%$** (slight error increase) |
| **Mild Degradation (D1)** | $359$ | $0.060292$ | $0.058297$ | $-0.001994$ | **$+3.31\%$** |
| **Moderate Degradation (D2)** | $805$ | $0.074864$ | $0.070961$ | $-0.003904$ | **$+5.21\%$** |
| **Severe Degradation (D3)** | $385$ | $0.089828$ | $0.074412$ | **$-0.015416$** | **$+17.16\%$** |
| **Uncertainty Miscalibration** | $429$ | $0.049440$ | $0.049397$ | $-0.000043$ | **$+0.09\%$** |

---

## 3. Key Findings on Degradation Dynamics

1. **Monotonic Error Reduction Scaling Across Degradation Tiers:**
   Across mild, moderate, and severe degradation conditions, the predictive accuracy advantage of B6 over B5 grows strictly monotonically as input corruption worsens:
   $$\Delta_{\mathrm{MAE}}^{\mathrm{severe}} (-0.015416) < \Delta_{\mathrm{MAE}}^{\mathrm{moderate}} (-0.003904) < \Delta_{\mathrm{MAE}}^{\mathrm{mild}} (-0.001994) < 0$$
   Under severe single-channel corruption, B6 reduces mean absolute prediction error by **$17.16\%$** relative to B5 by shifting decision authority to uncorrupted modalities. Clean uncorrupted inputs serve as a separate baseline reference condition where B6 incurs a minor $+0.36\%$ error difference due to slight weight smoothing.

2. **Quality Sensor Failure Modes (Imperfect Sensing):**
   - **Accurate Detection (`ACCURATE`, $N=3,775$):** When quality sensors accurately detect modality status, B6 delivers substantial overall error reduction ($\Delta_{\mathrm{MAE}} = -0.002544$).
   - **Undetected Degradation (`MISLEADING_UNNOTICED`, $N=406$):** When severe degradation corrupts a modality but the quality sensor fails to detect it ($Q_i \approx 0.90$), B6 cannot attenuate channel authority. In this condition, B6 defaults to B5 error parity ($\Delta_{\mathrm{MAE}} = -0.000417$).
   - **False Alarm Attenuation (`MISLEADING_FALSE_ALARM`, $N=390$):** When an uncorrupted modality receives an erroneous low quality score ($Q_i \approx 0.25$), B6 unnecessarily attenuates that modality's weight, causing a small error penalty ($\Delta_{\mathrm{MAE}} = +0.001481$) if alternative modalities have higher baseline variance.
   - **Uncertainty Miscalibration (`UNCERTAINTY_MISCALIBRATED`, $N=429$):** When uncertainty estimates are uncorrelated/inverted relative to true observation error, B6 remains robust and maintains near-parity ($\Delta_{\mathrm{MAE}} = -0.000043$).

---

## 4. Scientific Conclusion

In simulated degradation scenarios, quality-aware routing (B6) provides substantial predictive-error reduction against input corruption, with the largest benefit observed during severe degradation. However, this advantage is conditional on the synthetic simulation framework, degradation operators, and upstream quality-sensor accuracy; if quality sensors fail to detect corruption, B6 reverts to B5 performance.

