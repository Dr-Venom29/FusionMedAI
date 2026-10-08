# Empirical Results: Input Degradation Benchmark

## 1. Executive Summary of Primary Findings

Across the frozen cohort of $N=500$ controlled decision packets ($\text{seed}=115$):

1. **Quality Loss Magnitude (Severe D3)**:
   - **Retinal Fundus**: Mean severe quality loss of **$62.63\%$** ($\overline{\Delta Q_R} = -0.6175$) across 4 degradation operators.
   - **Diabetic Foot Ulcer**: Mean severe quality loss of **$67.14\%$** ($\overline{\Delta Q_F} = -0.6194$) across 4 operators.
   - **Structured Clinical EHR**: Mean severe quality loss of **$64.71\%$** ($\overline{\Delta Q_C} = -0.6471$) across 4 operators.

2. **Routing Authority Attenuation (Severe D3)**:
   - **Retinal Fundus**: Mean relative authority reduction ($\text{RAR}$) of **$35.18\%$** ($\overline{\Delta w_R} = -0.1739$).
   - **Diabetic Foot Ulcer**: Mean relative authority reduction ($\text{RAR}$) of **$50.46\%$** ($\overline{\Delta w_F} = -0.1276$).
   - **Structured Clinical EHR**: Mean relative authority reduction ($\text{RAR}$) of **$39.16\%$** ($\overline{\Delta w_C} = -0.0930$).

3. **Packet-Level Monotonicity**:
   - **Quality Degradation Monotonicity**: **$100.0\%$** of decision packets satisfy $Q^{(0)} \ge Q^{(1)} \ge Q^{(2)} \ge Q^{(3)}$.
   - **Routing Authority Monotonicity**: **$100.0\%$** of decision packets satisfy $w^{(0)} \ge w^{(1)} \ge w^{(2)} \ge w^{(3)}$.

4. **Scientific Quality Isolation (B6 vs B5)**:
   - The paired comparison showed significantly greater authority attenuation under B6 than B5 under severe degradation ($-0.1697$ vs $-0.0388$, paired difference $D = -0.1309$, $95\%$ bootstrap CI: $[-0.1319, -0.1300]$, strictly excluding zero), supporting the incremental contribution of the quality term within the tested controlled degradation benchmark.



---

## 2. Multi-Modality Cross-Degradation Scoreboard (Severe D3)

| Scenario | Degraded Modalities | Mean Fused Risk Shift ($\overline{\Delta R}$) | Mean DCRI Shift ($\overline{\Delta \text{DCRI}}$) | Mean Routing Entropy ($H$) | Weighted Conflict ($\overline{\sigma_w}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`R_degraded`** | Retina | $0.0494 \pm 0.0322$ | $0.0466 \pm 0.0360$ | $1.0700$ | $0.2264$ |
| **`F_degraded`** | Foot | $0.0567 \pm 0.0505$ | $0.0787 \pm 0.0570$ | $0.8933$ | $0.1849$ |
| **`C_degraded`** | Clinical | $0.0235 \pm 0.0190$ | $0.0176 \pm 0.0118$ | $0.9369$ | $0.2103$ |
| **`RF_degraded`** | Retina + Foot | $0.0530 \pm 0.0382$ | $0.1074 \pm 0.0412$ | $1.0033$ | $0.2038$ |
| **`RC_degraded`** | Retina + Clinical | $0.0560 \pm 0.0438$ | $0.0441 \pm 0.0330$ | $1.0319$ | $0.2245$ |
| **`FC_degraded`** | Foot + Clinical | $0.0600 \pm 0.0443$ | $0.0853 \pm 0.0587$ | $0.8158$ | $0.1804$ |
| **`RFC_degraded`** | All Three Channels | $0.0229 \pm 0.0194$ | $0.0981 \pm 0.0219$ | $0.9952$ | $0.2080$ |

---

## 3. Key Observations & Invariant Safety

1. **Graceful Multi-Modality Handling**: When all three modalities degrade simultaneously (`RFC_degraded`), the router maintains balanced relative weights ($H \approx 0.9952$) while the additive uncertainty penalty in DCRI correctly reflects increased epistemic burden ($\overline{\Delta \text{DCRI}} = 0.0981$).
2. **Hard-Mask Safety**: Unavailable channels ($A_i = 0$) receive strictly $w_i = 0.000000$ and zero-leakage perturbation invariance is mathematically certified.

