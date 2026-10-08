# Controlled Modality Distributions & Stratified Cohort Generation

## 1. Modality Combination Taxonomy

For the three diagnostic channels $\mathcal{M} = \{\text{Retina (R)}, \text{Foot (F)}, \text{Clinical (C)}\}$, there exist $2^3 = 8$ mutually exclusive modality availability configurations:

$$
\mathcal{C} = \{\text{RFC}, \text{RF}, \text{RC}, \text{FC}, \text{R}, \text{F}, \text{C}, \text{EMPTY}\}
$$

In Phase C11.9, the 7 non-empty combinations ($\mathcal{C}_{\text{active}}$) are subjected to controlled frequency distributions across a frozen cohort of $N=500$ decision packets ($\text{seed}=115$). The empty combination $\text{EMPTY}$ is evaluated as a boundary fail-closed check.

---

## 2. Pre-Registered Distribution Configurations

Three distinct frequency distributions are pre-registered to systematically test balanced, moderate, and heavy-tailed availability profiles:

### 2.1 Distribution D1: Balanced Benchmark (`D1_BALANCED`)
- **Profile**: Uniform probability across all 7 non-empty combinations ($P(c) = 1/7 \approx 14.2857\%$).
- **Packet Counts ($N=500$)**:
  - RFC: 72 packets ($14.4\%$)
  - RF: 72 packets ($14.4\%$)
  - RC: 72 packets ($14.4\%$)
  - FC: 71 packets ($14.2\%$)
  - R: 71 packets ($14.2\%$)
  - F: 71 packets ($14.2\%$)
  - C: 71 packets ($14.2\%$)
  - **Total**: 500 packets ($100.0\%$)

### 2.2 Distribution D2: Moderate Head-Tail Distribution (`D2_MODERATE_HEAD_TAIL`)
- **Profile**: Multi-modal head dominant, progressive decay across bimodal and unimodal tail regimes.
- **Probabilities & Packet Counts ($N=500$)**:
  - RFC: $35.0\%$ (175 packets)
  - RF: $25.0\%$ (125 packets)
  - RC: $15.0\%$ (75 packets)
  - FC: $10.0\%$ (50 packets)
  - R: $6.0\%$ (30 packets)
  - F: $5.0\%$ (25 packets)
  - C: $4.0\%$ (20 packets)
  - **Total**: 500 packets ($100.0\%$)

### 2.3 Distribution D3: Strong Long-Tail Distribution (`D3_STRONG_LONG_TAIL`)
- **Profile**: Controlled long-tail availability profile in which the tri-modal combination is assigned $50\%$ of the simulated cohort and unimodal channels form a sparse tail.
- **Probabilities & Packet Counts ($N=500$)**:
  - RFC: $50.0\%$ (250 packets)
  - RF: $25.0\%$ (125 packets)
  - RC: $10.0\%$ (50 packets)
  - FC: $8.0\%$ (40 packets)
  - R: $4.0\%$ (20 packets)
  - F: $2.0\%$ (10 packets)
  - C: $1.0\%$ (5 packets)
  - **Total**: 500 packets ($100.0\%$)

---

## 3. Stratified Deterministic Assignment Protocol

To eliminate sampling noise across distribution comparisons, packet allocation follows a deterministic stratified partitioning protocol:

1. The frozen cohort of $N=500$ packets is loaded in canonical sequential order ($k = 0, \dots, 499$).
2. For each distribution $D$, the exact target counts $\{N_c\}_{c \in \mathcal{C}_{\text{active}}}$ are partitioned contiguously across the 500 packets.
3. Each packet is transformed via `apply_combination_to_packet(packet, c)` which strictly enforces:
   - $A_i = \text{False} \implies Q_i = 0.0$
   - Active channels preserve exact risk $r_i$, confidence $C_i$, uncertainty $U_i$, and reliability $R_i$.

This guarantees exact cohort size conservation ($\sum_c N_c = 500$) and deterministic reproducibility.
