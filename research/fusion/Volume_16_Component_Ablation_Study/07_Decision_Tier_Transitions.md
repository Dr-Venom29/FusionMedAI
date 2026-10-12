# Chapter 07: Decision Tier Transitions

## 1. Decision Tier Framework

To translate numerical risk and DCRI shifts into concrete operational consequences, the framework defines a 3-tier hypothetical decision policy based on the calibrated composite index $\text{DCRI}$:

$$\text{Tier}(\text{DCRI}) = \begin{cases} \text{Tier 0 (Routine Review)} & \text{if } \text{DCRI} < \tau_1 \\ \text{Tier 1 (Additional Assessment)} & \text{if } \tau_1 \le \text{DCRI} < \tau_2 \\ \text{Tier 2 (Escalation for Human Review)} & \text{if } \text{DCRI} \ge \tau_2 \end{cases}$$

with frozen decision thresholds $\tau_1 = 0.20$ and $\tau_2 = 0.40$.

In reference architecture $A0$, the $N=500$ packets partition into:
- **Routine Review (Tier 0):** $232$ encounters ($46.4\%$)
- **Additional Assessment (Tier 1):** $187$ encounters ($37.4\%$)
- **Escalation for Human Review (Tier 2):** $81$ encounters ($16.2\%$)

Transition matrices $T \in \mathbb{R}^{3 \times 3}$ track how encounters migrate between reference tiers (rows) and ablated tiers (columns), where $T_{i, j}$ represents the count of packets moving from Tier $i$ in $A0$ to Tier $j$ in $Av$.

---

## 2. Transition Matrices and Stability Metrics

| Variant | Target Component | Unchanged Count (%) | Upgraded Count (%) | Downgraded Count (%) | Net Escalation Shift | Paradoxical 2-Tier Shifts |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A1** | No Confidence ($\alpha=0$) | $462$ ($92.4\%$) | $3$ ($0.6\%$) | $35$ ($7.0\%$) | $-23$ ($-28.4\%$) | $0$ ($0.0\%$) |
| **A2** | No Reliability Prior ($\beta=0$) | $487$ ($97.4\%$) | $0$ ($0.0\%$) | $13$ ($2.6\%$) | $-8$ ($-9.9\%$) | $0$ ($0.0\%$) |
| **A3** | No Uncertainty Penalty ($\gamma=0$) | $439$ ($87.8\%$) | $56$ ($11.2\%$) | $5$ ($1.0\%$) | $+15$ ($+18.5\%$) | $0$ ($0.0\%$) |
| **A4** | No Quality Bonus ($\eta=0$) | $497$ ($99.4\%$) | $2$ ($0.4\%$) | $1$ ($0.2\%$) | $0$ ($0.0\%$) | $0$ ($0.0\%$) |
| **A5** | No DCRI Discount ($\delta=0$) | $408$ ($81.6\%$) | $92$ ($18.4\%$) | $0$ ($0.0\%$) | $+36$ ($+44.4\%$) | $0$ ($0.0\%$) |
| **A_UNIFORM** | Uniform Baseline | $419$ ($83.8\%$) | $38$ ($7.6\%$) | $43$ ($8.6\%$) | $-29$ ($-35.8\%$) | $0$ ($0.0\%$) |

---

## 3. Detailed Variant Transition Profiles

### 3.1 Variant A1: De-escalation via Tabular Reallocation
Transition Matrix:
$$\begin{pmatrix} 230 & 2 & 0 \\ 11 & 175 & 1 \\ 0 & 24 & 57 \end{pmatrix}$$
- Resulting Tier Counts: Routine $241$, Additional $201$, Escalation $58$.
- **Operational Impact:** Removing confidence causes $24$ encounters previously categorized in Tier 2 (Escalation) to downgrade to Tier 1 (Additional Assessment), and $11$ encounters to downgrade from Tier 1 to Tier 0 (Routine Review). Net escalation drops by $28.4\%$.

### 3.2 Variant A2: Mild Downward Moderation
Transition Matrix:
$$\begin{pmatrix} 232 & 0 & 0 \\ 5 & 182 & 0 \\ 0 & 8 & 73 \end{pmatrix}$$
- Resulting Tier Counts: Routine $237$, Additional $190$, Escalation $73$.
- **Operational Impact:** High stability ($97.4\%$). Exactly $8$ encounters downgrade from Tier 2 to Tier 1, and $5$ downgrade from Tier 1 to Tier 0. Zero upward upgrades occur.

### 3.3 Variant A3: Significant Risk Inflation & Upward Migration
Transition Matrix:
$$\begin{pmatrix} 192 & 40 & 0 \\ 4 & 167 & 16 \\ 0 & 1 & 80 \end{pmatrix}$$
- Resulting Tier Counts: Routine $196$, Additional $208$, Escalation $96$.
- **Operational Impact:** $40$ Tier 0 encounters upgrade to Tier 1, and $16$ Tier 1 encounters upgrade to Tier 2. Net escalation surges from $81$ to $96$ ($+18.5\%$, net $+15$ assignments). Removing the router uncertainty penalty increases escalation review burden by over-weighting high-uncertainty foot ulcer channels.

### 3.4 Variant A4: Near-Complete Invariance
Transition Matrix:
$$\begin{pmatrix} 231 & 1 & 0 \\ 0 & 186 & 1 \\ 0 & 1 & 80 \end{pmatrix}$$
- Resulting Tier Counts: Routine $231$, Additional $188$, Escalation $81$.
- **Operational Impact:** $99.4\%$ stability. Net escalation change is zero ($81 \to 81$). Quality bonus has virtually no impact on discrete decision tiers under standard benchmark inputs.

### 3.5 Variant A5: Unidirectional Upward Escalation
Transition Matrix:
$$\begin{pmatrix} 176 & 56 & 0 \\ 0 & 151 & 36 \\ 0 & 0 & 81 \end{pmatrix}$$
- Resulting Tier Counts: Routine $176$, Additional $207$, Escalation $117$.
- **Operational Impact:** Strictly monotonic upward transitions ($92$ upgrades, $0$ downgrades). $56$ Tier 0 encounters move to Tier 1, and $36$ Tier 1 encounters move to Tier 2 ($+44.4\%$ increase in escalation assignments). This quantifies the exact operational buffer provided by the $\delta^* = 0.10$ discount.

### 3.6 Variant A_UNIFORM: Bidirectional Disruption
Transition Matrix:
$$\begin{pmatrix} 194 & 38 & 0 \\ 14 & 173 & 0 \\ 0 & 29 & 52 \end{pmatrix}$$
- Resulting Tier Counts: Routine $208$, Additional $240$, Escalation $52$.
- **Operational Impact:** Induces widespread bidirectional churn ($38$ upgrades, $43$ downgrades). Net escalation drops by $29$ cases ($-35.8\%$), showing that unweighted averaging washes out high-risk individual modality signals.

---

## 4. Invariant Verification

1. **Adjacency Conservation:** In all valid variants, $T_{0,2} = 0$ and $T_{2,0} = 0$. No encounter ever jumps two tiers across single component deletions.
2. **Conservation of Cohort Mass:** $\sum_{i,j} T_{i,j} = 500$ across all variants.
3. Gate C16-09 is formally satisfied.
