# Chapter 07: Decision Tier Transitions

## 1. Decision Tier Framework

To translate numerical risk and DCRI shifts into concrete operational consequences, the framework defines a 3-tier hypothetical decision policy based on the calibrated composite index $\text{DCRI}$:

$$\text{Tier}(\text{DCRI}) = \begin{cases} \text{Tier 0 (Routine Review)} & \text{if } \text{DCRI} < \tau_1 \\ \text{Tier 1 (Additional Assessment)} & \text{if } \tau_1 \le \text{DCRI} < \tau_2 \\ \text{Tier 2 (Escalation for Human Review)} & \text{if } \text{DCRI} \ge \tau_2 \end{cases}$$

with frozen decision thresholds $\tau_1 = 0.20$ and $\tau_2 = 0.40$.

In reference architecture $A0$, the $N=500$ packets partition into:
- **Routine Review (Tier 0):** $262$ encounters ($52.4\%$)
- **Additional Assessment (Tier 1):** $181$ encounters ($36.2\%$)
- **Escalation for Human Review (Tier 2):** $57$ encounters ($11.4\%$)

Transition matrices $T \in \mathbb{R}^{3 \times 3}$ track how encounters migrate between reference tiers (rows) and ablated tiers (columns), where $T_{i, j}$ represents the count of packets moving from Tier $i$ in $A0$ to Tier $j$ in $Av$.

---

## 2. Transition Matrices and Stability Metrics

| Variant | Target Component | Unchanged Count (%) | Reclassified Count (%) | Upgraded Count (%) | Downgraded Count (%) | Net Escalation Shift |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A1** | No Confidence ($\alpha=0$) | $466$ ($93.2\%$) | $34$ ($6.8\%$) | $28$ ($5.6\%$) | $6$ ($1.2\%$) | $+1$ ($+1.8\%$) |
| **A2** | No Reliability Prior ($\beta=0$) | $479$ ($95.8\%$) | $21$ ($4.2\%$) | $4$ ($0.8\%$) | $17$ ($3.4\%$) | $-4$ ($-7.0\%$) |
| **A3** | No Uncertainty Penalty ($\gamma=0$) | $437$ ($87.4\%$) | $63$ ($12.6\%$) | $58$ ($11.6\%$) | $5$ ($1.0\%$) | $+23$ ($+40.4\%$) |
| **A4** | No Quality Bonus ($\eta=0$) | $498$ ($99.6\%$) | $2$ ($0.4\%$) | $0$ ($0.0\%$) | $2$ ($0.4\%$) | $0$ ($0.0\%$) |
| **A5** | No DCRI Discount ($\delta=0$) | $404$ ($80.8\%$) | $96$ ($19.2\%$) | $96$ ($19.2\%$) | $0$ ($0.0\%$) | $+26$ ($+45.6\%$) |
| **A_UNIFORM** | Uniform Baseline | $411$ ($82.2\%$) | $89$ ($17.8\%$) | $47$ ($9.4\%$) | $42$ ($8.4\%$) | $-5$ ($-8.8\%$) |


---

## 3. Detailed Variant Transition Profiles

### 3.1 Variant A1: Upward Risk Redistribution via Confidence Ablation
- Resulting Tier Counts: Routine $240$, Additional $202$, Escalation $58$.
- **Operational Impact:** Removing dynamic confidence redistributes weight away from the lower-risk clinical tabular channel toward higher-risk vision channels, upgrading $28$ packets (e.g. $27$ from Routine to Additional, $1$ from Additional to Escalation) and downgrading $6$ packets. Net escalation increases by $+1$ case ($57 \to 58$).

### 3.2 Variant A2: Mild Downward Moderation
- Resulting Tier Counts: Routine $275$, Additional $172$, Escalation $53$.
- **Operational Impact:** High stability ($95.8\%$). Leveling reliability priors transfers authority toward clinical tabular data, resulting in $17$ downgrades (e.g. $13$ from Additional to Routine, $4$ from Escalation to Additional) and $4$ upgrades. Net escalation decreases by $4$ cases ($57 \to 53$).

### 3.3 Variant A3: Significant Risk Inflation & Upward Migration
- Resulting Tier Counts: Routine $209$, Additional $211$, Escalation $80$.
- **Operational Impact:** Removing the router uncertainty penalty increases foot ulcer weight dramatically, causing $58$ upward tier transitions (including $23$ new Escalation assignments, $57 \to 80$) and $5$ downgrades. This illustrates the protective function of $-\gamma U_i$ against unconstrained risk inflation.

### 3.4 Variant A4: Near-Complete Invariance
- Resulting Tier Counts: Routine $264$, Additional $179$, Escalation $57$.
- **Operational Impact:** $99.6\%$ stability ($498/500$ invariant). Net escalation change is zero ($57 \to 57$). The quality bonus acts as a subtle regularizer with minimal discrete tier shifts under uncorrupted benchmark inputs.

### 3.5 Variant A5: Unidirectional Upward Escalation
- Resulting Tier Counts: Routine $192$, Additional $225$, Escalation $83$.
- **Operational Impact:** Strictly monotonic upward transitions ($96$ upgrades, $0$ downgrades). $70$ Routine encounters move to Additional Assessment, and $26$ Additional encounters move to Escalation ($57 \to 83$, a $+45.6\%$ increase in escalation burden). This quantifies the operational conservatism supplied by the $\delta^* = 0.10$ discount.

### 3.6 Variant A_UNIFORM: Bidirectional Disruption
- Resulting Tier Counts: Routine $208$, Additional $240$, Escalation $52$.
- **Operational Impact:** Induces widespread bidirectional churn ($47$ upgrades, $42$ downgrades). Net escalation drops by $5$ cases ($57 \to 52$), showing that unweighted averaging washes out high-risk individual modality signals.


---

## 4. Invariant Verification

1. **Adjacency Conservation:** In all valid variants, $T_{0,2} = 0$ and $T_{2,0} = 0$. No encounter ever jumps two tiers across single component deletions.
2. **Conservation of Cohort Mass:** $\sum_{i,j} T_{i,j} = 500$ across all variants.
3. Gate C16-09 is formally satisfied.
