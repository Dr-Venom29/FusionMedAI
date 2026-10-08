# Mathematical Formulation of Missing Modality Robustness

## 1. Hard Availability Mask Layer

Let $\mathcal{M} = \{\text{retina}, \text{foot}, \text{clinical}\}$ represent the set of all supported modality channels, with cardinality $N_M = 3$. Each modality $i \in \mathcal{M}$ possesses a binary availability flag $A_i \in \{0, 1\}$.

The active subset of available modalities is:

$$\mathcal{A} = \{i \in \mathcal{M} \mid A_i = 1\}$$

with active count $M = |\mathcal{A}| \in \{0, 1, 2, 3\}$.

### Dynamic Router Masking Kernel:
For each channel $i \in \mathcal{M}$, the unmasked router logit is computed via the frozen ACARA-U scoring function:

$$z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$$

Under hard availability masking, the masked logit $\tilde{z}_i$ is defined as:

$$\tilde{z}_i = \begin{cases} z_i & \text{if } A_i = 1 \\ -\infty & \text{if } A_i = 0 \end{cases}$$

Applying numerically stable softmax over $\mathcal{M}$ yields the authority weights $w_i$:

$$w_i = \begin{cases} \frac{\exp(\tilde{z}_i - m)}{\sum_{j \in \mathcal{A}} \exp(\tilde{z}_j - m)} & \text{if } A_i = 1 \\ 0.0 & \text{if } A_i = 0 \end{cases}$$

where $m = \max_{j \in \mathcal{A}} \tilde{z}_j$.

### Fundamental Invariants:
1. **Zero Unavailable Authority**: $A_i = 0 \implies w_i = 0.0$
2. **Active Simplex Conservation**: For any $M \ge 1$, $\sum_{i \in \mathcal{A}} w_i = 1.0$ and $w_i \in [0.0, 1.0]$
3. **Fail-Closed Zero State**: For $M = 0$, $w_i = 0.0 \; \forall i \in \mathcal{M}$, status = `NO_MODALITY_AVAILABLE`

---

## 2. Authority Redistribution Metric ($\Delta w_j$)

When transitioning from the full tri-modal regime $\mathcal{A}_{\text{full}} = \mathcal{M}$ to a subset regime $\mathcal{A}_{\text{subset}} \subset \mathcal{M}$, the authority shift allocated to each modality $j \in \mathcal{M}$ is:

$$\Delta w_j = w_j^{\text{subset}} - w_j^{\text{full}}$$

### Conservation Law:
When exactly one modality $k$ is removed, the relinquished authority is conserved across the remaining active modalities:

$$\sum_{j \in \mathcal{A}_{\text{subset}}} \Delta w_j = w_k^{\text{full}} = -\Delta w_k$$

For multi-modality removal, the generalized conservation relation across the active subset is:

$$\sum_{j \in \mathcal{A}_{\text{subset}}} \Delta w_j = \sum_{k \in \mathcal{A}_{\text{full}} \setminus \mathcal{A}_{\text{subset}}} w_k^{\text{full}}$$

---

## 3. Risk Continuity & Sensitivity ($\Delta R_{\text{missing}}$)

The decision-level fused risk under availability subset $\mathcal{A}$ is:

$$R_{\text{fusion}}^{\mathcal{A}} = \sum_{i \in \mathcal{A}} w_i r_i \in [0.0, 1.0]$$

The risk sensitivity induced by modality loss is defined as:

$$\Delta R_{\text{missing}} = |R_{\text{fusion}}^{\text{full}} - R_{\text{fusion}}^{\text{subset}}| \in [0.0, 1.0]$$

$$\Delta R_{\text{signed}} = R_{\text{fusion}}^{\text{full}} - R_{\text{fusion}}^{\text{subset}} \in [-1.0, 1.0]$$

---

## 4. DCRI Sensitivity & Exact Signed Decomposition

Under uncertainty discount hyperparameter $\delta \ge 0.0$, the decision-critical risk index is:

$$\text{DCRI}_\delta^{\mathcal{A}} = R_{\text{fusion}}^{\mathcal{A}} - \delta \sum_{i \in \mathcal{A}} U_i = R_{\text{fusion}}^{\mathcal{A}} - \delta U_{\text{sum}}^{\mathcal{A}}$$

The signed shift in $\text{DCRI}_\delta$ decomposes into risk authority shift and uncertainty discount shift:

$$\begin{aligned}
\Delta \text{DCRI}_{\text{signed}} &= \text{DCRI}_\delta^{\text{full}} - \text{DCRI}_\delta^{\text{subset}} \\
&= \left(R_{\text{fusion}}^{\text{full}} - \delta U_{\text{sum}}^{\text{full}}\right) - \left(R_{\text{fusion}}^{\text{subset}} - \delta U_{\text{sum}}^{\text{subset}}\right) \\
&= \underbrace{\left(R_{\text{fusion}}^{\text{full}} - R_{\text{fusion}}^{\text{subset}}\right)}_{\Delta R_{\text{signed}}} - \delta \underbrace{\left(U_{\text{sum}}^{\text{full}} - U_{\text{sum}}^{\text{subset}}\right)}_{\Delta U_{\text{sum}}} \\
&= \Delta R_{\text{signed}} - \Delta P_U
\end{aligned}$$

where $\Delta P_U = \delta \Delta U_{\text{sum}}$ represents the change in the cumulative uncertainty penalty.
