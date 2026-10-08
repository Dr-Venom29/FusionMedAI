# Mathematical Formulation of DCRI Risk Aggregation

## 1. Convex Risk Combination

For a decision instance with active modality set $\mathcal{A} \subseteq \{\text{retina}, \text{foot}, \text{clinical}\}$, where $|\mathcal{A}| \ge 1$, each available modality contributes a normalized scalar risk projection $r_i \in [0.0, 1.0]$.

The dynamic router allocates non-negative authority weights $w_i \ge 0$ satisfying the partition of unity:
$$\sum_{i \in \mathcal{A}} w_i = 1.0, \quad w_j = 0.0 \quad \forall j \notin \mathcal{A}$$

The individual weighted risk contribution $K_i$ is defined as:
$$K_i = w_i r_i$$

The fused continuous risk index $R_{\text{fusion}}$ is the convex combination:
$$R_{\text{fusion}} = \sum_{i \in \mathcal{A}} K_i = \sum_{i \in \mathcal{A}} w_i r_i$$

### Boundary Invariant
Because $0 \le r_i \le 1$ and $\sum w_i = 1$ with $w_i \ge 0$:
$$0.0 \le \min_{i \in \mathcal{A}} r_i \le R_{\text{fusion}} \le \max_{i \in \mathcal{A}} r_i \le 1.0$$
Hence, $R_{\text{fusion}} \in [0.0, 1.0]$ unconditionally.

---

## 2. Uncertainty Burden & Discount Penalty

Each available modality produces a normalized predictive uncertainty metric $U_i \in [0.0, 1.0]$. The cumulative uncertainty burden across active channels is:
$$U_{\text{sum}} = \sum_{i \in \mathcal{A}} U_i$$
with corresponding mean uncertainty:
$$U_{\text{mean}} = \frac{1}{|\mathcal{A}|} \sum_{i \in \mathcal{A}} U_i$$

Given a scaling coefficient $\delta \ge 0$, the total uncertainty discount penalty $P_U(\delta)$ is:
$$P_U(\delta) = \delta U_{\text{sum}} = \delta \sum_{i \in \mathcal{A}} U_i$$
with per-modality penalty contributions:
$$p_i(\delta) = \delta U_i \implies \sum_{i \in \mathcal{A}} p_i(\delta) = P_U(\delta)$$

---

## 3. Decision-Critical Risk Index (DCRI)

The derived decision index is defined by subtracting the uncertainty penalty from the aggregated risk:
$$\text{DCRI}_\delta = R_{\text{fusion}} - P_U(\delta) = \sum_{i \in \mathcal{A}} w_i r_i - \delta \sum_{i \in \mathcal{A}} U_i$$

### Bounding and the Non-Clamping Specification
Since $R_{\text{fusion}} \in [0.0, 1.0]$ and $U_i \in [0.0, 1.0]$ with up to $M = |\mathcal{A}| \le 3$ active modalities:
$$0 \le P_U(\delta) \le \delta M$$
$$\implies \text{DCRI}_\delta \in [-\delta M, 1.0]$$

> [!IMPORTANT]
> **Specification on Negative DCRI**:
> DCRI is a derived decision metric representing risk adjusted for epistemic uncertainty. When predictive uncertainty is substantial and baseline risk is modest, $\text{DCRI}_\delta$ naturally takes negative values. In the frozen $N=500$ cohort, the observed minimum at $\delta=0.20$ was $-0.125243$ (against the theoretical lower bound $-\delta M = -0.60$ for tri-modal bundles). Phase C11.6 strictly preserves the true mathematical range and prohibits post-hoc artificial clamping to $[0, 1]$.

---

## 4. Fundamental Mathematical Invariants

1. **Conservation of Contributions**:
   $$\sum_{i \in \mathcal{A}} K_i = R_{\text{fusion}}$$
2. **Conservation of Penalties**:
   $$\sum_{i \in \mathcal{A}} p_i(\delta) = P_U(\delta)$$
3. **Zero-Uncertainty Equality**:
   $$U_i = 0 \quad \forall i \in \mathcal{A} \implies \text{DCRI}_\delta = R_{\text{fusion}} \quad \forall \delta \ge 0$$
4. **Zero-Delta Invariant**:
   $$\delta = 0.0 \implies \text{DCRI}_0 = R_{\text{fusion}}$$
5. **Delta Monotonicity**:
   $$\delta_2 > \delta_1 \ge 0 \land U_{\text{sum}} > 0 \implies \text{DCRI}_{\delta_2} < \text{DCRI}_{\delta_1}$$
6. **Fixed-Router Uncertainty Monotonicity**:
   Holding $w_i$, $r_i$, and the active modality set $\mathcal{A}$ fixed, if $U_{\text{sum}}' > U_{\text{sum}}$ and $\delta > 0$, then:
   $$\text{DCRI}' - \text{DCRI} = -\delta (U_{\text{sum}}' - U_{\text{sum}}) < 0 \implies \text{DCRI}' < \text{DCRI}$$
