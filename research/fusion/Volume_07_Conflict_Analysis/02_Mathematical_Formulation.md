# Mathematical Formulation of Cross-Modality Conflict

## 1. Pairwise Disagreement Metric ($X_{jk}$)

For an active modality set $\mathcal{A} \subseteq \{\text{retina}, \text{foot}, \text{clinical}\}$ with $|\mathcal{A}| \ge 2$, each modality $i \in \mathcal{A}$ outputs a calibrated risk projection $r_i \in [0.0, 1.0]$.

The directional signed difference between modalities $j$ and $k$ is:

$$D_{jk} = r_j - r_k \in [-1.0, 1.0]$$

The pairwise absolute disagreement is:

$$X_{jk} = |r_j - r_k| = |D_{jk}| \in [0.0, 1.0]$$

### Fundamental Invariants
1. **Symmetry**: $X_{jk} \equiv X_{kj}$ and $D_{jk} \equiv -D_{kj}$
2. **Zero Disagreement Identity**: $r_j = r_k \iff X_{jk} = 0.0$
3. **Tri-Modal Combinations**: $P = \binom{|\mathcal{A}|}{2} = 3$ pairs ($X_{RF}, X_{RC}, X_{FC}$).

---

## 2. Maximum Conflict Magnitude ($\Delta_{\max}$)

The primary conflict magnitude is defined as the supremum of pairwise disagreement across active channels:

$$\Delta_{\max} = \max_{j, k \in \mathcal{A}, j < k} |r_j - r_k| \in [0.0, 1.0]$$

If $|\mathcal{A}| < 2$, $\Delta_{\max}$ is formally `None` (discordance unavailable).

---

## 3. Mean Pairwise Disagreement ($\Delta_{\text{mean}}$)

The unweighted average disagreement across all active pairs is:

$$\Delta_{\text{mean}} = \frac{1}{P} \sum_{j < k, j,k \in \mathcal{A}} |r_j - r_k| \in [0.0, 1.0]$$

where $P = \binom{|\mathcal{A}|}{2}$. By definition:

$$0.0 \le \Delta_{\text{mean}} \le \Delta_{\max} \le 1.0$$

---

## 4. Weighted Consensus Dispersion ($V_w$ and $\sigma_w$)

To evaluate risk dispersion with respect to ACARA-U decision authority, the weighted variance around the fused risk consensus $R_{\text{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i$ is:

$$V_w = \sum_{i \in \mathcal{A}} w_i (r_i - R_{\text{fusion}})^2 \in [0.0, 0.25]$$

The weighted standard deviation (returning dispersion to the risk unit scale) is:

$$\sigma_w = \sqrt{V_w} = \sqrt{\sum_{i \in \mathcal{A}} w_i (r_i - R_{\text{fusion}})^2} \in [0.0, 0.5]$$

---

## 5. Weight Entropy & Dominance Metrics

To measure authority concentration under conflict:

$$H(w) = -\sum_{i \in \mathcal{A}} w_i \ln(w_i) \in [0.0, \ln|\mathcal{A}|]$$

$$w_{\max} = \max_{i \in \mathcal{A}} w_i \in \left[\frac{1}{|\mathcal{A}|}, 1.0\right]$$

---

## 6. Universal Ordering Invariant for Supported Modality Counts

For all supported multi-modality configurations ($M \in \{2, 3\}$) and any valid probability simplex weights $w$, the conflict metrics satisfy a strict mathematical ordering invariant:

$$\boxed{\Delta_{\max} \ge \Delta_{\text{mean}} \ge \sigma_w}$$

### Mathematical Proof:
1. **Bimodal Case ($M = 2$)**:
   - $\Delta_{\max} = \Delta_{\text{mean}} = |r_1 - r_2| = X$.
   - Fused risk $R_{\text{fusion}} = w_1 r_1 + w_2 r_2$, with weighted variance $V_w = w_1 w_2 X^2$.
   - Weighted standard deviation $\sigma_w = \sqrt{w_1 w_2} X \le \sqrt{0.25} X = 0.5 X$.
   - Therefore: $\Delta_{\max} = \Delta_{\text{mean}} \ge 2 \sigma_w \ge \sigma_w$.

2. **Tri-Modal Case ($M = 3$)**:
   - For any ordered risks $r_{(1)} \le r_{(2)} \le r_{(3)}$, the maximum disagreement is $\Delta_{\max} = r_{(3)} - r_{(1)}$.
   - The sum of pairwise differences is:
     $$(r_{(2)} - r_{(1)}) + (r_{(3)} - r_{(2)}) + (r_{(3)} - r_{(1)}) = 2(r_{(3)} - r_{(1)}) = 2 \Delta_{\max}$$
   - Mean pairwise disagreement is therefore identically:
     $$\Delta_{\text{mean}} = \frac{2}{3} \Delta_{\max} \approx 0.666667 \cdot \Delta_{\max}$$
   - By Popoviciu's variance inequality, the maximum weighted variance of any probability measure on an interval of length $\Delta_{\max}$ is $V_w \le \frac{1}{4}\Delta_{\max}^2 \implies \sigma_w \le 0.5 \Delta_{\max}$.
   - Since $0.5 < \frac{2}{3}$:
     $$\sigma_w \le 0.5 \Delta_{\max} < \frac{2}{3} \Delta_{\max} = \Delta_{\text{mean}} \le \Delta_{\max}$$

This universal ordering is verified across every packet in the cohort by Deep Verification Gate 12.
