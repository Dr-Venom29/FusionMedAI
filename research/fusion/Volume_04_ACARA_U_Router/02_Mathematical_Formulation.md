# 02 ACARA-U Router Mathematical Formulation — Phase C11.4

## 1. Modality Routing Scoring Function

For each modality $i \in \mathcal{M} = \{\text{Retina}, \text{Foot}, \text{Clinical}\}$, the pre-masking logit score $z_i \in \mathbb{R}$ is computed as:

$$z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$$

Where:
- $C_i \in [0.0, 1.0]$: Instance-level model confidence.
- $R_i \in [0.0, 1.0]$: Frozen empirical validation reliability prior ($R_R=0.929956, R_F=0.922266, R_C=0.825382$).
- $U_i \in [0.0, 1.0]$: Instance-level predictive uncertainty mapped to the common directional routing scale. The underlying uncertainty statistic is modality-specific and is not assumed to be statistically equivalent across modalities.
- $Q_i \in [0.0, 1.0]$: Instance-level input signal quality.
- $\Theta = (\alpha, \beta, \gamma, \eta)$ with $\alpha, \beta, \gamma, \eta \ge 0$.

---

## 2. Hard Availability Masking

To strictly eliminate influence from unmeasured or corrupted missing modalities ($A_i = 0$), the raw logit is masked:

$$\tilde{z}_i = \begin{cases} z_i & \text{if } A_i = 1 \\ -\infty & \text{if } A_i = 0 \end{cases}$$

---

## 3. Numerically Stable Softmax Normalization

Let $\mathcal{A} = \{i \in \mathcal{M} \mid A_i = 1\}$ denote the set of active modalities.

When $|\mathcal{A}| \ge 1$, weights are normalized via the max-subtracted softmax:

$$m = \max_{j \in \mathcal{A}} z_j$$

$$w_i = \begin{cases} \frac{\exp(z_i - m)}{\sum_{j \in \mathcal{A}} \exp(z_j - m)} & \text{if } i \in \mathcal{A} \\ 0.0 & \text{if } i \notin \mathcal{A} \end{cases}$$

### Zero-Modality Case

If $|\mathcal{A}|=0$, no softmax normalization is performed. The router returns zero weights for all modalities and the status `NO_MODALITY_AVAILABLE`.

This is a safe-rejection state rather than a valid fusion state.

### Mathematical Properties:
1. **Convex Weighting**: $\sum_{i \in \mathcal{A}} w_i = 1.0$.
2. **Strict Non-negativity & Boundedness**: $0.0 \le w_i \le 1.0 \quad \forall i$.
3. **Hard Masking Invariant**: $A_i = 0 \implies w_i = 0.0$ strictly.
4. **Numerical Stability**: Overflow is mathematically prevented as $\max_j (z_j - m) = 0 \implies \exp(z_j - m) \le 1.0$.

---

## 4. Routing Diagnostics: Weight Entropy

The router outputs the **Modality Weight Entropy** $H(w)$ to quantify authority dispersion:

$$H(w) = -\sum_{i \in \mathcal{A}, w_i > 0} w_i \ln(w_i)$$

- $H(w) = 0.0$: Complete authority concentration on a single modality (e.g. unimodal operation or extreme logit dominance).
- $H(w) = \ln(3) \approx 1.0986$: Perfect authority dispersion across all three modalities (uniform weighting).
- **Scope Note**: Routing Entropy is a behavioral diagnostic of weight distribution, distinct from instance-level predictive uncertainty $U_i$.
