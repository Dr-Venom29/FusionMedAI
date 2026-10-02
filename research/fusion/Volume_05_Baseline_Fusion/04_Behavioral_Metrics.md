# Behavioral & Comparative Evaluation Metrics

## Mathematical Definitions of Evaluation Metrics

To characterize the decision dynamics of baselines without relying on ground-truth clinical labels, Phase C11.5 defines four formal behavioral metric families.

---

### 1. Modality Weight Allocation & Marginal Statistics

For each baseline $b \in \mathcal{B}$ across cohort $\mathcal{P} = \{ p_k \}_{k=1}^N$:

$$\mu(w_i) = \frac{1}{N} \sum_{k=1}^N w_{i}^{(k)}, \quad \sigma(w_i) = \sqrt{\frac{1}{N-1} \sum_{k=1}^N \left( w_{i}^{(k)} - \mu(w_i) \right)^2}$$

Measures average authority assigned to each diagnostic channel under natural variation.

---

### 2. Routing Entropy ($H(w)$)

Measures the dispersion or concentration of authority across the active modality set $\mathcal{A}$:

$$H(w) = - \sum_{i \in \mathcal{A}} w_i \ln(w_i)$$

- For $|\mathcal{A}| = 3$: Maximum entropy is $\ln(3) \approx 1.098612$ (achieved uniquely by B2 Uniform Average).
- Unimodal selection (B1) yields $H(w) = 0.000000$.
- Dynamic routers (B3–B6) produce entropy values that vary with the routing signals supplied to the model.

---

### 3. Dominant Modality Frequency ($\text{Rate}_i$)

Quantifies how often each modality receives the plurality of decision authority:

$$\text{Dominant}(p_k) = \arg\max_{i \in \mathcal{A}} w_{i}^{(k)}$$

$$\text{Rate}_i = \frac{1}{N} \sum_{k=1}^N \mathbb{I}(\text{Dominant}(p_k) = i)$$

---

### 4. Cross-Modality Risk Disagreement ($X_{ij}, X_{\max}$)

Quantifies pairwise and maximal absolute risk divergence across diagnostic channels:

$$\begin{aligned}
X_{RF} &= |r_R - r_F| \\
X_{RC} &= |r_R - r_C| \\
X_{FC} &= |r_F - r_C| \\
X_{\max} &= \max(X_{RF}, X_{RC}, X_{FC})
\end{aligned}$$

Tracks the degree of cross-channel conflict and establishes the input space for downstream conflict resolution (DCRI).
