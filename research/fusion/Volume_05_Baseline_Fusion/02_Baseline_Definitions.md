# Baseline Formalization: B1 to B6 Ladder

## Mathematical Formulations of the Baseline Ladder

Phase C11.5 defines an incremental 6-tier baseline ladder to isolate the exact empirical value added by each component of the ACARA-U dynamic routing architecture:

$$\mathcal{M} = \{ \text{Retina } (R), \text{ Foot } (F), \text{ Clinical } (C) \}$$

Let $\mathcal{A} \subseteq \mathcal{M}$ denote the set of currently available modalities ($\mathcal{A} = \{ i \in \mathcal{M} \mid A_i = 1 \}$).

---

### Baseline B1: Reliability-Selected Unimodal
Selects the single most historically reliable modality among available channels:

$$i^* = \arg\max_{i \in \mathcal{A}} R_i$$

$$w_i = \begin{cases} 1.0 & \text{if } i = i^* \\ 0.0 & \text{otherwise} \end{cases}$$

$$R_{\text{fusion}} = r_{i^*}, \quad H(w) = 0.0$$

*Behavioral Interpretation*: Selects the modality with the highest frozen historical reliability and is therefore insensitive to instance-level confidence, uncertainty, and quality.

---

### Baseline B2: Uniform Average
Assigns equal authority across all currently available modalities:

$$w_i = \frac{1}{|\mathcal{A}|} \quad \forall i \in \mathcal{A}$$

$$R_{\text{fusion}} = \frac{1}{|\mathcal{A}|} \sum_{i \in \mathcal{A}} r_i, \quad H(w) = \ln |\mathcal{A}|$$

*Behavioral Interpretation*: Unweighted uniform baseline; treats all active sub-networks equally regardless of individual certainty, predictive dispersion, or quality signals.

---

### Baseline B3: Confidence-Only Fusion
Weights modalities proportionally to their immediate prediction certainty $C_i \in [0.5, 1.0]$:

$$z_i = 1.0 \times C_i$$

$$\tilde{z}_i = \begin{cases} z_i & \text{if } i \in \mathcal{A} \\ -\infty & \text{if } i \notin \mathcal{A} \end{cases}, \quad w_i = \frac{\exp(\tilde{z}_i - \max_{j \in \mathcal{A}} \tilde{z}_j)}{\sum_{k \in \mathcal{A}} \exp(\tilde{z}_k - \max_{j \in \mathcal{A}} \tilde{z}_j)}$$

*Behavioral Interpretation*: Dynamic weighting based entirely on individual model certainty, blind to global performance priors, uncertainty, and input quality.

---

### Baseline B4: Confidence + Global Reliability
Combines instance-level certainty with frozen historical validation priors:

$$z_i = 1.0 \times C_i + 1.0 \times R_i$$

$$w_i = \text{Softmax}_{\mathcal{A}}(\tilde{z}_i)$$

*Behavioral Interpretation*: Balances current model certainty against frozen historical validation track record.

---

### Baseline B5: Confidence + Reliability + Uncertainty Penalty (Ablation)
Integrates predictive uncertainty ($U_i$) into the routing logit:

$$z_i = 1.0 \times C_i + 1.0 \times R_i - 1.0 \times U_i$$

$$w_i = \text{Softmax}_{\mathcal{A}}(\tilde{z}_i)$$

*Behavioral Interpretation*: Actively penalizes modalities exhibiting high predictive uncertainty/dispersion, using modality-specific uncertainty measures defined in the frozen modality contracts (MC Dropout variance for Retina, predictive entropy for Foot, bootstrap standard deviation for Clinical).

---

### Baseline B6: Full ACARA-U Architecture
Full ACARA-U router incorporating input quality ($Q_i$) alongside confidence, reliability, and uncertainty:

$$z_i = 1.0 \times C_i + 1.0 \times R_i - 1.0 \times U_i + 1.0 \times Q_i$$

$$w_i = \text{Softmax}_{\mathcal{A}}(\tilde{z}_i)$$

*Behavioral Interpretation*: Complete dynamic routing engine incorporating historical priors, instance certainty, predictive dispersion, and input quality into decision authority allocation.
