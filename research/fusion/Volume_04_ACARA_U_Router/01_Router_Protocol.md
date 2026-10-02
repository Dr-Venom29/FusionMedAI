# 01 ACARA-U Router Protocol — Phase C11.4

## 1. Objective & Scope Declaration

Phase C11.4 implements and isolates the **ACARA-U v2 Dynamic Multimodal Router**.

The router ingests modality-level priors and instance-level attributes:
$$\langle C_i, R_i, U_i, Q_i, A_i \rangle \quad \forall i \in \{\text{Retina}, \text{Foot}, \text{Clinical}\}$$
and deterministically computes decision authority weights:
$$\langle w_R, w_F, w_C \rangle$$

---

## 2. Strict Boundary: What Phase C11.4 Does and Does Not Do

### C11.4 DOES:
- Consumes frozen C11.1 `ModalityOutput` contracts.
- Consumes frozen C11.2 `QualityResult` and availability guarantees.
- Ingests frozen C11.3 global validation reliability constants ($R_R=0.929956, R_F=0.922266, R_C=0.825382$).
- Computes modality routing logits $z_i$.
- Enforces hard availability masking ($A_i = 0 \implies \tilde{z}_i = -\infty \implies w_i = 0.0$).
- Applies numerically stable softmax normalization.
- Handles all 7 non-empty operational modality configurations plus the zero-modality edge case.
- Exposes diagnostic metadata (routing entropy, logit breakdown, dominant modality).
- Verifies mathematical monotonicity, numerical stability, and deterministic execution.

### C11.4 DOES NOT (Explicit Exclusions):
- ❌ Does not calculate fused risk scalar ($R_{\text{fusion}} = \sum w_i r_i$).
- ❌ Does not calculate Dynamic Clinical Risk Index ($DCRI = R_{\text{fusion}} - \delta \sum U_i$).
- ❌ Does not perform conflict resolution or inter-modality discordance adjustments.
- ❌ Does not train a neural router or optimize coefficients $\Theta = (\alpha, \beta, \gamma, \eta)$ on test data.
- ❌ Does not synthesize joint clinical patient ground truth labels.
- ❌ Does not claim clinical outcome superiority.

---

## 3. Canonical Uncertainty Normalization Contract

Before routing, the semantic scale of predictive uncertainty $U_i$ is unified across all modalities:
- **Retina**: MC Dropout predictive variance $\sigma_{\text{MC}}^2$ normalized by theoretical maximum multi-class variance scale ($0.16$), clipped to $[0.0, 1.0]$.
- **Foot**: MC Dropout predictive entropy $\mathcal{H}(p)$ normalized by maximum 4-class entropy $\log_2(4) = 2.0$, clipped to $[0.0, 1.0]$.
- **Clinical**: Bootstrap standard deviation $\sigma_{\text{boot}}$ normalized relative to binary dispersion scale ($0.25$), clipped to $[0.0, 1.0]$.

**Unified Routing-Scale Guarantee**:

Across all three modalities, $U_i \in [0.0, 1.0]$ is mapped to a common directional routing scale where lower values indicate lower predictive dispersion and higher values indicate higher predictive dispersion/ambivalence.

These uncertainty measures are **not statistically equivalent physical quantities**: Retina uses MC Dropout predictive variance, Foot uses predictive entropy, and Clinical uses bootstrap predictive standard deviation. The normalization therefore establishes a common directional scale for routing rather than statistical equivalence across modalities.

The term $-\gamma U_i$ penalizes modality authority monotonically.
