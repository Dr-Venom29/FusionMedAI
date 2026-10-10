# Chapter 01: Protocol & Acceptance Criteria

## 1. Executive Summary & Objective

Phase C11.15 investigates the residual mathematical sanity, single-input response directionality, and invariant stability of the **ACARA-U v2 Dynamic Multimodal Router** under the frozen reference configuration:
$$\Theta_0 = (\alpha=1.0, \, \beta=1.5, \, \gamma=1.0, \, \eta=0.5)$$
and the frozen decision-level uncertainty discount:
$$\delta^* = 0.10$$

The primary research objective is to rigorously verify whether the router engine obeys its foundational mathematical invariants and expected directional responses under strictly isolated input variations, without retuning hyperparameter coefficients or altering previously sealed artifacts.

---

## 2. Research Hypotheses

| Hypothesis | Description | Mathematical Expectation | Evaluation Criteria |
| :--- | :--- | :--- | :--- |
| **H1 (Confidence Monotonicity)** | Increasing confidence $C_i$ with other attributes fixed increases $w_i$ in multi-channel regimes. | $\frac{\partial w_i}{\partial C_i} > 0$ for $|\mathcal{A}| \ge 2$; $w_i = 1.0$ for $|\mathcal{A}|=1$. | $\Delta w_i > 10^{-7}$ for $|\mathcal{A}| \ge 2$. |
| **H2 (Reliability Monotonicity)** | Increasing historical reliability $R_i$ in the logit kernel fixture increases $w_i$ in multi-channel regimes. | $\frac{\partial w_i}{\partial R_i} > 0$ for $|\mathcal{A}| \ge 2$; $w_i = 1.0$ for $|\mathcal{A}|=1$. | $\Delta w_i > 10^{-7}$ for $|\mathcal{A}| \ge 2$. |
| **H3 (Uncertainty Monotonicity)** | Increasing predictive uncertainty $U_i$ with other attributes fixed decreases $w_i$ in multi-channel regimes. | $\frac{\partial w_i}{\partial U_i} < 0$ for $|\mathcal{A}| \ge 2$; $w_i = 1.0$ for $|\mathcal{A}|=1$. | $\Delta w_i < -10^{-7}$ for $|\mathcal{A}| \ge 2$. |
| **H4 (Quality Monotonicity)** | Increasing signal quality $Q_i$ with other attributes fixed increases $w_i$ in multi-channel regimes. | $\frac{\partial w_i}{\partial Q_i} > 0$ for $|\mathcal{A}| \ge 2$; $w_i = 1.0$ for $|\mathcal{A}|=1$. | $\Delta w_i > 10^{-7}$ for $|\mathcal{A}| \ge 2$. |
| **H5 (Invariant Conservation)** | The router maintains simplex conservation, hard masking, shift invariance, and numerical stability. | $\sum_{i \in \mathcal{A}} w_i = 1.0 \pm 10^{-10}$, $A_j=0 \implies w_j=0.0$, $\text{softmax}(z+c)=\text{softmax}(z)$. | Zero tolerance violations. |

---

## 3. Acceptance Criteria Matrix (S15-00 through S15-15)

| ID | Criterion Name | Target Condition | Verification Method |
| :--- | :--- | :--- | :--- |
| **S15-00** | Cryptographic Hash Manifest Integrity & Cross-Artifact Consistency | All artifact SHA-256 hashes match sealed freeze manifest (3/3) and 15/15 scorecard criteria are directly reconciled with test results. | Dynamic SHA-256 verification and cross-artifact consistency reconciliation |
| **S15-01** | Frozen Reference Configuration Integrity | Parameters locked to $\Theta_0 = (1.0, 1.5, 1.0, 0.5)$, $\delta^* = 0.10$. | Configuration verification |
| **S15-02** | Confidence Monotonicity | $\Delta C_i > 0 \implies \Delta w_i > 0$ across all multi-modality active sets. | Isolated single-input perturbation sweep |
| **S15-03** | Reliability Monotonicity | $\Delta R_i > 0 \implies \Delta w_i > 0$ in isolated mathematical logit fixture. | Kernel response sweep |
| **S15-04** | Uncertainty Monotonicity | $\Delta U_i > 0 \implies \Delta w_i < 0$ across all multi-modality active sets. | Isolated single-input perturbation sweep |
| **S15-05** | Quality Monotonicity | $\Delta Q_i > 0 \implies \Delta w_i > 0$ across all multi-modality active sets. | Isolated single-input perturbation sweep |
| **S15-06** | Simplex Conservation | $\sum w_i = 1.0 \pm 10^{-10}$ and $w_i \ge 0.0$ for all valid inputs. | Monte Carlo stress testing across regimes |
| **S15-07** | Availability Masking | $A_i = 0 \implies w_i = 0.0$ exact with zero leakage. | Missingness regime evaluation |
| **S15-08** | Softmax Shift Invariance | Softmax weights invariant under logit offsets $z_i \to z_i + c$ ($c \in [-500, 500]$) and production route uniform confidence shift $+0.10$. | Dual logit offset and route shift evaluation |
| **S15-09** | Reference Normalization Kernel Permutation Invariance | Reference normalization kernel evaluates all 6 channel permutations with identical weights (order independence). | Permutation order evaluation on reference kernel |
| **S15-10** | Numerical Stability & Input Validation | Stable finite weights under extreme logits; 19/19 invalid input cases rejected. | Extreme differential and input validation tests |
| **S15-11** | Empty Handling | $A = \emptyset \implies \text{NO\_MODALITY\_AVAILABLE}$ fail-closed status. | Degenerate case evaluation |
| **S15-12** | Masked-Value Corruption Invariance | Inactive channel feature perturbations have zero influence on active weights. | Feature corruption injection |
| **S15-13** | Pipeline Stage Decoupling & Contract | Router weight monotonicity is decoupled from fused risk; production DCRI used. | Production pipeline integration check |
| **S15-14** | Regression Invariance | All earlier sealed baselines (C11.14 suite: 22/22 tests) continue to pass. | Executed regression suite check |
| **S15-15** | Mutation Fault Detection | Verification gates reliably detect 14 distinct injected intentional faults (14/14). | Executed mutation testing suite |

---

## 4. Mathematical Formulation

### 4.1 ACARA-U Routing Kernel
For each modality channel $i \in \{\text{retina}, \text{foot}, \text{clinical}\}$, the raw logit is computed as:
$$z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$$

### 4.2 Hard Availability Masking
The masked logit $\tilde{z}_i$ is defined as:
$$\tilde{z}_i = \begin{cases} z_i & \text{if } A_i = 1 \\ -\infty & \text{if } A_i = 0 \end{cases}$$

### 4.3 Numerically Stable Softmax Weight Allocation
Let $\mathcal{A} = \{i : A_i = 1\}$ denote the set of active modalities. The normalized decision weight $w_i$ is given by:
$$w_i = \begin{cases} \dfrac{\exp(z_i - \max_{j \in \mathcal{A}} z_j)}{\sum_{j \in \mathcal{A}} \exp(z_j - \max_{k \in \mathcal{A}} z_k)} & \text{if } i \in \mathcal{A} \\ 0.0 & \text{if } i \notin \mathcal{A} \end{cases}$$

### 4.4 Partial Derivatives in Multi-Channel Regimes
For any active channel $i \in \mathcal{A}$ with $|\mathcal{A}| \ge 2$:
$$\frac{\partial w_i}{\partial z_i} = w_i (1 - w_i) > 0 \quad (\text{since } 0 < w_i < 1)$$
Applying the chain rule with respect to channel features:
$$\frac{\partial w_i}{\partial C_i} = \alpha w_i (1 - w_i) > 0 \quad (\text{since } \alpha = 1.0 > 0)$$
$$\frac{\partial w_i}{\partial U_i} = -\gamma w_i (1 - w_i) < 0 \quad (\text{since } \gamma = 1.0 > 0)$$
$$\frac{\partial w_i}{\partial Q_i} = \eta w_i (1 - w_i) > 0 \quad (\text{since } \eta = 0.5 > 0)$$
$$\frac{\partial w_i}{\partial R_i} = \beta w_i (1 - w_i) > 0 \quad (\text{since } \beta = 1.5 > 0)$$
In single-modality regimes ($|\mathcal{A}| = 1$), $w_i = \frac{e^0}{e^0} \equiv 1.0$, resulting in zero partial derivatives with respect to all feature variations.
