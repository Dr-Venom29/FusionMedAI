# Volume 04 — ACARA-U v2 Dynamic Router (Phase C11.4)

## Volume Overview & Scope

This volume formalizes, analyzes, and locks the **ACARA-U v2 Dynamic Multimodal Router** in **FusionMedAI**:

$$z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$$

$$\tilde{z}_i = \begin{cases} z_i & \text{if } A_i = 1 \\ -\infty & \text{if } A_i = 0 \end{cases}$$

$$w_i = \frac{\exp(\tilde{z}_i - \max_{j} \tilde{z}_j)}{\sum_{j=1}^M \exp(\tilde{z}_j - \max_k \tilde{z}_k)}$$

Phase C11.4 implements and isolates the dynamic routing mechanism itself. It receives normalized instance confidence ($C_i$), frozen historical reliability priors ($R_i$), predictive uncertainty ($U_i$), input quality ($Q_i$), and binary availability ($A_i$), allocating decision-level weights ($w_i$) without computing downstream fused risk ($R_{\text{fusion}}$), $DCRI$, or optimizing against clinical targets.

**Status:** C11.4 implementation and verification are complete and sealed (`18/18 gates passed`). Downstream multimodal decision fusion is reserved for Phase C11.5.

---

## Documents in this Volume

| Document | Focus Area | Key Formalizations |
| :--- | :--- | :--- |
| [`01_Router_Protocol.md`](./01_Router_Protocol.md) | Methodological Governance | Router scope boundary, inputs/outputs, frozen priors ingestion, and prohibited operations. |
| [`02_Mathematical_Formulation.md`](./02_Mathematical_Formulation.md) | Mathematical Mechanics | Logit scoring function, hard availability masking ($-\infty$), stable softmax, and routing entropy. |
| [`03_Input_Output_Contract.md`](./03_Input_Output_Contract.md) | Contract Specifications | `ModalityChannelInput`, `RouterInput`, `RouterCoefficients`, and `RouterResult` containers. |
| [`04_Behavioral_Evaluation.md`](./04_Behavioral_Evaluation.md) | Behavioral Verification | Controlled scenarios (A–D), monotonicity sweeps, authority reallocation, and 7 operational configs. |
| [`05_Freeze_Report.md`](./05_Freeze_Report.md) | Freeze Report & Handoff | Locked router configuration, verification manifest (18/18 gates), and handoff to Phase C11.5. |

---

## Summary of Router Invariants & Baselines

| Component | Formal Specification | Invariant / Boundary |
| :--- | :--- | :--- |
| **Logit Kernel** | $z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$ | $\alpha, \beta, \gamma, \eta \in [0.0, 5.0]$ |
| **Availability Mask** | $\tilde{z}_i = z_i$ if $A_i=1$ else $-\infty$ | $A_i = 0 \implies w_i = 0.0$ strictly |
| **Normalized Weights** | $w_i = \text{Softmax}(\tilde{z}_i)$ | $\sum_{i \in \mathcal{A}} w_i = 1.0, \quad 0 \le w_i \le 1$ |
| **Routing Entropy** | $H(w) = -\sum_{i \in \mathcal{A}} w_i \ln(w_i)$ | $0.0 \le H(w) \le \ln(3) \approx 1.0986$ |
| **Graceful Failure** | Config 0 ($\emptyset$ available) | Emits `NO_MODALITY_AVAILABLE` status, zero weights |
