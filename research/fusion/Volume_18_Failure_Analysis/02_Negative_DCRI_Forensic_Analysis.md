# Chapter 02: Negative DCRI Forensic Analysis

## 1. Mathematical Formulation & Root Cause
The Diabetic Complication Risk Index (DCRI) is formulated as:
$$\text{DCRI}_{0.10} = R_{\mathrm{fusion}} - 0.10 \sum_{i \in \mathcal{A}} U_i$$
Where $R_{\mathrm{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i$ is the simplex-weighted fused risk ($\\sum w_i = 1, w_i \ge 0$), and $\\sum_{i \in \mathcal{A}} U_i$ is the cumulative epistemic uncertainty across active channels.

A negative DCRI occurs if and only if:
$$P_U = 0.10 \sum_{i \in \mathcal{A}} U_i > R_{\mathrm{fusion}}$$
This condition arises in clinical encounters exhibiting:
1. Low underlying multi-channel risk predictions ($R_{\mathrm{fusion}} < 0.10$).
2. High cumulative model uncertainty ($U_{\mathrm{sum}} > 0.50$, yielding penalty $P_U > 0.05$).

## 2. Quantitative Distribution in Frozen Evaluation Cohort ($N = 500$)
- **Prevalence under B6**: Exactly **37 / 500 packets (7.40%)**.
- **Minimum Observed DCRI**: $-0.027143$.
- **Mean Negative DCRI**: $-0.010750 \pm 0.007612$.
- **Maximum Negative DCRI**: $-0.000095$.
- **Mean $R_{\mathrm{fusion}}$ for Negative Packets**: $0.068849 \pm 0.017842$.
- **Mean $U_{\mathrm{sum}}$ for Negative Packets**: $0.795996 \pm 0.114521$ (yielding mean penalty $0.079600$).

## 3. Comparison Across Fusion Policies
| Policy | Negative DCRI Count | % of Cohort | Min DCRI | Max DCRI | Mean DCRI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B1 (Unimodal WTA)** | 223 | 44.60% | -0.0916 | 0.8874 | 0.1916 |
| **B2 (Uniform Avg)** | 0 | 0.00% | 0.0034 | 0.7712 | 0.2334 |
| **B3 (Confidence)** | 1 | 0.20% | -0.0028 | 0.8123 | 0.2508 |
| **B4 (Conf + Rel)** | 1 | 0.20% | -0.0031 | 0.8245 | 0.2551 |
| **B5 (Conf + Rel - Unc)** | 35 | 7.00% | -0.0268 | 0.7981 | 0.2239 |
| **B6 (Full ACARA-U)** | 37 | 7.40% | -0.0271 | 0.8012 | 0.2265 |

### 3.1 Formula-Policy Interaction under Unimodal B1
Under the specified DCRI calculation, B1 exhibits 223 / 500 negative scores (44.60%), whereas Full ACARA-U (B6) exhibits 37 in the frozen baseline (7.40%). This difference is driven by a structural formula-policy interaction:
- B1 assigns 100% weight to the single highest-reliability modality (Retina, $w_R = 1.0$), while the linear uncertainty penalty subtracts cumulative epistemic uncertainty across all three available channels ($U_{\mathrm{sum}} = U_R + U_F + U_C$).
- When Retina predicts low complication risk ($r_R < 0.10$) while secondary modalities (Foot Ulcer or Clinical EHR) exhibit substantial uncertainty ($U_F + U_C > 0.80$), B1's single-channel risk is heavily offset by secondary uncertainty that did not contribute to its risk score.
- In contrast, ACARA-U dynamically distributes authority weights across active modalities based on confidence, reliability, uncertainty, and quality.
- These descriptive counts reflect the mathematical interaction between authority weights and the multi-channel penalty scope; they describe policy mechanics rather than establishing clinical superiority.

## 4. Decision Tier Disposition & Scope of Invariants
- **Mechanical Consequence of Threshold Rule**: Under the evaluated threshold rule, all 37 packets with negative DCRI in the frozen B6 baseline (and all 38 packets in the canonical rerun) were assigned to Tier 0 (Routine Review) because $\text{DCRI} < 0.00 < \tau_1 = 0.20$.
- **Absence of Clinical Correctness Guarantee**: This assignment is a direct mechanical consequence of the score and threshold definitions. While it demonstrates that negative DCRI does not trigger upward tier escalation under these rules, it does not establish whether these Tier 0 assignments were clinically correct or whether clinically important cases were missed.
- **Auditability**: Unclamped scores accurately indicate that cumulative epistemic uncertainty exceeds base risk signal amplitude, maintaining transparent traceability in research repositories.

## 5. Candidate Presentation Options (Requires Downstream Clinical Validation)
A display rectification such as:
$$\text{DCRI}_{\mathrm{display}} = \max(0.0, \text{DCRI})$$
has been discussed as a candidate presentation option for clinician-facing user interfaces to avoid displaying negative risk percentages. However:
1. Clamping conceals the magnitude of the negative excursion, masking cases where cumulative uncertainty substantially exceeds base risk.
2. Clamping does not resolve the underlying trade-off of high uncertainty lowering triage priority.
3. Raw unclamped scores must be retained in backend audit trails, and separate presentation of uncertainty or ambiguity flags is required.
This is a proposed presentation option subject to downstream clinical validation, not an operational recommendation established by Phase C11.18.
