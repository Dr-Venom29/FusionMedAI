# Phase C11.18: Discrepancy Reconciliation & Historical Cross-Audit

### 1. Executive Summary of Discrepancies
During the execution of Phase C11.18, a systematic forensic audit was conducted across all prior C11 evaluation artifacts, research volumes, and canonical rerun data. Five apparent numerical discrepancies were identified between historical notes, baseline reports, and canonical rerun artifacts:

| Item | Primary Artifact Value | Comparison Artifact Value | Status | Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **1. Negative DCRI Count** | **37 / 500 (7.40%)** in Frozen C11.17 Baseline (`decision_evaluation`) | **38 / 500 (7.60%)** in Canonical Redesigned Rerun (`canonical_redesigned_rerun`) | **Reconciled** | Root cause identified: Confidence representation update ($C_C = 2\lvert p - 0.5\rvert$ vs raw probability). |
| **2. `css` Column Definition** | Stored `css` $\equiv$ `delta_max` ($\Delta_{\max}$) in `03_packet_level_results.csv` | Theoretical $\Delta_{\mathrm{mean}} = \frac{1}{3}\sum \lvert r_j - r_k\rvert$ | **Reconciled** | Artifact column `css` is maximum pairwise disparity; documented alongside analytical mean $\Delta_{\mathrm{mean}}$. |
| **3. Two-Tier Jumps (B6 vs B1)** | **11 packets** (all Upward: Tier 0 $\to$ Tier 2) | Historical memo mentioned 14 packets | **Reconciled** | Exact packet-level audit proves 11 upward jumps and 0 downward jumps in frozen $N=500$ cohort. |
| **4. Multi-Channel 2-Tier Jumps** | **0 packets** across B6 vs B2, B3, B4, B5 | Theoretical risk of multi-channel jumps | **Verified** | No two-tier jumps observed between B6 and B2–B5 in the evaluated 500-packet cohort. |
| **5. B6 vs B5 Reclassification** | **3 packets** reclassified (all Upward: Tier 0 $\to$ Tier 1) | Expected symmetric drift | **Reconciled** | Enabling quality term ($\eta=0.5$) shifts 3 borderline packets across $\tau_1=0.20$. |

> [!NOTE]
> **Scope Note on Evaluation Cohorts**: The 37-packet result belongs strictly to the frozen C11.17 baseline ($N=500$, seed $115$, raw probability confidence). The 38-packet result belongs strictly to the canonical redesigned rerun ($N=500$, seed $115$, boundary-referenced confidence $C_C = 2|p - 0.5|$). These results are reported separately because the confidence formulation differs; they must not be combined or substituted for one another.

---

## 2. Forensic Reconciliation 1: Negative DCRI Count (37 vs 38 Packets)

### 2.1 The Apparent Conflict
- In `experiments/fusion/decision_evaluation/03_packet_level_results.csv` (and `04_summary_metrics.json`), Policy B6 produces exactly **37 negative DCRI packets** (prevalence $7.40\%$).
- In `experiments/fusion/canonical_redesigned_rerun/04_canonical_c11_full_metrics.json`, under `canonical`, Policy B6 reports **38 negative DCRI packets** (prevalence $7.60\%$).

### 2.2 Mathematical Mechanism & Root Cause
The 1-packet difference is not a calculation error or non-determinism. It is the direct mathematical consequence of the formal redesign in the clinical confidence representation:
1. **Frozen Baseline Cohort (`legacy`)**: Modality predictive confidence entered the router as raw probabilities $C_i = \hat{p}_i$. Under this weighting, the mean weights were:
   $$\bar{w}_{\mathrm{retina}} = 0.5010, \quad \bar{w}_{\mathrm{foot}} = 0.2609, \quad \bar{w}_{\mathrm{clinical}} = 0.2381$$
2. **Canonical Redesigned Cohort (`canonical`)**: Clinical confidence was updated to the decision-boundary referenced confidence metric:
   $$C_{C, \mathrm{clinical}} = 2 \cdot \lvert \hat{p}_{\mathrm{clinical}} - 0.50 \rvert$$
   This calibration increased clinical channel authority, shifting mean weights to:
   $$\bar{w}_{\mathrm{retina}} = 0.4271, \quad \bar{w}_{\mathrm{foot}} = 0.2231, \quad \bar{w}_{\mathrm{clinical}} = 0.3498$$
3. **The Pivotal Encounter (`PACKET_0068`)**:
   - In the legacy formulation: $R_{\mathrm{fusion}} = 0.0812$, $P_U = 0.10 \cdot 0.8088 = 0.0809$, yielding $\text{DCRI} = +0.00030$ (positive, Tier 0).
   - In the canonical redesigned formulation: the increased weight assigned to the clinical modality (which predicted near-zero risk $r_C = 0.012$) lowered $R_{\mathrm{fusion}}$ to $0.0799$. With unchanged uncertainty penalty $P_U = 0.0809$, $\text{DCRI} = -0.00098$ (negative, Tier 0).

### 2.3 Interpretation and Scope
All negative-DCRI packets in both reported cohorts map strictly to Tier 0 (Routine Review) under the specified threshold rule ($\text{DCRI} < \tau_1 = 0.20$). Negative DCRI is an expected mathematical property of subtracting cumulative uncertainty from a low fused risk baseline ($P_U > R_{\mathrm{fusion}}$). This confirms the mechanical behavior of the decision mapping for these packets, but does not determine whether the resulting tier is clinically correct or whether clinically important cases were missed. The 37-packet count is certified for the frozen C11.17 baseline artifact; the 38-packet count is certified for the canonical redesigned rerun pipeline.

---

## 3. Forensic Reconciliation 2: Conflict Metric Definition (`css` vs `delta_max`)

### 3.1 The Apparent Conflict
In `experiments/fusion/decision_evaluation/03_packet_level_results.csv`, two columns measure cross-modality conflict: `css` and `delta_max`. Mathematical verification reveals:
$$\max_{p \in \text{Cohort}} \lvert \text{css}(p) - \text{delta\_max}(p) \rvert = 0.0000000000000000$$
The column `css` is bitwise identical to `delta_max`.

### 3.2 Theoretical Distinction
In literature and earlier project volumes:
- **Maximum Pairwise Disparity ($X_{\max}$)**:
  $$X_{\max} = \max(\lvert r_R - r_F\rvert, \lvert r_R - r_C\rvert, \lvert r_F - r_C\rvert)$$
- **Average Pairwise Disparity ($X_{\mathrm{mean}}$)**:
  $$X_{\mathrm{mean}} = \frac{1}{3} \left( \lvert r_R - r_F\rvert + \lvert r_R - r_C\rvert + \lvert r_F - r_C\rvert \right)$$

### 3.3 Quantitative Divergence
Comparing stored `delta_max` to newly calculated $X_{\mathrm{mean}}$ across the 500 packets:
- Maximum discrepancy: $\max |X_{\max} - X_{\mathrm{mean}}| = 0.330633$
- Mean discrepancy: $\text{mean} |X_{\max} - X_{\mathrm{mean}}| = 0.158302$
- Spearman rank correlation: $\rho = 0.9412$ ($p < 10^{-100}$)

### 3.4 Official Resolution
In all Phase C11.18 artifacts and Volume 18 research documentation, both metrics are explicitly computed, stored, and labeled:
- `delta_max`: Extreme worst-case channel divergence.
- `delta_mean`: Global multi-channel divergence.
This resolves the column conflation while preserving bitwise fidelity with C11.17.

---

## 4. Forensic Reconciliation 3: Two-Tier Jumps and Baselines

### 4.1 Unimodal Baseline (B1) Flip-Flops
In early exploratory memos, the number of two-tier jumps for B1 was cited as "14 packets". The full deterministic audit of the frozen C11.17 transition matrix (`05_decision_transitions.csv`) reveals:
- **Upward Two-Tier Jumps (B1 Tier 0 $\to$ B6 Tier 2)**: Exactly **11 packets** ($2.20\%$).
- **Downward Two-Tier Jumps (B1 Tier 2 $\to$ B6 Tier 0)**: Exactly **0 packets** ($0.00\%$).
- **One-Tier Upward (B1 Tier 0 $\to$ B6 Tier 1 or B1 Tier 1 $\to$ B6 Tier 2)**: Exactly **67 packets** ($13.40\%$).
- **One-Tier Downward (B1 Tier 2 $\to$ B6 Tier 1 or B1 Tier 1 $\to$ B6 Tier 0)**: Exactly **106 packets** ($21.20\%$).
- **Total Reclassifications (B6 vs B1)**: Exactly **184 packets** ($36.80\%$).

### 4.2 Multi-Channel Baselines (B2–B5) Stability
When comparing B6 against all multi-channel fusion policies:
- **B6 vs B2 (Uniform Average)**: 81 reclassifications (43 Up, 38 Down), **0 two-tier jumps**.
- **B6 vs B3 (Confidence Fusion)**: 57 reclassifications (9 Up, 48 Down), **0 two-tier jumps**.
- **B6 vs B4 (Confidence + Reliability)**: 64 reclassifications (8 Up, 56 Down), **0 two-tier jumps**.
- **B6 vs B5 (ACARA-U without Quality)**: 3 reclassifications (3 Up, 0 Down), **0 two-tier jumps**.

**Policy Observation**: The 11 two-tier jumps observed between B6 and B1 reflect the sensitivity of unimodal selection to single-sensor values. In the evaluated 500-packet cohort, no two-tier jumps were observed between B6 and any multi-channel baseline (B2–B5).

---

## 5. Forensic Reconciliation 4: Asymmetric Shift Between B6 and B5

The comparison between B6 (full ACARA-U with quality term $\eta = 0.5$) and B5 (ablation without quality $\eta = 0.0$) yields exactly 3 reclassified packets out of 500 ($0.60\%$):
- `PACKET_0124`: B5 Tier 0 $\to$ B6 Tier 1
- `PACKET_0289`: B5 Tier 0 $\to$ B6 Tier 1
- `PACKET_0411`: B5 Tier 0 $\to$ B6 Tier 1

**Root Cause**: In all three cases, the dominant modality possessing elevated disease risk also exhibited superior image quality ($Q_i \ge 0.95$), while the discordantly low-risk modality had slightly lower quality ($Q_j = 0.85$). Enabling the quality term $\eta = 0.5$ increased the weight of the higher-quality channel sufficiently to shift DCRI across the threshold cutoff $\tau_1 = 0.20$ ($0.1985 \to 0.2014$), advancing the simulated packet into Tier 1. This mechanism accounts for the directional shift observed in the artifacts; clinical correctness was not established by this transition analysis alone.
