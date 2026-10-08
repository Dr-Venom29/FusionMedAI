# Phase C11.11: Modality Calibration Protocol & Pre-specified Hypotheses

## 1. Scientific Objective

Phase C11.11 investigates whether incorporating calibrated modality-level probabilities improves or alters the behavior of the ACARA-U decision-level fusion system compared with uncalibrated probabilities, under the controlled decision-level benchmark.

The protocol tests calibration as an upstream component of the fusion pipeline, avoiding any assumption or claim that $R_{\text{fusion}}$ represents a clinically calibrated probability.

---

## 2. Strict Methodological Boundaries

1. **Unpaired Cohort Reality**: The constituent datasets (Retina, Foot, Clinical) originate from independent clinical cohorts. No joint multimodal ground truth exists.
2. **Prohibition of Fabricated Multimodal Ground Truth**: The protocol explicitly forbids fabricating synthetic patient-level disease outcomes to compute fusion AUROC, AUPRC, Brier score, or ECE.
3. **Derived Decision Index Classification**: Fused risk ($R_{\text{fusion}}$) and $\text{DCRI}$ are strictly treated as derived composite decision indices.
4. **Legitimate Modality-Level Calibration**: Calibration quality metrics (ECE, Brier, NLL, slopes) are computed strictly on each constituent modality's independent single-task validation/test splits.

---

## 3. Pre-specified Research Questions & Hypotheses

| ID | Research Question | Pre-specified Hypothesis | Interpretation Rule |
| :--- | :--- | :--- | :--- |
| **RQ1** | Do frozen calibration layers reduce single-modality calibration error? | **H1**: Frozen calibration transforms reduce validation ECE across all 3 constituent modalities without parameter retraining. | Directional reduction ($\text{ECE}_{\text{cal}} < \text{ECE}_{\text{raw}}$) across all channels. |
| **RQ2** | Does modality calibration alter scalar risk projections? | **H2**: Modality-level calibration alters continuous risk projections $r_i$ and composite fused risk $R_{\text{fusion}}$. | $95\%$ bootstrap CI for $\Delta R_{\text{fusion}}$ strictly excludes zero. |
| **RQ3** | Does calibration alter dynamic routing authority? | **H3**: Propagating calibrated probabilities into the ACARA-U router alters routing weights $w_i$. | $95\%$ bootstrap CI for $\Delta w_i$ strictly excludes zero. |
| **RQ4** | Are fundamental router invariants conserved under calibration? | **H4**: Weight simplex $\sum_{i \in \mathcal{A}} w_i = 1.0$ and non-negativity $w_i \ge 0$ strictly hold across all conditions. | Exact conservation within numerical precision ($10^{-6}$). |
| **RQ5** | Do calibration effects remain bounded? | **H5**: Calibration produces bounded perturbations that satisfy pre-specified behavioral limits without pathological entropy collapse or conflict explosion. | $|\Delta H(w)| < 0.20$, $|\Delta \text{DCRI}| < 0.10$, and $\Delta \text{Conflict} < 0.05$. |
| **RQ6** | Do calibration effects persist under input degradation? | **H6**: Under the three representative degradation operators evaluated in C11.11 (Retina Blur, Foot Blur, Clinical Masking), the calibration-related authority shift persisted across $D0 \to D3$. | Monotonic authority decay preserved under calibrated inputs. |

---

## 4. Frozen Experimental Parameters

All experimental parameters are frozen from previous phases without retuning:

- **Cohort Size**: $N = 500$ Controlled Decision Packets
- **Random Seed**: $115$ (process-independent SHA-256 derivation)
- **Router Coefficients**: $\alpha = 1.0, \beta = 1.5, \gamma = 1.0, \eta = 0.5$
- **Uncertainty Penalty**: $\delta = 0.20$
- **Retina Calibrator**: Temperature Scaling ($T = 1.6218$, from Phase C5 v004 on validation split)
- **Foot Calibrator**: Vector Scaling ($W = [1.0410, 1.0430, 0.8711, 1.1301], b = [0.0292, 0.0625, 0.0630, -0.1547]$, from Phase C6 on validation split)
- **Clinical Calibrator**: Frozen Platt / Logit Scaling ($a = 0.983834, b = -0.002721$, from Phase C7 on validation split)

