# Phase C11.6: DCRI Risk Aggregation Protocol

## 1. Scientific Objectives

Phase C11.6 evaluates how adaptive modality authority weights allocated by ACARA-U translate into aggregated risk projections and uncertainty-discounted decision metrics.

```text
Phase C11.4: Dynamic Router Architecture   ───► Allocates weights w_i
Phase C11.5: Comparative Baseline Ladder  ───► Evaluates weight allocation behavior
Phase C11.6: DCRI Risk Aggregation        ───► Converts (w_i, r_i, U_i) to R_fusion & DCRI
```

### Primary Research Questions
1. **Aggregation Stability**: Does convex weighted combination $R_{\text{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i$ produce a bounded and interpretable scalar index across all 7 availability regimes?
2. **Uncertainty Sensitivity**: How sensitive is the derived index $\text{DCRI}_\delta$ to the penalty scaling coefficient $\delta \in \{0.0, 0.05, 0.1, 0.2, 0.5, 1.0\}$?
3. **Double-Use Mechanics**: How does the explicit DCRI uncertainty discount $P_U = \delta \sum U_i$ interact with the routing-level uncertainty penalty in the ACARA-U logit kernel?

---

## 2. Input-Output Contract Specification

### Input Data
Every controlled decision packet contains:
$$\{r_i, C_i, U_i, Q_i, R_i, A_i\} \quad \forall i \in \{\text{retina}, \text{foot}, \text{clinical}\}$$
together with ACARA-U router outputs:
$$\{w_i, z_i, H(w)\}$$

### Output Contract (`DCRIResult`)
```text
DCRIResult:
  packet_id                          : str
  r_fusion                           : float in [0.0, 1.0]
  u_sum                              : float in [0.0, 3.0]
  u_mean                             : float in [0.0, 1.0]
  uncertainty_penalty                : float >= 0.0
  dcri                               : float in [-delta * |A|, 1.0]
  delta                              : float >= 0.0
  modality_weights                   : Dict[str, float] (sum_{i in A} w_i = 1.0)
  modality_risks                     : Dict[str, float] (r_i in [0.0, 1.0])
  weighted_risk_contributions        : Dict[str, float] (K_i = w_i * r_i)
  modality_uncertainties             : Dict[str, float] (U_i in [0.0, 1.0])
  uncertainty_penalty_contributions  : Dict[str, float] (p_i = delta * U_i)
  active_modalities                  : List[str]
  num_active                         : int (0 to 3)
  status                             : 'SUCCESS' or 'NO_MODALITY_AVAILABLE'
```

---

## 3. Strict Methodological Boundaries

To ensure scientific integrity and prevent protocol drift, Phase C11.6 strictly adheres to the following rules:

1. **Controlled Decision Packets**: Predictions are combined as synthetic decision-level instances from held-out validation records; no claims are made regarding unified patient cohorts.
2. **No Neural Training or Learned Weights**: DCRI aggregation is a deterministic mathematical operator; no model is trained on combined indices.
3. **No Unified Clinical Outcome Claims**: The three underlying models predict distinct endpoints (Retina = DR stage, Foot = DFU infection/ischemia, Clinical = 30-day readmission). DCRI is an operational decision support index, not a biological joint probability.
4. **No Metric Misuse**: Discriminative classification metrics (ROC-AUC, PR-AUC, Brier score, ECE) must not be applied to DCRI.
5. **No Premature Delta Selection**: Phase C11.6 documents sensitivity across a 6-point grid; optimal $\delta$ selection belongs exclusively to Phase C11.12.
6. **No Clamping of DCRI**: Negative DCRI values are mathematically valid under the existing formula and must be reported without truncation.
