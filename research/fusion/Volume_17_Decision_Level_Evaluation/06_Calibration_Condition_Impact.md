# Chapter 06: Calibration Condition Impact

## 1. Modality Probability Calibration Protocol
Dimension E evaluates the effect of constituent model calibration on downstream fusion decisions. Constituent neural network models often produce overconfident or uncalibrated probability distributions.

Phase C11.11 established frozen calibration mappings for each constituent modality:
1. **Retina**: Temperature Scaling ($T = 1.6218$) on 5-class diabetic retinopathy probabilities.
2. **Foot**: Vector Scaling ($W = [1.0410, 1.0430, 0.8711, 1.1301], b = [0.0292, 0.0625, 0.0630, -0.1547]$) on 4-class Wagner scores.
3. **Clinical**: Platt Logit Scaling ($\text{slope} = 0.983834, \text{intercept} = -0.002721$) on binary readmission predictions.

The evaluation compares:
- **Calibrated Condition (Reference B6)**: Modalities evaluated using their calibrated probability outputs.
- **Uncalibrated Condition**: Modalities evaluated using raw, uncalibrated logit/probability outputs.

## 2. Quantitative Comparative Results ($N=500$)
Evaluating ACARA-U across the $N=500$ cohort (seed 115) under both conditions yields:

| Metric / Parameter | Uncalibrated B6 | Calibrated B6 (Reference) | Paired Shift $\Delta$ | 95% Bootstrap CI | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Risk $\overline{R}_{\text{fusion}}$** | 0.285285 | 0.289900 | +0.004615 | [+0.003544, +0.005752] | YES ($0 \notin \text{CI}$) |
| **Mean DCRI $\overline{\text{DCRI}}$** | 0.221891 | 0.226506 | +0.004615 | [+0.003544, +0.005752] | YES ($0 \notin \text{CI}$) |
| **Mean Penalty $\overline{P}_U$** | 0.063394 | 0.063394 | 0.000000 | [0.000000, 0.000000] | NO (Identical) |
| **Routine (Tier 0)** | 241 (48.2%) | 232 (46.4%) | -9 (-1.8%) | N/A | N/A |
| **Assessment (Tier 1)** | 177 (35.4%) | 187 (37.4%) | +10 (+2.0%) | N/A | N/A |
| **Escalation (Tier 2)** | 82 (16.4%) | 81 (16.2%) | -1 (-0.2%) | N/A | N/A |
| **Negative DCRI Count** | 39 (7.8%) | 37 (7.4%) | -2 (-0.4%) | N/A | N/A |

### Invariance of Epistemic Uncertainty Penalties
The mean uncertainty penalty remains identical ($\overline{P}_U = 0.063394$, paired difference $\Delta = 0.000000$). Probability calibration transforms the constituent probability logits via temperature and vector scaling, while the predictive uncertainty estimates $U_i$ produced by the constituent uncertainty quantification heads remain unchanged. Because the uncertainty penalty is solely a function of $U_i$ and the frozen discount parameter $\delta = 0.10$, the change in DCRI matches the change in fused risk exactly ($\Delta(\mathrm{DCRI}) = \Delta(R_{\mathrm{fusion}}) = +0.004615$).

## 3. Transition Matrix Between Calibration Conditions
The $3 \times 3$ transition matrix maps Uncalibrated decisions (rows) to Calibrated decisions (columns) across the paired $N=500$ cohort:

| Uncalibrated Tier (Row) \ Calibrated Tier (Col) | Tier 0 (Routine) | Tier 1 (Assessment) | Tier 2 (Escalation) | Row Total |
| :--- | :---: | :---: | :---: | :---: |
| **Tier 0 (Routine Review)** | 231 | 10 | 0 | 241 |
| **Tier 1 (Additional Assessment)** | 1 | 174 | 2 | 177 |
| **Tier 2 (Escalation)** | 0 | 3 | 79 | 82 |
| **Column Total** | **232** | **187** | **81** | **500** |

### Transition Definitions and Summary:
- **Direction**: Rows represent Uncalibrated baseline assignments; columns represent Calibrated reference assignments.
- **Upgrade**: Movement to a higher-numbered tier under calibration (Tier $0 \to 1$: 10 cases; Tier $1 \to 2$: 2 cases; total = 12 cases, 2.4%).
- **Downgrade**: Movement to a lower-numbered tier under calibration (Tier $1 \to 0$: 1 case; Tier $2 \to 1$: 3 cases; total = 4 cases, 0.8%).
- **Total Invariant / Unchanged**: 484 packets ($96.8\%$).
- **Total Reclassified**: 16 packets ($3.2\%$).

## 4. Key Takeaways and Scope Limitations
1. **Measured Mean Shift**: Probability calibration produces a modest upward risk shift ($\Delta = +0.004615$, 95% CI: $[+0.003544, +0.005752]$), driven primarily by temperature scaling softening overconfident low-risk predictions in the Retina channel.
2. **High Tier Stability**: 96.8% (484/500) of patient decision tiers remain invariant between uncalibrated and calibrated model inputs.
3. **Reclassification Distribution**: Of the 16 reclassified packets, 12 are upgrades (transitioning to a higher decision tier) and 4 are downgrades.
4. **Methodological Scope**: Reclassification demonstrates the sensitivity of fixed decision thresholds ($\tau_1 = 0.20, \tau_2 = 0.40$) to probability calibration. This algorithmic shift does not independently prove that calibrated tier assignments are clinically superior or more diagnostically accurate without validation against external gold-standard patient outcomes.
