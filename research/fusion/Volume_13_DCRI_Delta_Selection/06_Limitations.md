# Chapter 06 — Methodological Boundaries & Limitations

## 1. Decision Index vs Clinical Diagnostic Scope

The Decision Confidence & Risk Index ($\text{DCRI}_\delta$) is an engineered decision-support index designed to synthesize calibrated risk with epistemic model uncertainty. 

> [!CAUTION]
> **Important Scientific Scope:**
> 1. $\text{DCRI}$ is **NOT** a calibrated posterior probability of disease.
> 2. $\text{DCRI}$ cannot be directly compared against clinical threshold cutoffs established for pure probabilities (e.g. $r \ge 0.50$).
> 3. Negative $\text{DCRI}$ values do **NOT** imply negative risk; they signify encounters where epistemic uncertainty exceeds estimated risk.

---

## 2. Controlled Decision Packet Cohort Constraints

1. **Controlled Multi-Source Benchmark Units**: The evaluation cohort ($N=500$, seed 115) comprises controlled multimodal decision evaluation packets assembled from validated single-modality model predictions (Retina, Diabetic Foot Ulcer, and Tabular Clinical).
2. **Benchmark Nature**: The packets serve as controlled benchmark units for characterizing routing mathematics, availability fail-safes, and uncertainty sensitivity scaling. They do not represent longitudinally paired real-patient encounters or prospective clinical trial records.
3. **Absence of Multimodal Clinical Ground Truth**: No downstream multimodal clinical endpoints were used to select $\delta$. The selection protocol evaluates mathematical and behavioral properties on the specified controlled cohort; it does not establish clinical diagnostic validity, calibration of DCRI as a clinical disease probability, or improved patient-level clinical outcomes.

---

## 3. Internal Pipeline Research Freeze Boundary

Following the formal completion of Phase C11.13:
- The parameter $\delta^* = 0.10$ is locked as an internal research freeze for the specified ACARA-U multimodal fusion pipeline under reference configuration $\Theta_0$.
- No further post-hoc tuning, grid searching, or per-modality adjustments of $\delta$ are permitted in Phase C11.14+.
