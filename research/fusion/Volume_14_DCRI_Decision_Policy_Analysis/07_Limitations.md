# Chapter 07 — Methodological Boundaries & Limitations

## 1. Controlled Decision-Packet Benchmark Boundary

1. **Synthetic Multi-Source Packets**: The primary evaluation cohort consists of $N=500$ controlled decision packets (`seed=115`) assembled from independent retrospective single-modality datasets (APTOS 2019 retina, ADPM V3.3 foot ulcer, UCI 130-Hospitals clinical EHR). It does not represent a genuinely paired, multi-center prospective patient population.
2. **Operational Workflow Placeholders**: Action tiers (Routine Review, Additional Assessment, Escalation for Human Review) and operating thresholds ($\tau_1=0.20, \tau_2=0.40$) are standardized mathematical constructs designed to characterize sensitivity, not clinically validated diagnostic thresholds.
3. **No Clinical Claims**: No claims of clinical efficacy, diagnostic safety, disease-risk calibration, patient-level outcome improvements, or clinical net benefit follow from this analysis.

---

## 2. Inadmissibility of Fabricated Decision-Curve Analysis

1. **Requirement for True Target Outcomes**: Legitimate clinical Decision-Curve Analysis (DCA) and net-benefit calculation require:
   - A verified clinical gold-standard outcome label (e.g., biopsy confirmation, 30-day mortality, confirmed ulcer progression).
   - An interpretable decision threshold reflecting clinical relative harm/benefit ratios.
2. **Boundary Enforcement**: Because the controlled decision packets do not possess paired multimodal ground-truth outcomes, calculating artificial "net benefit" curves would constitute methodological fabrication. Consequently, Phase C11.14 strictly restricts reporting to observable policy assignments, transition matrices, and reclassification rates.

---

## 3. Scope of Parameter Lock ($\delta^* = 0.10$)

- **Internal Benchmark Stability Only**: The parameter $\delta^* = 0.10$ was frozen in Phase C11.13 based on multi-tier feasibility, negative rate containment, and rank stability within the controlled benchmark framework.
- **Not a Universal Clinical Optimum**: It does not represent a universally optimal clinical penalty parameter for real-world healthcare deployments. Future clinical translation requires validation on prospective, genuinely paired multimodal patient cohorts.
