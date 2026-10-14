# Volume 18: Diagnostic Failure & Error Analysis
**FusionMedAI Research Record -- Phase C11.18**

---

## Executive Overview
Volume 18 documents Phase C11.18 of the FusionMedAI research program, conducting an exhaustive forensic audit on the frozen Phase C11.17 decision-level evaluation artifacts. It investigates the difficult edge cases, negative scores, extreme cross-modality conflicts, decision tier transitions, and epistemic uncertainty failure patterns exposed under the full ACARA-U dynamic router (Policy B6) and its comparator baselines (B1-B5) across the controlled cohort ($N = 500$, seed $115$).

---

## Key Forensic Findings

```
========================================================================================
                          VOLUME 18 FORENSIC AUDIT SUMMARY
========================================================================================
  Evaluated Cohort:         500 controlled decision packets (Tri-modal RFC, Seed 115)
  Sealed Input Manifest:    SHA-256 verified against C11.17 freeze manifest (4/4 PASS)
  Verification Gates:       10 / 10 GATES PASSED (Reported Software Gates)
----------------------------------------------------------------------------------------
  Negative DCRI (B6):       37 / 500 (7.40%) | Min: -0.027143 | Mean: -0.010750
                            Tier Invariant: 100% (37/37) map strictly to Tier 0 Routine
  Unimodal Negative DCRI:   223 / 500 (44.60%) in B1 (single-channel risk vs multi-channel penalty)
  Extreme Conflict (Top 5%): Delta_max >= 0.9427 (N=25) | 18 Tier 1, 7 Tier 2, 0 Tier 0 in B6
                            Zero contradictory cases mapped to Routine Review under evaluated B6
  Two-Tier Decision Jumps:  B6 vs B1: 11 Upward (+2) | 0 Downward (-2)
                            B6 vs Multi-Channel (B2-B5): 0 Two-Tier Jumps (100% Eliminated)
  Boundary Proximity:       59 / 500 (11.8%) within +/- 0.02 of thresholds (35 tau1, 24 tau2)
  Historical Reconciliation: Resolved 37 vs 38 count (confidence redesign on PACKET_0068)
                            Resolved CSS vs Delta_max column equivalence (X_max stored)
========================================================================================
```

---

## Table of Contents

| Chapter | Document Title | Description |
| :--- | :--- | :--- |
| **01** | [01_Failure_Analysis_Protocol.md](01_Failure_Analysis_Protocol.md) | Formal research protocol, frozen parameters, and operational definitions. |
| **02** | [02_Negative_DCRI_Forensic_Analysis.md](02_Negative_DCRI_Forensic_Analysis.md) | Mathematical causes, distribution, and safety invariants of negative DCRI. |
| **03** | [03_Extreme_Conflict_Dynamics.md](03_Extreme_Conflict_Dynamics.md) | Top 5% extreme cross-modality conflict audit and tier assignment characterization (18 Tier 1, 7 Tier 2, 0 Tier 0). |
| **04** | [04_Decision_Tier_Transitions.md](04_Decision_Tier_Transitions.md) | Paired transitions across B6 vs B1-B5, two-tier jumps, and asymmetry metrics. |
| **05** | [05_Uncertainty_and_Quality_Patterns.md](05_Uncertainty_and_Quality_Patterns.md) | Diagnostic analysis of Patterns A-D and near-threshold sensitivity. |
| **06** | [06_Discrepancy_Reconciliation.md](06_Discrepancy_Reconciliation.md) | Complete resolution of 37 vs 38 count, `css` vs `delta_max`, and 2-tier jump counts. |
| **07** | [07_Failure_Analysis_Report.md](07_Failure_Analysis_Report.md) | Comprehensive executive failure and error report. |
| **08** | [08_Verification_Audit.md](08_Verification_Audit.md) | Audit of the 10 forensic gates, input/output cryptographic digests, and seal. |

---

## Artifact Inventory (`experiments/fusion/failure_analysis/`)

| Artifact | Type | Abbreviated SHA-256 (Full in `08_verification_report.json`) | Rows / Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| `01_failure_analysis_protocol.md` | Protocol | `0bb25b29...` | 5.8 KB | Phase C11.18 protocol specification. |
| `02_negative_dcri_analysis.csv` | Data | `6cdf3d54...` | 37 rows | All 37 negative DCRI packets under Policy B6. |
| `03_conflict_extremes.csv` | Data | `9e3141f2...` | 25 rows | Top 5% extreme-conflict packets ($\Delta_{\max} \ge 0.9427$). |
| `04_tier_downgrade_analysis.csv` | Data | `f515704a...` | 500 rows | Complete paired transitions and shift directions vs B1-B5. |
| `05_uncertainty_quality_analysis.json` | JSON | `83e6cdcbb...` | 5.5 KB | Pattern A-D distributions, correlations, and paired statistics. |
| `06_discrepancy_reconciliation.md` | Audit | `bda33189...` | 8.3 KB | Forensic reconciliation of all inter-run discrepancies. |
| `07_failure_analysis_report.md` | Report | `d906e0caa...` | 14.4 KB | Full executive failure report across 10 sections. |
| `08_verification_report.json` | Manifest | `PASS` (10/10) | 4.7 KB | Machine-readable cryptographic audit report with full digests. |

---

## Readiness Certification
Phase C11.18 documents the specified failure patterns observed in the frozen 500-packet evaluation cohort. The reported automated checks passed the ten declared software gates. These findings support proceeding to the documentation and artifact-freeze stage (**Phase C11.19 — Final Research Freeze & Deployment Readiness**). They do not establish clinical safety, generalization to real patients, or readiness for clinical deployment.
