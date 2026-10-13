# Chapter 12: Comprehensive Results Summary

## 1. Executive Summary
This report summarizes the empirical outcomes of **Phase C11.17: Final Decision-Level Evaluation of ACARA-U** across the canonical $N=500$ evaluation cohort (seed 115). All quantitative values are derived directly from the canonical artifacts (`03_canonical_records.json`, `04_summary_metrics.json`, `06_statistical_analysis.json`) and verified by the independent verification suite (`08_verification_report.json`).

### Primary Decision Metrics Across All Evaluated Policies ($N=500$)
| Metric / Policy | B1 (Unimodal WTA) | B2 (Uniform Avg) | B3 (Confidence) | B4 (Conf + Rel) | B5 (Conf+Rel-Unc) | B6 (Full ACARA-U) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Fused Risk $\overline{R}_{\text{fusion}}$** | 0.255037 | 0.296804 | 0.282063 | 0.287679 | 0.257236 | **0.261017** |
| **Mean DCRI $\overline{\text{DCRI}}$** | 0.191643 | 0.233410 | 0.218670 | 0.224286 | 0.193843 | **0.197624** |
| **Mean Penalty $\overline{P}_U$** | 0.063394 | 0.063394 | 0.063394 | 0.063394 | 0.063394 | **0.063394** |
| **Routing Entropy $\overline{H}$** | 0.000000 | 1.098612 | 1.084120 | 1.084262 | 1.040305 | **1.038987** |
| **Mean Routing Weights $(\bar{w}_R, \bar{w}_F, \bar{w}_C)$** | $(1.0000, 0.0000, 0.0000)$ | $(0.3333, 0.3333, 0.3333)$ | $(0.3368, 0.3145, 0.3487)$ | $(0.3496, 0.3241, 0.3263)$ | $(0.4094, 0.2199, 0.3707)$ | **$(0.4271, 0.2231, 0.3498)$** |
| **Routine Review (Tier 0)** | 275 (55.0%) | 208 (41.6%) | 239 (47.8%) | 230 (46.0%) | 269 (53.8%) | **262 (52.4%)** |
| **Additional Assessment (Tier 1)** | 84 (16.8%) | 240 (48.0%) | 206 (41.2%) | 208 (41.6%) | 178 (35.6%) | **181 (36.2%)** |
| **Escalation (Tier 2)** | 141 (28.2%) | 52 (10.4%) | 55 (11.0%) | 62 (12.4%) | 53 (10.6%) | **57 (11.4%)** |
| **Negative DCRI Count** | 223 (44.6%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 36 (7.2%) | **38 (7.6%)** |

## 2. Key Empirical Findings Across Dimensions

1. **Reference Invariant Concordance**: Full ACARA-U (B6) produces Mean $R_{\text{fusion}} = 0.261017$, Mean $\text{DCRI} = 0.197624$, Mean Routing Entropy $\overline{H} = 1.038987$, and 38 negative DCRI records ($7.6\%$).
2. **Quality Bonus Routing Shift (B6 vs B5)**: Adding the adaptive quality bonus weight ($\eta = 0.5$) shifts mean fused risk by $\Delta = +0.003781$ ($0.257236 \to 0.261017$) across identical inputs and redirects weight toward higher-quality channels ($\Delta w_R = +0.017739$). This isolates an observable routing shift; it does not demonstrate improved clinical diagnostic accuracy.
3. **Availability Mask Enforcement**: Across 16,000 availability regime records (4 policies $\times$ 8 regimes $\times$ 500 packets), zero inactive weight leaks were observed ($w_m \equiv 0.0$ across all 5 channel maps in `DCRIResult`). Unimodal singletons collapse mathematically to unity routing across all 6,000 tested singleton records, and empty inputs strictly trigger the fail-closed software fallback (`NO_MODALITY_AVAILABLE`) across 2,000 checked empty records.
4. **Input Degradation Under-Escalation**: As input corruption increases from D0 to D3, mean uncertainty penalty climbs from $0.063394$ to $0.101519$, while mean DCRI falls from $0.197624$ to $0.146913$. This highlights a structural property of subtractive uncertainty formulations requiring safety investigation.
5. **Conflict Stratification**: In tri-modal encounters ($N=500$), continuous discordance stratifies into LOW ($N=45$), MODERATE ($N=92$), and HIGH ($N=363$) with mean max pairwise disagreement $\overline{\Delta}_{\max} = 0.520331$ and weighted conflict dispersion $\bar{\sigma}_w = 0.214882$ ($\Theta_0$) / $0.214306$ ($\Theta_{\text{unit}}$).
6. **Calibration Condition**: Probability calibration produces a modest upward risk and DCRI shift ($\Delta = +0.004298$, $0.256720 \to 0.261017$) while routing weights remain invariant under risk-only calibration.

## 3. Scope & Research Limitations
1. **Evaluation Cohort**: Results are evaluated across a fixed synthetic cohort of $N=500$ packets under deterministic seed 115. They do not constitute an evaluation on real-world clinical patient populations.
2. **Absence of Clinical Gold Standards**: The evaluation dataset does not include verified patient outcomes or pathologist-confirmed diagnostic ground truth. Observed tier assignments reflect algorithmic policy mappings under frozen thresholds ($\tau_1=0.20, \tau_2=0.40$), not clinically validated triage accuracy.
3. **Subtractive Uncertainty Limitation**: Decreased escalation under degraded inputs demonstrates that subtracting epistemic uncertainty lowers assigned decision tiers. This trade-off must be addressed before any clinical deployment claims.

## 4. Upstream Modality Checkpoints, Validation Splits & Contract Governance

All C11 evaluations consume the frozen upstream checkpoints, calibration models, and validation splits recorded in `experiments/fusion/reliability/provenance.json`:

| Modality | Artifact Role | Repository Path | Verified SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Retina** (`efficientnet_b3`) | Model Checkpoint | `experiments/retina/efficientnet_b3/checkpoints/best_model.pt` | `263e8a3f9a9fb7aa995a8aafc40f13b9741d7da02889cf69b3168a9e569f17f2` |
| **Retina** | Temperature Scaling ($T=1.6218$) | `experiments/retina/calibration/v004_temperature_scaling/temperature_scaling.pt` | `df64b3d1fe86ba23279b80c69d90273f99103380ab2ce1ff1064a3ce61b9ab4f` |
| **Retina** | Validation Split ($N_{\text{val}}=366$) | `datasets/retina/processed/splits/val.csv` | `3e94a150a6fa84d3af2a296db3bec1f4dadb506195eaa2b2d6e81ca2719339e1` |
| **Foot** (`efficientnet_b3`) | Model Checkpoint | `experiments/foot/architecture_benchmark/efficientnet_b3/checkpoints/best_model.pt` | `c264b95367a2416eb3d99ada7ba93afba9d714b0394b5586a7dc85df95183cbb` |
| **Foot** | Vector Scaling | `experiments/foot/calibration/vector_scaling/vector_scaling.pt` | `fdaf2f936140e424b0f8168ed24d895ca7da992238091224dd9efae7bd01974b` |
| **Foot** | Validation Split ($N_{\text{val}}=1,008$) | `datasets/foot/processed/splits/val.csv` | `e26a2e908b2b057100733cb1bc05d01d78b8167c2a3904686cc8efd779a57f9c` |
| **Clinical** (`CatBoost_HPO`) | Training Split ($N_{\text{train}}=70,118$) | `datasets/clinical/processed/splits/train.csv` | `708b3cd8bd788acda579d51d357fbbb51ad19ed60ac65cd3fe9a15deeddbec06` |
| **Clinical** (`CatBoost_HPO`) | Validation Split ($N_{\text{val}}=14,911$) | `datasets/clinical/processed/splits/val.csv` | `5fb03e2ca10025370ea01cd1cc76faae99a241269acbc05319d3ba14bc7f315c` |
| **Fusion Cohort** | Canonical $N=500$ Packet Manifest (`seed=115`) | `experiments/fusion/canonical_redesigned_rerun/01_canonical_redesigned_packet_manifest.json` | `cf877d74e306d6dbc3ac7d9e2485b1c46da4cd64a8e35a82fe68274854dc4cd4` |

### Enforced Contract & Invariant Protections (`verification/fusion/contract/test_evidence_orchestration.py`)
1. **Target Identity**: Every modality output (`ModalityPrediction`) declares its distinct clinical target (`diabetic_retinopathy_5class_severity`, `diabetic_foot_wagner_grade_binary_4class`, `hospital_readmission_30day`) and label space.
2. **Cross-Target Rejection**: Attempting to aggregate heterogeneous clinical targets into a unified patient clinical probability raises `CrossTargetAggregationError`; `EvidenceReport.common_target_prediction` is strictly `None`.
3. **Exact Shannon Entropy & 5-Map Inactive Zeroing**: `RouterResult` and `RoutingResult` enforce $H(w) = -\sum_{i \in \mathcal{A}, w_i > 0} w_i \ln w_i$, and `DCRIResult` enforces zero values across all 5 channel maps (`weights`, `modality_risks`, `modality_uncertainties`, `weighted_risk_contributions`, `uncertainty_penalty_contributions`) for inactive modalities.
4. **Fail-Closed Empty Input Handling**: An empty modality input safely produces status `NO_MODALITY_AVAILABLE` with zero weights across all channels.
