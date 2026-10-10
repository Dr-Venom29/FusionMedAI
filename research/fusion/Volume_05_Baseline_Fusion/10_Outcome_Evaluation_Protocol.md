# Protocol: ACARA-U Outcome-Grounded Comparative Evaluation

> **Status:** 🟢 SEALED & VERIFIED (8/8 Gates Passed)  
> **Evaluation Design:** Synthetic Latent Oracle Simulation ($N=5,000$ Confirmatory Packets, 10 Independent Seeds `301`–`310`)  
> **Inference Method:** Two-Stage Hierarchical Cluster Bootstrap ($B=2,000$, Resampling Cohorts $\to$ Packets)  
> **Frozen Parameters:** $\Theta_0 = (1.0, 1.5, 1.0, 0.5)$, $\delta^* = 0.10$, $\tau_1 = 0.20, \tau_2 = 0.40$  
> **Prespecified Practical Superiority Threshold:** $\Delta_{\mathrm{MAE}} < -0.005$  

---

## 1. Executive Summary & Objective

The primary objective of this experiment is to evaluate:
> **Does full ACARA-U dynamic routing (B6) produce superior predictive accuracy and decision utility compared to simpler uncertainty-ablated fusion (B5), rather than merely redistributing routing weights?**

Because real-world clinical datasets for retina, foot ulcer, and EHR are unlinked and lack cross-patient ground truth, previous routing evaluations (C11.1–C11.15) were conducted on controlled uncalibrated decision packets. This protocol establishes an **outcome-grounded synthetic oracle benchmark** where true latent risk $Y^* \in [0.02, 0.98]$ is generated prior to and independently of modality observations.

---

## 2. Research Hypotheses & Evaluation Metrics

| Hypothesis | Comparison | Primary Metric | Prespecified Decision Criteria | Confirmatory Verdict (Synthetic Oracle) |
| :--- | :--- | :--- | :--- | :--- |
| **H1 (Primary Outcome)** | B6 vs B5 across full synthetic cohort | $\Delta_{\mathrm{MAE}} = \mathrm{MAE}_{\mathrm{B6}} - \mathrm{MAE}_{\mathrm{B5}}$ | **Dual Condition:**<br>1. Practical Superiority: $\Delta_{\mathrm{MAE}} < -0.005$<br>2. Statistical Significance: $95\%$ Hierarchical CI strictly $< 0$ | **STATISTICALLY SIGNIFICANT MODEST IMPROVEMENT (Practical Threshold Not Met)**<br>• Condition 1: **NOT MET** ($\Delta_{\mathrm{MAE}} = -0.001843 \ge -0.005$)<br>• Condition 2: **MET** ($95\%$ Hierarchical CI: $[-0.002112, -0.001582]$, cohort $t = -20.30$, $p = 7.95 \times 10^{-9}$) |
| **H2 (Quality Contribution)** | B6 ($\eta=0.5$) vs $\mathrm{B6}_{\eta=0}$ (B5) | Stratified by sensor fidelity | Dynamically evaluated across fidelity subsets with hierarchical subgroup CIs | **CONFIRMED (Observed Quality Benefit Under Accurate Sensing)**<br>Accurate Degradation ($N=771$): $\Delta = -0.008420$ ($95\%$ CI: $[-0.009344, -0.007530]$)<br>Undetected: $\Delta = -0.000417$<br>False Alarm: $\Delta = +0.001481$ |
| **H3 (Degradation Scaling)** | B6 vs B5 across degradation tiers | $\Delta_{\mathrm{MAE}}$ stratified by mild, moderate, severe corruption | Strict monotonic error reduction across degradation tiers: $\Delta_{\mathrm{MAE}}^{\mathrm{severe}} < \Delta_{\mathrm{MAE}}^{\mathrm{mod}} < \Delta_{\mathrm{MAE}}^{\mathrm{mild}} < 0$ (clean inputs evaluated as uncorrupted reference) | **CONFIRMED (Monotonic Reduction Across Degradation Conditions)**<br>• Severe: $\Delta_{\mathrm{MAE}} = -0.015416$ ($17.16\%$ relative reduction)<br>• Moderate: $\Delta_{\mathrm{MAE}} = -0.003904$ ($5.21\%$ reduction)<br>• Mild: $\Delta_{\mathrm{MAE}} = -0.001994$ ($3.31\%$ reduction)<br>• Clean Reference: $\Delta_{\mathrm{MAE}} = +0.000198$ ($0.36\%$ increase) |
| **H4 (Tail Robustness)** | B6 vs B5 by availability regime | Stratified MAE across cardinality $|\mathcal{A}| \in \{1, 2, 3\}$ | Singleton invariance ($\Delta = 0.0$), multi-modality gain | **CONFIRMED** ($\Delta = 0.0$ on singletons, $-0.002$ to $-0.004$ on multi) |
| **H5 (DCRI Policy Utility)** | DCRI ($\delta=0.10$) vs Fused Risk Policy | Expected Decision Loss & High-Risk False Downgrades | Loss trade-off characterization under illustrative asymmetric cost matrix | **CONFIRMED (Explicit Trade-Off in Synthetic Cases)**<br>False downgrade rate increases ($7.61\% \to 12.72\%$) |

---

## 3. Synthetic Oracle Data-Generating Process

1. **Latent Target:** $Y^* \sim \mathrm{Beta}(2.0, 2.0)$ bounded in $[0.02, 0.98]$.
2. **Modality Noise:** $\sigma_{\mathrm{retina}} = 0.075 < \sigma_{\mathrm{foot}} = 0.095 < \sigma_{\mathrm{clinical}} = 0.140$, mirroring validation reliability priors.
3. **Imperfect Sensing & Miscalibration:** Includes clean cases ($60.4\%$), mild degradation ($7.2\%$), moderate degradation ($16.1\%$), severe degradation ($7.7\%$), and uncertainty miscalibration scenarios ($8.6\%$).
4. **Independent Execution:** Oracle target $Y^*$ and all modality records are generated and sealed before any routing algorithm is executed.

---

## 4. Confirmatory Scoreboard ($10$ Cohorts, $N=5,000$ Synthetic Packets)

| Metric / Dimension | B2 (Uniform) | B5 (Conf+Rel-Unc) | B6 (Full ACARA-U) | Paired $\Delta (\mathrm{B6} - \mathrm{B5})$ | 95% Hierarchical Cluster CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall Mean MAE** | $0.068835$ | $0.060955$ | **$0.059113$** | **$-0.001843$** | $[-0.002112, -0.001582]$ |
| **Clean Scenarios MAE (Reference)** | $0.063852$ | $0.055285$ | $0.055483$ | $+0.000198$ | $[-0.000051, +0.000445]$ |
| **Mild Degradation MAE** | $0.067340$ | $0.060292$ | $0.058297$ | $-0.001994$ | $[-0.002810, -0.001178]$ |
| **Moderate Degradation MAE** | $0.079980$ | $0.074864$ | $0.070961$ | $-0.003904$ | $[-0.004812, -0.002996]$ |
| **Severe Degradation MAE** | $0.096540$ | $0.089828$ | **$0.074412$** | **$-0.015416$** | $[-0.017920, -0.012912]$ ($17.16\%$ relative MAE reduction) |
| **Uncertainty Miscalibration MAE** | $0.069120$ | $0.049440$ | $0.049397$ | $-0.000043$ | $[-0.000620, +0.000534]$ |
| **Singleton Regimes (`R`, `F`, `C`)** | $0.079850$ | $0.079850$ | $0.079850$ | **$0.000000$** | $[0.000000, 0.000000]$ (Exact invariance) |
| **Decision Cost Loss ($\mathcal{L}_{\mathrm{cost}}$)** | $0.2450$ | $0.2142$ | **$0.2100$** | **$-0.0042$** | $[-0.0060, -0.0024]$ |

---

## 5. Artifact Manifest & Cryptographic Hashes

All outcome evaluation artifacts are stored in `experiments/fusion/outcome_evaluation/` and verified via `verification/fusion/outcome_evaluation/verify_outcome_artifacts.py`:

| Artifact | Purpose | SHA-256 Hash | Verification Status |
| :--- | :--- | :--- | :---: |
| `protocol.json` | Frozen experimental protocol and hypotheses | `093913fae1f09247c02bca91d929713776fd388c8e143ab56e3ab9ab644987eb` | **VERIFIED (OE-08)** |
| `dev_results.json` | Development seeds validation ($N=2,500$) | `b19c0c641e1fd45c064cf5e0402bfc27e6a7f2dd0658b3bd57ecfa434a7ba386` | **VERIFIED (OE-08)** |
| `confirmatory_results.json` | Confirmatory cohorts raw records ($N=5,000$) | `c9f48fc8e2be6c73a35fefa5ff166ed7ccfd94364471a8310774f2a3da996fbc` | **VERIFIED (OE-08)** |
| `summary.json` | Confirmatory synthesis and hypothesis verdicts | `ec3970670b31ff004ea1871c1d805d048bc9f902939c91badb1f27c49fe7dedc` | **VERIFIED (OE-08)** |
| `freeze_manifest.json` | Cryptographic seal of all 4 artifacts | — | **SEALED** |

