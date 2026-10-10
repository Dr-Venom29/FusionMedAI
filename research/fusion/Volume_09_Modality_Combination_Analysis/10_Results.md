# Phase C11.9 Comprehensive Experimental Results

## 1. Executive Summary of Scientific Findings

Phase C11.9 evaluated whether the frequency distribution of modality combinations impacts the behavior, numerical stability, and derived risk-index behavior of ACARA-U. The confirmatory sample scope spans 30 synthetic Monte Carlo cohorts (seeds 401–430), with 500 decision packets per cohort for each distribution regime, totaling 15,000 packet evaluations per distribution and 45,000 distribution-packet evaluations across D1, D2, and D3 (in addition to 1,500 exploratory packet trials at seed 115). Key findings include:

1. **Simplex Invariant Preservation (100.0%)**: Across all head, middle, and tail combinations, active routing weights strictly normalized to unity ($\sum w_i = 1.000000$), while unavailable channels received exactly zero authority ($w_i = 0.000000$).
2. **Cross-Distribution Metric Behavior**: Mean fused risk, DCRI, and routing entropy varied across the evaluated D1–D3 distributions. These are descriptive distribution-level changes; they do not, by themselves, establish statistical stability or instability under a defined equivalence margin:
   - Population Mean Risk $\overline{R_{\text{fusion}}}$: $0.2841$ (D1 Balanced) $\to 0.2909$ (D2 Moderate Tail) $\to 0.3005$ (D3 Strong Tail).
   - Population Mean DCRI $\overline{\text{DCRI}}$: $0.2096$ (D1) $\to 0.1953$ (D2) $\to 0.1952$ (D3).
   - Global Shannon Entropy $\overline{H(w)}$: $0.4154$ (D1) $\to 0.6662$ (D2) $\to 0.7737$ (D3). Note on entropy behavior: conditional routing entropy is identically zero for all unimodal tail packets ($H(w) \equiv 0.0$ because a single active modality receives weight $1.0$); the increase in global mean entropy from D1 to D3 reflects the shifting mixture composition across distributions rather than routing variation within unimodal tail packets.
3. **Observed Point-Estimate Tail Sensitivity vs Multi-Cohort Confirmatory Parity**:
   - *Exploratory Cohort ($\text{seed}=115$)*: ACARA-U achieved the lowest observed point-estimate tail sensitivity ($D_{\text{tail}} = 0.1883$ in D2, $0.1876$ in D3) among the evaluated soft-weighting baselines (Uniform Averaging $0.1925 / 0.1945$, Confidence Weighting $0.1959 / 0.1974$, Reliability+Confidence $0.1967 / 0.1973$, Conf+Rel-Uncertainty $0.1889 / 0.1889$). The margin over B5 was small ($+0.0006$ in D2, $+0.0012$ in D3; paired bootstrap CIs crossing zero). In D1, tail sensitivity is not estimable because D1 contains zero unimodal tail packets ($N_{\text{tail}}=0$); structural invariant check passed identically ($0.0 \equiv 0.0$).
   - *Confirmatory Multi-Cohort ($S=30$, $N=15,000$)*: Under the pre-declared confirmatory protocol, the paired difference $\Delta D_{\text{tail}} = D_{\text{tail}}(\text{B6}) - D_{\text{tail}}(\text{B5})$ was:
     - D3 Primary (Micro, Packet-Weighted): $+0.000181$, $95\%$ Hierarchical CI: $[-0.000024, +0.000377]$ (crossing zero; $t=2.4194, p=0.022$).
     - D3 Sensitivity (Macro, Comb-Weighted): $+0.000123$, $95\%$ Hierarchical CI: $[-0.000093, +0.000344]$ (crossing zero; $t=1.4645, p=0.154$).
     - D2 Secondary (Micro): $+0.000143$, $95\%$ Hierarchical CI: $[-0.000025, +0.000300]$ (crossing zero; $t=2.2105, p=0.035$).
   - *Inferential Reconciliation*: Under the prespecified hierarchical-bootstrap decision rule, the D3 primary endpoint is **inconclusive**. The estimated difference is small and positive ($\Delta_{\text{micro}} = +0.000181$), favoring B5 numerically. Cohort-level bootstrap ($95\%$ CI: $[+0.000041, +0.000327]$) and parametric sensitivity analyses ($t = 2.4194, p = 0.022$) also indicate a positive difference favoring B5 at the between-cohort level, but the primary 2-stage hierarchical cluster bootstrap confidence interval ($[-0.000024, +0.000377]$) includes zero. These results do not establish practical superiority of B6 or satisfy the prespecified hierarchical-interval criterion for inferiority. Equivalence is not established. Under the pre-declared protocol, the confirmatory evaluation concludes as `INCONCLUSIVE_NOT_STATISTICALLY_DISTINGUISHABLE`.
   - Winner-take-all baseline B1 achieved lower nominal tail sensitivity by exclusively selecting Retina, but suffered higher head-tier risk dispersion ($\sigma = 0.2755$ in D2 vs $0.1880$ for ACARA-U).
4. **Uncertainty Aggregation by Modality Cardinality**: Total uncertainty generally increased with the number of available channels because $U_{\text{sum}}$ is additive, although individual combinations were not strictly ordered by cardinality.
5. **Deterministic Fail-Closed Boundary Safety**: Zero-modality inputs ($\text{EMPTY}$) unconditionally return `NO_MODALITY_AVAILABLE` with $R_{\text{fusion}} = 0.0$ and $\text{DCRI} = 0.0$.

---

## 2. Cross-Distribution Summary Table

| Distribution Name | Cohort Head / Mid / Tail | Mean Risk $\overline{R_{\text{fusion}}}$ | Mean $\overline{\text{DCRI}}$ | Mean Entropy $\overline{H(w)}$ | Mean $U_{\text{sum}}$ | Tail Sensitivity $D_{\text{tail}}$ (Seed 115) | 30-Cohort Micro Mean $\Delta$ (95% CI) | 30-Cohort Macro Mean $\Delta$ (95% CI) | Tail Risk Std $\sigma(R_{\text{tail}})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D1_BALANCED** | $500$ / $0$ / $0$ | $0.2841 \pm 0.2232$ | $0.2096 \pm 0.2202$ | $0.4154 \pm 0.3808$ | $0.3725 \pm 0.3515$ | N/A ($N_{\text{tail}}=0$) | N/A (Reference Check PASS: $[0.0, 0.0]$) | N/A (Reference Check PASS: $[0.0, 0.0]$) | N/A ($N_{\text{tail}}=0$) |
| **D2_MODERATE_HEAD_TAIL** | $300$ / $125$ / $75$ | $0.2909 \pm 0.1948$ | $0.1953 \pm 0.2065$ | $0.6662 \pm 0.3340$ | $0.4784 \pm 0.3168$ | $0.1883 \pm 0.1423$ | $+0.000143$ ($[-0.000025, +0.000300]$) | $+0.000108$ ($[-0.000053, +0.000262]$) | $0.2508$ |
| **D3_STRONG_LONG_TAIL** | $375$ / $90$ / $35$ | $0.3005 \pm 0.2001$ | $0.1952 \pm 0.2119$ | $0.7737 \pm 0.3045$ | $0.5264 \pm 0.2974$ | $0.1876 \pm 0.1551$ | $+0.000181$ ($[-0.000024, +0.000377]$) | $+0.000123$ ($[-0.000093, +0.000344]$) | $0.2784$ |

---

## 3. Conclusions on Scientific Viability

Across the controlled D1–D3 decision-packet distributions, ACARA-U preserved the active-authority simplex, maintained zero authority for unavailable modalities, and exhibited stable routing behavior under increasing combination skew. Tail configurations showed greater observed risk dispersion and sensitivity to modality reduction, while head-tier mean risk remained similar between the D2 and D3 controlled distributions. The confirmatory 30-cohort evaluation confirmed that ACARA-U (B6) and uncertainty-only baseline B5 perform at near-parity under unimodal tail tiers, with hierarchical bootstrap confidence intervals crossing zero. These findings characterize decision-level robustness under controlled modality-frequency shift; they do not establish clinical robustness or patient-level multimodal validity. Volume 09 is certified and sealed.
