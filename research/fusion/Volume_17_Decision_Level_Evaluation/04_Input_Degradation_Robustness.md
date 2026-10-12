# Chapter 04: Input Degradation Robustness

## 1. Input Degradation Protocol
In clinical workflows, input quality degrades due to sensor noise, optical blur, motion artifact, poor contrast, illumination drift, and incomplete clinical documentation. Dimension C evaluates the end-to-end response of ACARA-U (B6) and comparator B5 under systematic input degradation across four pre-defined severities:
- **D0 (Reference / Clean)**: Unperturbed reference inputs.
- **D1 (Mild Degradation)**: Gaussian blur kernel ($\sigma = 1.5$ px), mild contrast compression (scale factor $0.85$, clipped to $[0, 1]$), $-25$ illumination shift (linear offset $-0.10$ normalized), and 1 omitted clinical domain.
- **D2 (Moderate Degradation)**: Gaussian blur kernel ($\sigma = 3.5$ px), substantial contrast reduction (scale factor $0.65$), $-45$ illumination shift (offset $-0.18$ normalized), and 2 omitted clinical domains.
- **D3 (Severe Degradation)**: Gaussian blur kernel ($\sigma = 7.0$ px), heavy optical opacity attenuation (scale factor $0.40$), $-60$ illumination drop (offset $-0.24$ normalized), and 3 omitted clinical domains.

Perturbations are applied deterministically and identically across the canonical $N=500$ evaluation cohort (seed 115) using the frozen degradation operators defined in `src/fusion/degradation/degradation_spec.py`. Live quality computation engines calculate channel qualities, which update channel routing and predictive uncertainties. Comparator coverage focuses on B6 versus B5 to specifically evaluate whether the adaptive quality bonus weight ($\eta = 0.5$ in B6 vs $\eta = 0.0$ in B5) attenuates degraded channel authority relative to fixed priors.

## 2. Quantitative Degradation Response (ACARA-U B6)
The table below documents the cohort-level metrics observed across the four degradation severities ($N=500$, paired across identical packet identities):

| Severity Level | Mean Quality $\overline{Q}$ | Mean Risk $\overline{R}_{\text{fusion}}$ | Mean Penalty $\overline{P}_U$ | Mean DCRI $\overline{\text{DCRI}}$ | Routine (Tier 0) | Assessment (Tier 1) | Escalation (Tier 2) | Negative DCRI Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D0 (Clean)** | 0.8142 | 0.289900 | 0.063394 | 0.226506 | 232 (46.4%) | 187 (37.4%) | 81 (16.2%) | 37 (7.4%) |
| **D1 (Mild)** | 0.7289 | 0.283402 | 0.071391 | 0.212011 | 248 (49.6%) | 183 (36.6%) | 69 (13.8%) | 42 (8.4%) |
| **D2 (Moderate)** | 0.5841 | 0.276985 | 0.083231 | 0.193754 | 263 (52.6%) | 178 (35.6%) | 59 (11.8%) | 69 (13.8%) |
| **D3 (Severe)** | 0.3812 | 0.277318 | 0.101519 | 0.175799 | 280 (56.0%) | 165 (33.0%) | 55 (11.0%) | 100 (20.0%) |

## 3. Analysis of Degradation Dynamics

### 3.1 Epistemic Penalty Surge vs Risk Drift
As degradation severity escalates from D0 to D3:
1. **Uncertainty Penalty**: Mean uncertainty penalty $\overline{P}_U$ climbs monotonically from $0.063394$ to $0.101519$ ($+60.1\%$) due to degradation-induced uncertainty accumulation across channels.
2. **Fused Risk Drift**: Fused risk shifts downward slightly from $0.289900$ to $0.277318$ ($\Delta = -0.012582$), as dynamic weighting redistributes authority toward surviving, less degraded channels.
3. **Net DCRI Suppression**: Because the uncertainty penalty is subtracted from fused risk under the frozen formulation ($\text{DCRI} = R_{\text{fusion}} - \delta \sum_{i \in \mathcal{A}} U_i$), the surge in penalty drives net DCRI monotonically downward from $0.226506$ to $0.175799$. Unclamped negative DCRI instances rise from 37 (7.4%) to 100 (20.0%).

### 3.2 Paired Differences and Bootstrap Evaluation
Paired differences relative to Clean D0 were evaluated across identical packet pairs ($N=500$, $B=1,000$ bootstrap iterations, seed 7117, empirical percentile method):
- **D1 vs D0**: $\Delta_{\text{DCRI}} = -0.014495$ (95% CI: $[-0.015501, -0.013526]$, excludes zero).
- **D2 vs D0**: $\Delta_{\text{DCRI}} = -0.032752$ (95% CI: $[-0.033619, -0.031938]$, excludes zero).
- **D3 vs D0**: $\Delta_{\text{DCRI}} = -0.050707$ (95% CI: $[-0.051714, -0.049702]$, excludes zero).

### 3.3 Critical Findings & Decision-Tier Impact
As degradation severity increases, the mean uncertainty penalty rises from 0.063394 at D0 to 0.101519 at D3, while mean DCRI falls from 0.226506 to 0.175799. Tier 2 assignments also decrease from 16.2% to 11.0%, while Tier 0 assignments increase from 46.4% to 56.0%. 

Under the frozen subtractive DCRI formulation, increased uncertainty can therefore lower the assigned decision tier rather than increase escalation. This behavior is a potential under-escalation concern and requires explicit investigation before any clinical-use claims. These results establish the observed response to the defined perturbations; they do not establish that degraded inputs are handled safely.
