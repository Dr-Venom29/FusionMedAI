# Controlled Risk Perturbation & Sensitivity Evaluation

## 1. Perturbation Protocol

To verify that conflict metrics respond predictably to shifts in individual modality risk, controlled risk sweeps were executed across $r_{\text{target}} \in [0.00, 1.00]$ while holding all other channel properties constant:

| Perturbed Modality Risk ($r_C$) | $\Delta_{\max}$ | $\Delta_{\text{mean}}$ | $\sigma_w$ | $R_{\text{fusion}}$ | $\text{DCRI}_{0.20}$ | Severity Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$r_C = 0.00$** | 0.488410 | 0.325607 | 0.205256 | 0.480035 | 0.368128 | **HIGH** |
| **$r_C = 0.10$** | 0.488410 | 0.325607 | 0.198906 | 0.505527 | 0.393620 | **HIGH** |
| **$r_C = 0.20$** | 0.488410 | 0.325607 | 0.195240 | 0.531019 | 0.419112 | **HIGH** |
| **$r_C = 0.35$** | 0.488410 | 0.325607 | 0.194729 | 0.569257 | 0.457350 | **HIGH** |
| **$r_C = 0.50$** | 0.488410 | 0.325607 | 0.200140 | 0.607495 | 0.495588 | **HIGH** |
| **$r_C = 0.75$** | 0.750000 | 0.500000 | 0.222384 | 0.671225 | 0.559318 | **HIGH** |
| **$r_C = 1.00$** | 1.000000 | 0.666667 | 0.258752 | 0.734955 | 0.623048 | **HIGH** |

---

## 2. Monotonicity & Robustness Properties

1. **Piecewise Monotonicity**: When an existing non-extreme modality is perturbed, $\Delta_{\max}$ remains constant as long as the perturbed channel does not exceed the envelope of the existing extreme modalities; once it exceeds the envelope, $\Delta_{\max}$ increases linearly.
2. **Dispersion Parabolic Minimum**: Weighted standard deviation $\sigma_w$ achieves a minimum when the perturbed channel risk equals the weighted consensus of the other channels, and increases quadratically as it moves outward.
3. **Deterministic Repeatability**: Identical float outputs obtained across repeated perturbation passes ($f(X) \equiv f(X)$).
