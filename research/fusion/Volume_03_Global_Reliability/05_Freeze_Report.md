# 05 Reliability Freeze Report & Router Handoff — Phase C11.3

## 1. Executive Summary & Reliability Lock

Phase C11.3 concludes by freezing the static modality-level reliability priors into immutable constants:

```json
{
  "retina_reliability_R_R": 0.929956,
  "foot_reliability_R_F": 0.922266,
  "clinical_reliability_R_C": 0.825382
}
```

These values are locked in [`src/fusion/reliability/global_reliability.py`](../../src/fusion/reliability/global_reliability.py) and [`experiments/fusion/reliability/global_reliability.json`](../../experiments/fusion/reliability/global_reliability.json).

---

## 2. Downstream Router Interface (Phase C11.4 Handoff)

In Phase C11.4, the ACARA-U dynamic router will ingest these priors directly into the logit kernel:

$$z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$$

### Interface Contract:
- **`R_R = 0.929956`** for Retina modality channel.
- **`R_F = 0.922266`** for Foot ulcer modality channel.
- **`R_C = 0.825382`** for Clinical EHR modality channel.

If a modality is unavailable ($A_i = 0$), the hard availability mask applies:

$$\tilde{z}_i = -\infty \implies w_i = 0$$

Regardless of the value of $R_i$.
