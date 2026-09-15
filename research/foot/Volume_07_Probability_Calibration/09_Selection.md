# Chapter 09 — Final Calibrator Selection Rationale

## 9.1 Multi-Metric Selection Hierarchy

As established in the selection framework, post-hoc calibration selection evaluates a multi-metric tradeoff between likelihood (NLL), error (ECE, MCE, Brier), prediction class behavior, and model parsimony.

### Decision Rule
> **Primary Criterion**: Lower validation NLL.  
> **Parsimony Tolerance Rule**: If Temperature Scaling validation NLL is within 0.005 of Vector Scaling validation NLL ($\text{NLL}_{\text{temp}} \le \text{NLL}_{\text{vec}} + 0.005$), Temperature Scaling is preferred for model parsimony (1 parameter vs 8 parameters) and strict prediction rank preservation. Otherwise, Vector Scaling is selected.

---

## 9.2 Empirical Evaluation against Rationale

1. **Validation NLL Comparison**:
   - Temperature Scaling Validation NLL: **0.8694**
   - Vector Scaling Validation NLL: **0.8044**
   - Difference ($\Delta \text{NLL} = \text{NLL}_{\text{temp}} - \text{NLL}_{\text{vec}}$): **0.0650** ($> 0.005$)

2. **Held-Out Test Set Performance**:
   - Vector Scaling achieves lower test NLL ($0.8749$ vs $0.8785$).
   - Vector Scaling achieves lower test ECE ($0.0313$ vs $0.0388$), representing a **26.18% ECE reduction** compared to raw probabilities ($0.0424$).
   - Vector Scaling improves test Macro F1 ($0.6758$ vs $0.6683$) and test accuracy ($0.6769$ vs $0.6690$).

---

## 9.3 Selection Decision

> **SELECTED METHOD: Vector Scaling (`vector_scaling`)**

The selected calibration configuration is frozen and exported to `experiments/foot/final_model/calibration.json`:

```json
{
  "method": "vector_scaling",
  "weights": [
    1.041,
    1.043,
    0.8711,
    1.1301
  ],
  "bias": [
    0.0292,
    0.0625,
    0.063,
    -0.1547
  ],
  "fit_split": "validation",
  "validation_nll": 0.8044,
  "test_nll": 0.8749,
  "test_ece": 0.0313,
  "selection_rationale": "Selected via lower validation NLL exceeding the 0.005 parsimony tolerance threshold.",
  "seed": 42,
  "status": "selected"
}
```
