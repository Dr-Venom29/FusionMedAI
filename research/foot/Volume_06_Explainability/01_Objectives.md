# Chapter 01 — Objectives & Research Position

## 1.1 Core Research Question

Phase 10.6 evaluates the post-hoc spatial explainability of the frozen Phase 10.5 primary Foot Ulcer classification model (**EfficientNet-B3**). 

The primary research question is:

> **Does the selected EfficientNet-B3 model base its Wagner-grade predictions on anatomically relevant and visually meaningful wound regions (ulcer bed, periwound margin, necrotic tissue), rather than obvious image capture artifacts, background skin, or framing borders?**

---

## 1.2 Distinction Between Implementation & Research Evaluation

Generating a spatial heat map using Grad-CAM is an implementation step. Research evaluation requires:

1. **Pre-specifying an evaluation protocol** prior to inspecting results to avoid confirmation bias.
2. **Empirically verifying target layer representations** on the frozen network topology.
3. **Evaluating explanations on both correct predictions and misclassifications** to diagnose model failure modes.
4. **Quantifying attribution behavior using dataset-wide attribution concentration and controlled sanity-check diagnostics.** Where validated spatial ground-truth annotations are unavailable, attribution is not evaluated as lesion-localization accuracy.

---

## 1.3 Hypotheses

- **$H_1$ (Anatomical Grounding)**: High-attribution regions in correctly classified cases will preferentially overlap visible wound and periwound regions rather than unrelated background or image borders.
- **$H_2$ (Grade-Specific Attribution)**: Different Wagner grades will exhibit distinguishable attribution patterns over visible wound characteristics, with Grade 4 expected to show stronger attribution around visibly necrotic/gangrenous regions.
  - *Grade 1*: Focused on visible superficial epidermal/dermal lesions.
  - *Grade 2*: Focused on open wound bed regions.
  - *Grade 3*: Focused on visible surface characteristics associated with Grade 3 classification.
  - *Grade 4*: Focused on dark, visibly necrotic/gangrenous tissue.
- **$H_3$ (Error Attribution Diagnostic)**: Misclassifications (particularly between Wagner Grade 2 and Grade 3) are consistent with visual similarity between wound-bed characteristics observed in Grade 2 and Grade 3 cases, rather than spurious background shortcuts.
