# Chapter 03 — Evaluation Protocol

## 3.1 Case Sampling & Full Test Set Execution

The explainability evaluation protocol operates across the complete frozen held-out test set ($N = 1,006$).

1. **Full Test Set Analysis ($N = 1,006$)**: Grad-CAM attributions and high-attribution spatial area fractions are computed for every test sample.
2. **Qualitative Sampling**:
   - Correct predictions: Representative cases sampled across Wagner Grades 1–4 ($N = 40$).
   - Misclassifications: Samples drawn across erroneous prediction pairs, with specific emphasis on Grade 2 $\leftrightarrow$ Grade 3 confusion cases ($N = 66$).
3. **Controlled Sanity-Check Subsample**: A reproducible, stratified sample ($N = 40$, Seed=42) evaluated for model randomization and target-class sensitivity.

---

## 3.2 Explanation Modes

For every sampled case, attributions are generated under two primary modes:
1. **Predicted-Class Attribution (Primary)**: Grad-CAM for the class corresponding to $\hat{y} = \arg\max_c P(y=c|X)$.
2. **Alternative-Class Attribution (Comparative)**: Grad-CAM for alternative classes $c \neq \hat{y}$ to evaluate class-discriminative behavior.

---

## 3.3 Quantitative Evaluation Diagnostics

### Diagnostic 1: High-Attribution Spatial Area Fraction
Measures the proportion of spatial pixels in the $224 \times 224$ feature attribution map where $\text{CAM}(x, y) \ge 0.5$. This diagnostic identifies diffuse or over-extended explanations across dataset subsets.

### Diagnostic 2: Attribution Mass Concentration
Calculates the proportion of total CAM intensity mass concentrated within the top 10% and top 20% highest activation pixels:

$$\text{Concentration}_K = \frac{\sum_{(x,y) \in \Omega_K} \text{CAM}(x, y)}{\sum_{(x,y)} \text{CAM}(x, y)}$$

where $\Omega_K$ denotes the set of top $K\%$ highest activation pixels.

### Diagnostic 3: Adebayo Model Randomization Test
Evaluates Pearson Correlation Coefficient ($\text{PCC}$) between Grad-CAM generated from the trained model vs an uninitialized (randomly parameterized) model. A near-zero correlation indicates that the learned model parameters materially determine the resulting Grad-CAM patterns.

### Diagnostic 4: Target-Class Contrast Test
Evaluates $\text{PCC}$ between Grad-CAM maps produced for the predicted class vs non-predicted classes. A near-zero correlation indicates that Grad-CAM maps differ substantially across target classes, supporting class-dependent attribution behavior.

> [!NOTE]
> **Ground-Truth Boundary Note**: Pointing Game hit rates and perturbation deletion/insertion AUCs are excluded from formal acceptance criteria due to the absence of clinically validated lesion segmentation ground-truth masks.
