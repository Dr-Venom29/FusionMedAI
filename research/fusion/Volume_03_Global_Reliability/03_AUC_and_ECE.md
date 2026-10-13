# 03 AUC and ECE Evaluation Protocol — Phase C11.3

## 1. Discrimination Capacity Formulation ($\text{AUC}_i$)

To maintain complete methodological parity across multi-class vision tasks and binary tabular tasks:

### 1.1 Multi-Class Modalities (Retina & Foot)
Both Retina ($K=5$) and Foot ($K=4$) utilize the **Macro One-vs-Rest (OvR) ROC-AUC**:

$$\text{AUC}_{\text{macro-ovr}} = \frac{1}{K} \sum_{k=0}^{K-1} \text{AUC}_k$$

Where $\text{AUC}_k$ is the area under the binary ROC curve for class $k$ treated as the positive target against all remaining classes.

### 1.2 Binary Modality (Clinical)
The clinical modality ($K=2$) uses the standard binary ROC-AUC evaluated on the positive 30-day readmission class probability:

$$\text{AUC}_{\text{clinical}} = \text{ROC-AUC}(y, p_1)$$

---

## 2. Expected Calibration Error Formulation ($\text{ECE}_i$)

To ensure bin count and binning methodology consistency across modalities:

### 2.1 10-Bin Equal-Frequency (Quantile) Partitioning
Rather than using arbitrary fixed-width bins (which produce empty bins in low-density probability regions), ECE is computed using $M=10$ **equal-frequency (quantile) bins**:

$$q_m = \text{Quantile}\left( \text{confidences}, \frac{m}{M} \right), \quad m \in \{0, 1, \dots, 10\}$$

### 2.2 Mathematical Formula
For multi-class models, confidence is defined as $\hat{p} = \max_k P(Y=k \mid x)$ and accuracy is $\mathbb{I}(\hat{y} = y)$. For binary models, confidence is $p_1$ and accuracy is $y$.

$$\text{ECE} = \sum_{m=1}^{10} \frac{\lvert B_m \rvert}{N} \cdot \left\lvert \text{acc}(B_m) - \text{conf}(B_m) \right\rvert$$

Where:
- $\lvert B_m \rvert$ is the number of samples falling in quantile bin $m$.
- $\text{acc}(B_m) = \frac{1}{\lvert B_m \rvert} \sum_{i \in B_m} \text{acc}_i$ is the empirical empirical accuracy in bin $m$.
- $\text{conf}(B_m) = \frac{1}{\lvert B_m \rvert} \sum_{i \in B_m} \text{conf}_i$ is the average predicted confidence in bin $m$.


---

## 3. Clinical Calibration Split Provenance (`val.csv` vs. Locked `test.csv`)

For the Clinical modality (`CatBoost_HPO`), post-hoc calibrators are fitted on `datasets/clinical/processed/splits/val.csv` ($N_{\text{val}} = 14,911$) while `datasets/clinical/processed/splits/test.csv` ($N_{\text{test}} = 14,913$) remains strictly locked:

| Clinical Calibration Method | Validation ROC-AUC ($N=14,911$) | Validation ECE ($10$-bin eq-freq) | Validation Brier | Locked Test ROC-AUC ($N=14,913$) | Locked Test ECE | Locked Test Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw CatBoost** | $0.650231$ | $0.004802$ | $0.099026$ | $0.649503$ | $0.003198$ | $0.095340$ |
| **Platt Scaling** | $0.650231$ | $0.003028$ | $0.099019$ | $0.649503$ | $0.005411$ | $0.095358$ |
| **Isotonic Regression (Selected)** | $0.653780$ (binned) / $0.650764$ (raw) | $0.000000$ (in-sample) | $0.098606$ | $0.647526$ | $0.006179$ | $0.095553$ |
| **Beta Calibration** | $0.650231$ | $0.004001$ | $0.099037$ | $0.649503$ | $0.006230$ | $0.095348$ |
| **5-Fold Cross-Fitted Isotonic on `val.csv`** | $0.650764$ | $0.006229$ (out-of-fold) | — | *(Test split untouched)* | *(Test split untouched)* | — |

