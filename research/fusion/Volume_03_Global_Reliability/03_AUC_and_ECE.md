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

$$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} \cdot \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

Where:
- $|B_m|$ is the number of samples falling in quantile bin $m$.
- $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \text{acc}_i$ is the empirical empirical accuracy in bin $m$.
- $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \text{conf}_i$ is the average predicted confidence in bin $m$.
