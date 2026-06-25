# Methodology

## Dataset Alignment Statement

### Important Research Assumption

FusionMedAI does **NOT** perform patient-level multimodal learning.

Datasets are independent and originate from different patient populations.

### Decision-level Fusion & ACARA-U

ACARA-U performs **decision-level aggregation** using:

- Risk Scores
- Confidence Scores
- Reliability Scores
- Uncertainty Scores

No raw patient features are fused across datasets.
