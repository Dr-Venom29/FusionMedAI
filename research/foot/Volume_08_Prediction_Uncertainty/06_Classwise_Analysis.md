# Chapter 06 — Classwise Uncertainty Analysis

## 1. Classwise Uncertainty Breakdown

To determine whether prediction uncertainty varies across Wagner grades, uncertainty statistics were disaggregated across ground truth classes (Grade 1, Grade 2, Grade 3, Grade 4) on the held-out test split ($N=1,006$).

---

## 2. Empirical Classwise Tabulation

| Grade | Ground Truth Class | Samples | Accuracy | Mean Confidence | Mean Entropy (nats) | Mean Variance | Mean MI (nats) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **G1** | Grade 1 (Superficial) | 237 | 0.5485 | 0.6130 | 0.9296 | 0.000475 | 0.0030 |
| **G2** | Grade 2 (Deep Ulcer) | 246 | 0.7480 | 0.6675 | 0.8253 | 0.000479 | 0.0032 |
| **G3** | Grade 3 (Deep Ulcer + Osteomyelitis/Abscess) | 280 | 0.6929 | 0.6532 | 0.8583 | 0.000468 | 0.0031 |
| **G4** | Grade 4 (Gangrene) | 243 | 0.7160 | 0.7392 | 0.6764 | 0.000321 | 0.0024 |

---

## 3. Analysis & Key Insights

1. **Grade 4 (Gangrene) Characteristics**: Grade 4 exhibited high classification accuracy (**71.60%**) and the lowest mean predictive entropy (**0.6764 nats**) and variance (**0.000321**) in this test set.
2. **Grade 1, 2, and 3 Characteristics**: Grade 1, Grade 2, and Grade 3 exhibited higher mean predictive entropy (0.8253 to 0.9296 nats) and lower classification accuracy, indicating greater prediction uncertainty for these classes in this evaluation.
