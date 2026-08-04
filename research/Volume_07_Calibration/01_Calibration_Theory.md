# 1. Calibration Theory

Medical AI systems, particularly deep neural networks, are prone to producing overconfident predictions. While a model might be highly accurate, the predicted probability (softmax output) often does not reflect the true likelihood of correctness.

## Why Calibration Matters

A well-calibrated model ensures that if it predicts a class with 90% confidence, it should be correct 90% of the time. In clinical settings like Diabetic Retinopathy screening, misinterpreting an overconfident incorrect prediction can lead to severe diagnostic errors. Calibration bridges the gap between model accuracy and reliability.
