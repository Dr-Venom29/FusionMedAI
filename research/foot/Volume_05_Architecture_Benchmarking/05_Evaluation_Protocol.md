# 05 Evaluation Protocol — Phase 10.5

## Evaluation Directives

1. **Untouched Test Set Policy**: The frozen test set (1,006 images / 181 source groups) remains completely untouched during model training and hyperparameter decisions.
2. **Single Evaluation Execution**: Best validation-loss checkpoint (`best_model.pt`) evaluated exactly once against the frozen test partition per candidate model.
3. **Primary Benchmark Metric**: **Macro F1-Score**
4. **Statistical Comparison Policy**: Compute 95% Bootstrap Confidence Intervals (1,000 resamples, seed=42) for Macro F1 and Balanced Accuracy to prevent declaring superiority based on sampling noise alone.
5. **Efficiency Metrics**: Track parameter count (M), model size (MB), average batch latency (ms), and throughput (img/s).
6. **Confusion Matrix Comparison**: Save dedicated $4 \times 4$ confusion matrix plot for each architecture (`datasets/foot/metadata/benchmark/<model_name>_confusion_matrix.png`).
7. **Quality-Stratified Error Logging**: Prediction-level misclassification error logs saved to `error_analysis.csv` per architecture.
