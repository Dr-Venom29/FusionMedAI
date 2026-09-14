# 08 Error Analysis — Phase 10.4

## Error Breakdown & Failure Modes

- **Total Test Errors**: 365 errors out of 1,006 test samples (**36.28% Error Rate**).
- **Primary Failure Mode**: Grade 2 vs Grade 3 misclassifications (65 Grade 2 → Grade 3; 19 Grade 3 → Grade 2).
- **Secondary Failure Mode**: Grade 3 vs Grade 4 misclassifications (60 Grade 3 → Grade 4).
- **Archived Audit File**: Full sample-level predictions and confidence scores saved to `experiments/foot/baseline_resnet50_unweighted/error_analysis.csv`.
