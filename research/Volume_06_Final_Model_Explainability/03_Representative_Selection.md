# 03 Representative Selection

Qualitative analysis of Explainable AI is highly susceptible to cherry-picking, where researchers only present the model's most successful predictions. FusionMedAI combats this by implementing a rigorous, reproducible diversified sampling algorithm in `src/xai/selector.py`.

## Selection Criteria

The pipeline extracts exactly 25 representative images based on the following diverse scenarios:
1.  **Most Confident Correct**: Where the model is certain and accurate.
2.  **Least Confident Correct**: Where the model is accurate but highly uncertain.
3.  **Most Confident Incorrect**: Cases of severe overconfidence leading to error.
4.  **Least Confident Incorrect**: Complete model confusion.
5.  **Highest/Lowest Entropy**: Selecting the boundaries of the model's overall probability distribution dispersion.
6.  **Per-Grade Representatives**: At least one correct sample from every Diabetic Retinopathy grade (0-4).
7.  **Per-Grade Failures**: At least one incorrect sample from every grade.
8.  **Severe Confusion**: A sample where the model's prediction differs from the ground truth by 2 or more grades (e.g., predicting Grade 1 when the truth is Grade 4).

## Diversified Percentile Sampling

Rather than deterministically grabbing the top `N` cases for each criteria, the sampling algorithm is designed to be statistically representative. For multi-sample requests, the selector draws from:
*   The absolute **Top** case.
*   The **Median** case within that criteria.
*   A **Random** case sampled from the top 10th percentile.

This ensures the generated gallery exposes true average model behavior alongside the edge cases, providing an honest assessment of the retinal module's performance.

## Execution Result

Experimental execution selected exactly 25 representative cases spanning all predefined sampling categories.
