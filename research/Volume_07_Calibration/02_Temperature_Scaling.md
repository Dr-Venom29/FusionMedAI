# 2. Temperature Scaling

Temperature Scaling is a simple yet highly effective post-hoc calibration method introduced by Guo et al. (2017).

## Mechanism

Given the logit vector `z_i` for input `i`, the standard predicted probability is `q_i = softmax(z_i)`.
Temperature scaling introduces a single scalar parameter `T > 0` and modifies the probability to:
`q_i = softmax(z_i / T)`

- If `T > 1`, the probability distribution softens, effectively reducing overconfidence.
- If `T < 1`, the distribution becomes sharper.
- If `T = 1`, the original probabilities are recovered.

Crucially, because `T` is a single scalar that scales all logits equally, the argmax (the predicted class) remains unchanged. Therefore, Temperature Scaling perfectly preserves the model's classification accuracy.

## Optimization

The parameter `T` is optimized on a held-out validation set by minimizing the Negative Log-Likelihood (NLL). We utilize the L-BFGS optimizer for fast convergence.
