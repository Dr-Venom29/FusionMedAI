# 3. Evaluation Metrics

To quantitatively assess the calibration quality of our Retina module, we track several core metrics:

## Expected Calibration Error (ECE)
ECE measures the expected difference between the model's confidence and its actual accuracy. Predictions are partitioned into `M` equally spaced bins. ECE is the weighted average of the absolute difference between the accuracy and confidence of each bin.

## Maximum Calibration Error (MCE)
MCE measures the worst-case deviation between confidence and accuracy across all bins. It is particularly relevant for high-stakes medical applications where bounding the maximum error is critical.

## Negative Log-Likelihood (NLL)
NLL is the standard cross-entropy loss evaluated on the validation set. It is a strictly proper scoring rule, meaning it is minimized only when the predicted probabilities exactly match the true distribution.

## Brier Score
The multiclass Brier score calculates the mean squared difference between the predicted probabilities and the one-hot encoded true labels. Lower Brier scores indicate better calibration and accuracy.
