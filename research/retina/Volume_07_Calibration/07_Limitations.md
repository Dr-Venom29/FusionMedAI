# 7. Limitations

While Temperature Scaling is effective, it has a few known limitations:

1. **Shift in Distribution**: Post-hoc calibration relies on a validation set. If the real-world deployment data differs significantly from this validation set (covariate shift or out-of-distribution data), the model may become miscalibrated again.
2. **Class-Agnostic**: A single scalar $T$ is applied to all classes identically. If the model exhibits different confidence profiles for different diseases, Temperature Scaling cannot correct this disparity.
3. **No Correction for Base Errors**: Calibration fixes the probabilities of the predictions, but it does not fix incorrect predictions. The underlying model accuracy sets a hard ceiling on performance.
4. **Worst-case Metrics**: Temperature Scaling minimizes NLL rather than Maximum Calibration Error. Consequently, localized confidence bins with very few samples may still exhibit large MCE values despite substantial improvements in global calibration.
