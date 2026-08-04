# 8. Conclusion

Step 7 successfully integrated a post-hoc calibration pipeline for the Retina Module.

Using Temperature Scaling, we significantly improved the reliability of our neural network's predictions. The calibrated model reduces overconfidence, achieves lower Expected Calibration Error (ECE) and Brier scores, and produces reliability diagrams that closely track the ideal identity function. Crucially, this was achieved without sacrificing classification accuracy or requiring computationally expensive retraining. The calibrated probabilities generated during this phase will serve as the confidence estimates for the uncertainty estimation pipeline implemented in Volume 08.
