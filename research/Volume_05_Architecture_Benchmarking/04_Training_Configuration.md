# Chapter 4: Training Configuration

## 4.1 Weighted Cross-Entropy
Due to the severe class imbalance in the APTOS dataset, standard cross-entropy often causes models to ignore minority classes (e.g., Severe DR). We calculate class frequencies dynamically and apply inverse class weighting to ensure balanced learning.

## 4.2 Optimization & Scheduling
- **AdamW**: Selected for robust weight decay regularization, preventing overfitting better than standard Adam.
- **CosineAnnealingLR**: Smoothly decays the learning rate, allowing the model to settle into local minima towards the end of training.

## 4.3 Mixed Precision (AMP)
Automatic Mixed Precision (AMP) is utilized to train models using FP16 where computationally safe, reducing VRAM footprint and accelerating matrix multiplications on modern GPUs.

## 4.4 Early Stopping & Checkpoint Strategy
To prevent overfitting and standardize the comparison across models that converge at different rates, training runs up to a maximum of 50 epochs but terminates early if validation QWK does not improve for 10 consecutive epochs. Only the `best_model.pt` (based on validation QWK) is saved.
