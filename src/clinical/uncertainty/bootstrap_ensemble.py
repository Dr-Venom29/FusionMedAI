"""
Bootstrap Ensemble for CatBoost Tabular Models (Clinical Phase C8).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


class BootstrapCatBoostEnsemble:
    """
    Bootstrap ensemble of CatBoost classifiers to quantify predictive uncertainty.
    
    Each ensemble member is trained on a distinct bootstrap resample of the training partition
    (with replacement) using the strictly locked HPO candidate hyperparameters.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        catboost_params: Optional[Dict[str, Any]] = None,
        random_seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.models: List[CatBoostClassifier] = []
        self.feature_names: Optional[List[str]] = None

        if catboost_params is None:
            # Locked Phase C5/C6/C7 HPO candidate hyperparameters
            self.catboost_params = {
                "depth": 4,
                "learning_rate": 0.1383,
                "iterations": 350,
                "l2_leaf_reg": 2.911,
                "subsample": 0.655,
                "loss_function": "Logloss",
                "eval_metric": "Logloss",
                "verbose": False,
                "thread_count": -1,
            }
        else:
            self.catboost_params = catboost_params

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        train_dir: Optional[Path] = None,
    ) -> "BootstrapCatBoostEnsemble":
        """
        Fit M bootstrap ensemble models on resampled training sets.
        """
        self.feature_names = feature_names
        n_samples = X_train.shape[0]
        rng = np.random.RandomState(self.random_seed)

        self.models = []
        for m in range(self.n_estimators):
            # Sample with replacement
            bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_b = X_train[bootstrap_indices]
            y_b = y_train[bootstrap_indices]

            model_params = dict(self.catboost_params)
            model_params["random_seed"] = self.random_seed + m + 1
            if train_dir is not None:
                m_dir = train_dir / f"member_{m:03d}"
                m_dir.mkdir(parents=True, exist_ok=True)
                model_params["train_dir"] = str(m_dir)

            model = CatBoostClassifier(**model_params)
            model.fit(X_b, y_b, verbose=False)
            self.models.append(model)

        return self

    def predict_distribution(
        self,
        X: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Generate stochastic prediction distributions across all ensemble members.
        
        Returns:
            all_probs: (N, M) matrix of predictions from each member
            mean_prob: (N,) ensemble mean probability
            std_prob: (N,) predictive standard deviation (uncertainty)
            var_prob: (N,) predictive variance
            q025: (N,) 2.5th percentile of predictions
            q975: (N,) 97.5th percentile of predictions
            entropy: (N,) binary predictive entropy
        """
        if not self.models:
            raise ValueError("Ensemble has not been fitted yet.")

        n_samples = X.shape[0]
        n_members = len(self.models)
        all_probs = np.zeros((n_samples, n_members), dtype=np.float64)

        for m, model in enumerate(self.models):
            all_probs[:, m] = model.predict_proba(X)[:, 1]

        mean_prob = np.mean(all_probs, axis=1)
        var_prob = np.var(all_probs, axis=1, ddof=1 if n_members > 1 else 0)
        std_prob = np.sqrt(var_prob)
        q025 = np.percentile(all_probs, 2.5, axis=1)
        q975 = np.percentile(all_probs, 97.5, axis=1)

        # Predictive binary entropy (aleatoric proxy on mean probability)
        p_clipped = np.clip(mean_prob, 1e-12, 1.0 - 1e-12)
        entropy = - (p_clipped * np.log2(p_clipped) + (1.0 - p_clipped) * np.log2(1.0 - p_clipped))

        return {
            "all_probs": all_probs,
            "mean_prob": mean_prob,
            "std_prob": std_prob,
            "var_prob": var_prob,
            "q025": q025,
            "q975": q975,
            "entropy": entropy,
        }
