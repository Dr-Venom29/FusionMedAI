"""TabNet Classifier Model Implementation for Tabular Architecture Benchmarking.

Neural architecture utilizing sequential attention for tabular decision-making.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from src.clinical.modeling.models import BaseClinicalModel


class TabNetModel(BaseClinicalModel):
    """TabNet Classifier implementing the unified BaseClinicalModel interface."""

    def __init__(
        self,
        n_d: int = 16,
        n_a: int = 16,
        n_steps: int = 3,
        gamma: float = 1.3,
        lambda_sparse: float = 1e-3,
        learning_rate: float = 2e-2,
        max_epochs: int = 50,
        patience: int = 10,
        batch_size: int = 1024,
        virtual_batch_size: int = 128,
        random_state: int = 42,
        verbose: int = 0,
    ):
        config = {
            "n_d": n_d,
            "n_a": n_a,
            "n_steps": n_steps,
            "gamma": gamma,
            "lambda_sparse": lambda_sparse,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "patience": patience,
            "batch_size": batch_size,
            "virtual_batch_size": virtual_batch_size,
            "random_state": random_state,
            "verbose": verbose,
        }
        super().__init__(model_name="tabnet", config=config)
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        
        # Set torch seed
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=learning_rate),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params=dict(step_size=10, gamma=0.5),
            mask_type="entmax",
            seed=random_state,
            verbose=verbose,
            device_name=device_name,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "TabNetModel":
        """Fit TabNet model with optional validation eval_set for early stopping."""
        X_tr = X_train.astype(np.float32)
        y_tr = y_train.astype(np.int64)

        eval_data = []
        eval_names = []
        if eval_set is not None:
            eval_data.append((eval_set[0].astype(np.float32), eval_set[1].astype(np.int64)))
            eval_names.append("val")

        self.model.fit(
            X_train=X_tr,
            y_train=y_tr,
            eval_set=eval_data if eval_data else None,
            eval_name=eval_names if eval_names else None,
            eval_metric=["logloss"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            num_workers=0,
            drop_last=False,
        )
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities [P(y=0), P(y=1)]."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("TabNetModel must be fitted before predict_proba.")
        return self.model.predict_proba(X.astype(np.float32))


def run_tabnet_standalone(
    max_epochs: int = 40,
    patience: int = 8,
    lr: float = 0.02,
    batch_size: int = 1024,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute standalone TabNet training and evaluation."""
    import time
    import json
    from pathlib import Path
    import pandas as pd
    from src.clinical.modeling.preprocessing import ClinicalPreprocessor
    from src.clinical.modeling.metrics import (
        compute_classification_metrics,
        compute_calibration_metrics,
    )

    print("=" * 70)
    print("FusionMedAI: Standalone TabNet Classifier Evaluation")
    print("=" * 70)

    repo_root = Path(__file__).resolve().parents[3]
    splits_dir = repo_root / "datasets" / "clinical" / "processed" / "splits"
    c5_base = repo_root / "datasets" / "clinical" / "metadata" / "modeling" / "c5"
    configs_dir = c5_base / "model_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Loading canonical splits & fitting locked preprocessor...")
    df_train = pd.read_csv(splits_dir / "train.csv")
    df_val = pd.read_csv(splits_dir / "val.csv")
    df_test = pd.read_csv(splits_dir / "test.csv")

    preprocessor = ClinicalPreprocessor(scale_numerical=True)
    preprocessor.fit(df_train)
    X_train, y_train, _ = preprocessor.transform(df_train)
    X_val, y_val, _ = preprocessor.transform(df_val)
    X_test, y_test, _ = preprocessor.transform(df_test)

    print(f"\n[2/3] Training TabNet (max_epochs={max_epochs}, patience={patience}, lr={lr})...")
    model = TabNetModel(
        n_d=16,
        n_a=16,
        n_steps=3,
        gamma=1.3,
        lambda_sparse=1e-3,
        learning_rate=lr,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=128,
        random_state=random_state,
        verbose=0,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    train_time = time.perf_counter() - t0

    print(f"\n[3/3] Evaluating Performance (Train Time: {train_time:.2f}s)...")
    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    val_m50 = compute_classification_metrics(y_val, y_prob_val, threshold=0.50, model_name="tabnet")
    val_m20 = compute_classification_metrics(y_val, y_prob_val, threshold=0.20, model_name="tabnet")
    val_cal = compute_calibration_metrics(y_val, y_prob_val, n_bins=10, model_name="tabnet")

    test_m50 = compute_classification_metrics(y_test, y_prob_test, threshold=0.50, model_name="tabnet")
    test_m20 = compute_classification_metrics(y_test, y_prob_test, threshold=0.20, model_name="tabnet")
    test_cal = compute_calibration_metrics(y_test, y_prob_test, n_bins=10, model_name="tabnet")

    print(f"  [Validation] ROC-AUC: {val_m50['roc_auc']:.4f} | PR-AUC: {val_m50['pr_auc']:.4f} | Brier: {val_m50['brier_score']:.4f} | ECE: {val_cal['expected_calibration_error']:.4f}")
    print(f"  [Test]       ROC-AUC: {test_m50['roc_auc']:.4f} | PR-AUC: {test_m50['pr_auc']:.4f} | Brier: {test_m50['brier_score']:.4f} | ECE: {test_cal['expected_calibration_error']:.4f}")
    print(f"  [Clinical th=0.20] Test Sens: {test_m20['sensitivity']:.4f} | Test Spec: {test_m20['specificity']:.4f} | Test PPV: {test_m20['ppv']:.4f}")

    cfg_path = configs_dir / "tabnet_default_config.json"
    with open(cfg_path, "w") as f:
        json.dump(model.get_config(), f, indent=2)
    print(f"\n  -> Config saved to: {cfg_path.relative_to(repo_root)}")

    return {
        "model": "tabnet",
        "train_time": train_time,
        "val_metrics": val_m50,
        "test_metrics": test_m50,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standalone TabNet Model Runner")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum epochs (default: 40)")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (default: 8)")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate (default: 0.02)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size (default: 1024)")
    args = parser.parse_args()

    run_tabnet_standalone(
        max_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
    )
