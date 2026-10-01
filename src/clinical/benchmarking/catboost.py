import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

REPO_ROOT = Path(__file__).resolve().parents[3]
script_dir = str(Path(__file__).resolve().parent)
while script_dir in sys.path:
    sys.path.remove(script_dir)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clinical.modeling.models import BaseClinicalModel
from src.clinical.modeling.preprocessing import ClinicalPreprocessor
from src.clinical.modeling.metrics import (
    compute_classification_metrics,
    compute_calibration_metrics,
)


class CatBoostModel(BaseClinicalModel):
    """CatBoost Classifier implementing the unified BaseClinicalModel interface."""

    def __init__(
        self,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        subsample: float = 0.8,
        random_state: int = 42,
        early_stopping_rounds: Optional[int] = 30,
        verbose: int = 0,
        thread_count: int = -1,
        train_dir: Optional[str] = None,
    ):
        if train_dir is not None:
            target_train_dir = Path(train_dir)
        elif Path("/kaggle/working").exists():
            target_train_dir = Path("/kaggle/working/catboost_info")
        else:
            target_train_dir = (
                REPO_ROOT
                / "datasets"
                / "clinical"
                / "metadata"
                / "modeling"
                / "catboost_info"
            )
        target_train_dir.mkdir(parents=True, exist_ok=True)
        target_train_dir = str(target_train_dir)
        config = {
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": l2_leaf_reg,
            "subsample": subsample,
            "random_state": random_state,
            "early_stopping_rounds": early_stopping_rounds,
            "verbose": verbose,
            "thread_count": thread_count,
            "train_dir": target_train_dir,
        }
        super().__init__(model_name="catboost", config=config)
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            subsample=subsample,
            random_seed=random_state,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            thread_count=thread_count,
            eval_metric="Logloss",
            train_dir=target_train_dir,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "CatBoostModel":
        """Fit CatBoost model with optional validation eval_set for early stopping."""
        if eval_set is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=self.verbose,
                use_best_model=True,
            )
        else:
            self.model.fit(X_train, y_train, verbose=self.verbose)
        self.is_fitted = True
        return self


def run_catboost_standalone(
    iterations: int = 300,
    depth: int = 6,
    learning_rate: float = 0.05,
    l2_leaf_reg: float = 3.0,
    subsample: float = 0.8,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute standalone CatBoost training and evaluation."""
    print("=" * 70)
    print("FusionMedAI: Standalone CatBoost Classifier Evaluation")
    print("=" * 70)

    splits_dir = REPO_ROOT / "datasets" / "clinical" / "processed" / "splits"
    modeling_base = REPO_ROOT / "datasets" / "clinical" / "metadata" / "modeling"
    configs_dir = modeling_base / "model_configs"
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

    print(f"\n[2/3] Training CatBoost (iterations={iterations}, depth={depth}, lr={learning_rate})...")
    model = CatBoostModel(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        subsample=subsample,
        random_state=random_state,
        early_stopping_rounds=30,
        verbose=0,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    train_time = time.perf_counter() - t0

    print(f"\n[3/3] Evaluating Performance (Train Time: {train_time:.2f}s)...")
    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    val_m50 = compute_classification_metrics(y_val, y_prob_val, threshold=0.50, model_name="catboost")
    val_m20 = compute_classification_metrics(y_val, y_prob_val, threshold=0.20, model_name="catboost")
    val_cal = compute_calibration_metrics(y_val, y_prob_val, n_bins=10, model_name="catboost")

    test_m50 = compute_classification_metrics(y_test, y_prob_test, threshold=0.50, model_name="catboost")
    test_m20 = compute_classification_metrics(y_test, y_prob_test, threshold=0.20, model_name="catboost")
    test_cal = compute_calibration_metrics(y_test, y_prob_test, n_bins=10, model_name="catboost")

    print(f"  [Validation] ROC-AUC: {val_m50['roc_auc']:.4f} | PR-AUC: {val_m50['pr_auc']:.4f} | Brier: {val_m50['brier_score']:.4f} | ECE: {val_cal['expected_calibration_error']:.4f}")
    print(f"  [Test]       ROC-AUC: {test_m50['roc_auc']:.4f} | PR-AUC: {test_m50['pr_auc']:.4f} | Brier: {test_m50['brier_score']:.4f} | ECE: {test_cal['expected_calibration_error']:.4f}")
    print(f"  [Clinical th=0.20] Test Sens: {test_m20['sensitivity']:.4f} | Test Spec: {test_m20['specificity']:.4f} | Test PPV: {test_m20['ppv']:.4f}")

    cfg_path = configs_dir / "catboost_default_config.json"
    with open(cfg_path, "w") as f:
        json.dump(model.get_config(), f, indent=2)
    print(f"\n  -> Config saved to: {cfg_path.relative_to(REPO_ROOT)}")

    return {
        "model": "catboost",
        "train_time": train_time,
        "val_metrics": val_m50,
        "test_metrics": test_m50,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone CatBoost Model Runner")
    parser.add_argument("--iterations", type=int, default=300, help="Number of boosting iterations (default: 300)")
    parser.add_argument("--depth", type=int, default=6, help="Tree depth (default: 6)")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate (default: 0.05)")
    args = parser.parse_args()

    run_catboost_standalone(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.lr,
    )
