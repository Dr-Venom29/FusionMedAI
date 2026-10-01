"""Hyperparameter Optimization Module for Architecture Benchmarking.

Executes bounded validation-only hyperparameter searches (seed=42).
Strict constraint: test.csv is never used during optimization.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import optuna

from src.clinical.benchmarking.catboost import CatBoostModel
from src.clinical.benchmarking.tabnet import TabNetModel

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_catboost_hpo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 15,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Perform bounded HPO for CatBoost using validation PR-AUC as optimization target."""
    trial_records = []

    def objective(trial: optuna.Trial) -> float:
        depth = trial.suggest_int("depth", 4, 8)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
        iterations = trial.suggest_int("iterations", 150, 400, step=50)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)
        subsample = trial.suggest_float("subsample", 0.6, 0.9)

        model = CatBoostModel(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            subsample=subsample,
            random_state=random_state,
            early_stopping_rounds=25,
            verbose=0,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        y_prob_val = model.predict_proba(X_val)[:, 1]

        val_pr_auc = float(average_precision_score(y_val, y_prob_val))
        val_roc_auc = float(roc_auc_score(y_val, y_prob_val))

        trial_records.append({
            "model": "catboost",
            "trial_number": trial.number,
            "depth": depth,
            "learning_rate": round(learning_rate, 4),
            "iterations": iterations,
            "l2_leaf_reg": round(l2_leaf_reg, 3),
            "subsample": round(subsample, 3),
            "val_pr_auc": round(val_pr_auc, 4),
            "val_roc_auc": round(val_roc_auc, 4),
        })
        return val_pr_auc

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["random_state"] = random_state
    best_params["early_stopping_rounds"] = 25
    best_params["verbose"] = 0

    df_trials = pd.DataFrame(trial_records)
    return best_params, df_trials


def run_tabnet_hpo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 10,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Perform bounded HPO for TabNet using validation PR-AUC as optimization target."""
    trial_records = []

    def objective(trial: optuna.Trial) -> float:
        n_d = trial.suggest_categorical("n_d", [8, 16, 24])
        n_a = n_d  # Keep n_a == n_d per best practices
        n_steps = trial.suggest_int("n_steps", 3, 5)
        gamma = trial.suggest_float("gamma", 1.2, 1.8)
        lambda_sparse = trial.suggest_float("lambda_sparse", 1e-4, 1e-2, log=True)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.05, log=True)

        model = TabNetModel(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            learning_rate=learning_rate,
            max_epochs=40,
            patience=8,
            batch_size=1024,
            virtual_batch_size=128,
            random_state=random_state,
            verbose=0,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        y_prob_val = model.predict_proba(X_val)[:, 1]

        val_pr_auc = float(average_precision_score(y_val, y_prob_val))
        val_roc_auc = float(roc_auc_score(y_val, y_prob_val))

        trial_records.append({
            "model": "tabnet",
            "trial_number": trial.number,
            "n_d": n_d,
            "n_a": n_a,
            "n_steps": n_steps,
            "gamma": round(gamma, 3),
            "lambda_sparse": round(lambda_sparse, 5),
            "learning_rate": round(learning_rate, 4),
            "val_pr_auc": round(val_pr_auc, 4),
            "val_roc_auc": round(val_roc_auc, 4),
        })
        return val_pr_auc

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["n_a"] = best_params["n_d"]
    best_params["max_epochs"] = 40
    best_params["patience"] = 8
    best_params["batch_size"] = 1024
    best_params["virtual_batch_size"] = 128
    best_params["random_state"] = random_state
    best_params["verbose"] = 0

    df_trials = pd.DataFrame(trial_records)
    return best_params, df_trials


def run_hpo_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """Run validation HPO for both CatBoost and TabNet architectures."""
    print("  -> Starting CatBoost validation HPO search (15 trials)...")
    cb_best, cb_df = run_catboost_hpo(X_train, y_train, X_val, y_val, n_trials=15, random_state=random_state)
    
    print("  -> Starting TabNet validation HPO search (10 trials)...")
    tn_best, tn_df = run_tabnet_hpo(X_train, y_train, X_val, y_val, n_trials=10, random_state=random_state)

    df_all_hpo = pd.concat([cb_df, tn_df], ignore_index=True)
    best_configs = {
        "catboost_tuned": cb_best,
        "tabnet_tuned": tn_best,
    }
    return best_configs, df_all_hpo


def run_hpo_standalone(
    model_name: str = "catboost",
    n_trials: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Execute standalone HPO search on validation set."""
    import argparse
    from pathlib import Path
    from src.clinical.modeling.preprocessing import ClinicalPreprocessor

    print("=" * 70)
    print(f"FusionMedAI: Standalone Bounded Validation HPO ({model_name.upper()})")
    print("=" * 70)

    repo_root = Path(__file__).resolve().parents[3]
    splits_dir = repo_root / "datasets" / "clinical" / "processed" / "splits"
    modeling_base = repo_root / "datasets" / "clinical" / "metadata" / "modeling"
    modeling_base.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Loading canonical splits & fitting locked preprocessor...")
    df_train = pd.read_csv(splits_dir / "train.csv")
    df_val = pd.read_csv(splits_dir / "val.csv")

    preprocessor = ClinicalPreprocessor(scale_numerical=True)
    preprocessor.fit(df_train)
    X_train, y_train, _ = preprocessor.transform(df_train)
    X_val, y_val, _ = preprocessor.transform(df_val)

    hpo_dfs = []
    if model_name in ["catboost", "all"]:
        print(f"\n[2/3] Running CatBoost HPO ({n_trials} trials)...")
        cb_best, cb_df = run_catboost_hpo(X_train, y_train, X_val, y_val, n_trials=n_trials, random_state=random_state)
        print(f"  -> Best CatBoost Params: {cb_best}")
        hpo_dfs.append(cb_df)

    if model_name in ["tabnet", "all"]:
        tn_trials = max(5, n_trials if model_name == "tabnet" else 10)
        print(f"\n[2/3] Running TabNet HPO ({tn_trials} trials)...")
        tn_best, tn_df = run_tabnet_hpo(X_train, y_train, X_val, y_val, n_trials=tn_trials, random_state=random_state)
        print(f"  -> Best TabNet Params: {tn_best}")
        hpo_dfs.append(tn_df)

    df_out = pd.concat(hpo_dfs, ignore_index=True) if hpo_dfs else pd.DataFrame()
    out_path = modeling_base / "hyperparameter_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n[3/3] HPO trials saved to: {out_path.relative_to(repo_root)}")
    return df_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standalone Bounded Validation HPO Runner")
    parser.add_argument(
        "--model",
        type=str,
        default="catboost",
        choices=["catboost", "tabnet", "all"],
        help="Target architecture for HPO search (default: catboost)",
    )
    parser.add_argument("--trials", type=int, default=15, help="Number of trials (default: 15)")
    args = parser.parse_args()

    run_hpo_standalone(
        model_name=args.model,
        n_trials=args.trials,
    )
