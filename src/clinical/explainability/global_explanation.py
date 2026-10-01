"""Global SHAP feature importance and directionality analysis."""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clinical.explainability.group_analysis import get_clinical_feature_group


def compute_global_importance(
    shap_values_test: np.ndarray,
    shap_values_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Compute global mean |SHAP| values, rankings, and directionality correlations."""
    n_features = len(feature_names)
    mean_abs_test = np.mean(np.abs(shap_values_test), axis=0)
    mean_abs_val = np.mean(np.abs(shap_values_val), axis=0)

    # Directionality correlation between feature value and SHAP contribution
    direction_corrs = []
    for j in range(n_features):
        f_vals = X_test[:, j]
        s_vals = shap_values_test[:, j]
        if np.std(f_vals) > 1e-8 and np.std(s_vals) > 1e-8:
            corr = float(np.corrcoef(f_vals, s_vals)[0, 1])
        else:
            corr = 0.0
        direction_corrs.append(corr)

    df = pd.DataFrame({
        "feature": feature_names,
        "feature_group": [get_clinical_feature_group(f) for f in feature_names],
        "mean_abs_shap_test": mean_abs_test,
        "mean_abs_shap_val": mean_abs_val,
        "directionality_corr": direction_corrs,
    })

    df = df.sort_values(by="mean_abs_shap_test", ascending=False).reset_index(drop=True)
    df["rank_test"] = np.arange(1, n_features + 1)
    
    total_shap = df["mean_abs_shap_test"].sum()
    df["relative_contribution_pct"] = (df["mean_abs_shap_test"] / total_shap) * 100.0
    df["cumulative_contribution_pct"] = df["relative_contribution_pct"].cumsum()

    def interpret_direction(corr: float) -> str:
        if abs(corr) < 0.05:
            return "Non-linear / Context-dependent"
        elif corr > 0:
            return "Positive (Higher feature value -> Higher predicted risk)"
        else:
            return "Negative (Higher feature value -> Lower predicted risk)"

    df["effect_direction"] = df["directionality_corr"].apply(interpret_direction)
    return df


def plot_global_bar(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Plot horizontal bar chart of top N global SHAP features."""
    top_df = importance_df.head(top_n).sort_values("mean_abs_shap_test", ascending=True)
    plt.figure(figsize=(10, 8), dpi=300)
    colors = sns.color_palette("mako", len(top_df))
    plt.barh(top_df["feature"], top_df["mean_abs_shap_test"], color=colors, edgecolor="black", alpha=0.85)
    plt.xlabel("Mean Absolute SHAP Value (Test Partition)", fontsize=12, fontweight="bold")
    plt.ylabel("Clinical Feature (D=119)", fontsize=12, fontweight="bold")
    plt.title(f"Top {top_n} Global Feature Importances (CatBoost HPO Frozen Model)", fontsize=14, fontweight="bold", pad=15)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
    sample_size: int = 2000,
) -> None:
    """Generate SHAP summary beeswarm plot for top N features."""
    top_features = importance_df.head(top_n)["feature"].tolist()
    top_indices = [feature_names.index(f) for f in top_features]

    np.random.seed(42)
    sample_n = min(sample_size, len(X))
    sample_idx = np.random.choice(len(X), sample_n, replace=False)

    plt.figure(figsize=(11, 9), dpi=300)
    y_pos = np.arange(len(top_indices))[::-1]

    for row_idx, feat_idx in enumerate(top_indices):
        f_vals = X[sample_idx, feat_idx]
        s_vals = shap_values[sample_idx, feat_idx]

        f_min, f_max = np.percentile(f_vals, 1), np.percentile(f_vals, 99)
        norm_vals = np.clip((f_vals - f_min) / (f_max - f_min + 1e-8), 0, 1)

        jitter = np.random.normal(0, 0.08, size=sample_n)
        sc = plt.scatter(s_vals, y_pos[row_idx] + jitter, c=norm_vals, cmap="coolwarm", s=12, alpha=0.6, rasterized=True)

    plt.yticks(y_pos, [feature_names[i] for i in top_indices], fontsize=10)
    plt.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    plt.xlabel("SHAP Value (Impact on Model Readmission Log-Odds)", fontsize=12, fontweight="bold")
    plt.title(f"SHAP Summary Beeswarm Plot: Top {top_n} Features (Test Set)", fontsize=14, fontweight="bold", pad=15)
    cbar = plt.colorbar(sc, orientation="vertical", pad=0.02)
    cbar.set_label("Feature Value (Low -> High)", fontsize=10)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    plt.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_dependence(
    feature_name: str,
    shap_values: np.ndarray,
    X: np.ndarray,
    probs: np.ndarray,
    feature_names: List[str],
    output_path: Path,
) -> None:
    """Generate SHAP dependence plot for a single feature."""
    if feature_name not in feature_names:
        return
    f_idx = feature_names.index(feature_name)
    f_vals = X[:, f_idx]
    s_vals = shap_values[:, f_idx]

    plt.figure(figsize=(8, 5), dpi=300)
    plt.scatter(f_vals, s_vals, c=probs, cmap="viridis", s=10, alpha=0.4, rasterized=True)
    plt.axhline(0, color="red", linestyle="--", alpha=0.7)
    plt.xlabel(f"Standardized Value: {feature_name}", fontsize=11, fontweight="bold")
    plt.ylabel("SHAP Contribution to Log-Odds", fontsize=11, fontweight="bold")
    plt.title(f"SHAP Dependence Plot: {feature_name}", fontsize=13, fontweight="bold", pad=12)
    cbar = plt.colorbar()
    cbar.set_label("Predicted Readmission Risk", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
