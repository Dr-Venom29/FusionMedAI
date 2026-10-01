"""Computational Complexity & Operational Cost Audit.

Measures:
- Training execution time (seconds)
- Inference latency (ms per 1,000 samples)
- Inference throughput (samples / sec)
- Memory / Serialized model size (KB on disk)
- Parameter count (where well-defined, e.g. neural models)
"""

import time
import os
import sys
import tempfile
import joblib
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def audit_model_complexity(
    model_name: str,
    model_obj: Any,
    fit_time_seconds: float,
    X_sample: np.ndarray,
) -> Dict[str, Any]:
    """Profile inference latency, throughput, disk size, and parameter count for a fitted model."""
    n_samples = len(X_sample)
    
    # Measure inference latency over repeated trials
    n_runs = 5
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model_obj.predict_proba(X_sample)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    mean_inference_time = float(np.mean(latencies))
    # Latency per 1,000 encounters in milliseconds
    latency_per_1k_ms = round((mean_inference_time / n_samples) * 1000 * 1000, 3)
    throughput_samples_per_sec = round(n_samples / mean_inference_time, 1)

    # Estimate serialized model size on disk
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        if hasattr(model_obj, "model"):
            joblib.dump(model_obj.model, tmp_path)
        else:
            joblib.dump(model_obj, tmp_path)
        size_bytes = os.path.getsize(tmp_path)
        size_kb = round(size_bytes / 1024, 2)
    except Exception:
        size_kb = 0.0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Count parameters where meaningful (e.g. PyTorch neural models)
    param_count: Optional[int] = None
    try:
        # Check if underlying model is a PyTorch Module or TabNetClassifier
        if hasattr(model_obj, "model") and hasattr(model_obj.model, "network"):
            # TabNetClassifier network is a PyTorch Module
            net = model_obj.model.network
            param_count = int(sum(p.numel() for p in net.parameters()))
        elif hasattr(model_obj, "parameters"):
            param_count = int(sum(p.numel() for p in model_obj.parameters()))
    except Exception:
        param_count = None

    return {
        "model": model_name,
        "train_time_seconds": round(fit_time_seconds, 3),
        "inference_time_total_seconds": round(mean_inference_time, 4),
        "inference_latency_ms_per_1000": latency_per_1k_ms,
        "throughput_samples_per_sec": throughput_samples_per_sec,
        "serialized_size_kb": size_kb,
        "parameter_count": param_count,
    }


def run_complexity_audit_standalone(model_name: str = "all") -> pd.DataFrame:
    """Run computational complexity comparison across baseline and advanced architectures."""
    import argparse
    from pathlib import Path
    import pandas as pd
    from src.clinical.modeling.models import get_baseline_models
    from src.clinical.modeling.preprocessing import ClinicalPreprocessor
    from src.clinical.benchmarking.catboost import CatBoostModel

    from src.clinical.benchmarking.runtime import get_runtime_output_root

    print("=" * 70)
    print("FusionMedAI: Model Computational Complexity & Efficiency Profiler")
    print("=" * 70)

    repo_root = Path(__file__).resolve().parents[3]
    splits_dir = repo_root / "datasets" / "clinical" / "processed" / "splits"
    output_root = get_runtime_output_root(repo_root, "complexity")

    print("\n[1/3] Loading canonical splits & fitting preprocessor...")
    df_train = pd.read_csv(splits_dir / "train.csv")
    df_val = pd.read_csv(splits_dir / "val.csv")

    preprocessor = ClinicalPreprocessor(scale_numerical=True)
    preprocessor.fit(df_train)
    X_train, y_train, _ = preprocessor.transform(df_train)
    X_val, y_val, _ = preprocessor.transform(df_val)

    baselines = get_baseline_models(random_state=42)
    models_to_test = {}

    if model_name in ["all", "logistic_regression"]:
        models_to_test["logistic_regression"] = baselines["logistic_regression"]
    if model_name in ["all", "random_forest"]:
        models_to_test["random_forest"] = baselines["random_forest"]
    if model_name in ["all", "xgboost"]:
        models_to_test["xgboost"] = baselines["xgboost"]
    if model_name in ["all", "lightgbm"]:
        models_to_test["lightgbm"] = baselines["lightgbm"]
    if model_name in ["all", "catboost"]:
        models_to_test["catboost"] = CatBoostModel(iterations=300, depth=6, random_state=42, verbose=0)
    if model_name in ["all", "tabnet"]:
        try:
            from src.clinical.benchmarking.tabnet import TabNetModel
            models_to_test["tabnet"] = TabNetModel(max_epochs=40, patience=8, random_state=42, verbose=0)
        except ImportError:
            if model_name == "tabnet":
                raise
            print("  -> [Notice] pytorch-tabnet not installed; skipping TabNet.")

    print(f"\n[2/3] Profiling complexity for {len(models_to_test)} architectures...")
    records = []
    for name, mdl in models_to_test.items():
        print(f"  -> Fitting and profiling '{name}'...")
        t0 = time.perf_counter()
        if isinstance(mdl, CatBoostModel) or type(mdl).__name__ == "TabNetModel":
            mdl.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            mdl.fit(X_train, y_train)
        fit_dur = time.perf_counter() - t0

        prof = audit_model_complexity(name, mdl, fit_dur, X_val[:1000])
        records.append(prof)
        print(f"     Fit Time: {prof['train_time_seconds']:.2f}s | Latency (1k): {prof['inference_latency_ms_per_1000']:.2f}ms | Disk Size: {prof['serialized_size_kb']:.1f} KB")

    df_prof = pd.DataFrame(records)
    out_path = output_root / "complexity_results.csv"
    df_prof.to_csv(out_path, index=False)
    try:
        out_rel = out_path.relative_to(repo_root)
    except ValueError:
        out_rel = out_path
    print(f"\n[3/3] Complexity profile saved to: {out_rel}")
    return df_prof


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standalone Complexity Profiler")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "catboost", "tabnet", "xgboost", "lightgbm", "random_forest", "logistic_regression"],
        help="Target architecture to profile (default: all)",
    )
    args = parser.parse_args()

    run_complexity_audit_standalone(model_name=args.model)
