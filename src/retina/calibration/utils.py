import json
import subprocess
from pathlib import Path
from typing import Dict, Any
import torchvision
import numpy as np
import pandas as pd

def generate_manifest(
    model_name: str,
    checkpoint_name: str,
    checkpoint_sha256: str,
    dataset_version: str,
    temperature: float,
    num_bins: int,
    num_validation_samples: int,
    seed: int,
    save_path: Path
) -> Dict[str, Any]:
    """Generates and saves the manifest.json for reproducibility."""
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        git_commit = "unknown"
        
    import torch
    import platform
    import sys
    
    manifest = {
        "calibration_method": "Temperature Scaling",
        "model": model_name,
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_version": dataset_version,
        "temperature": temperature,
        "num_bins": num_bins,
        "num_validation_samples": num_validation_samples,
        "seed": seed,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "git_commit": git_commit,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "cpu",
        "python_version": sys.version,
        "timestamp": __import__("time").time()
    }
    
    with open(save_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    return manifest

def log_to_tensorboard(writer, tag: str, metrics: Dict[str, float], step: int = 0):
    """Logs a dict of metrics to TensorBoard."""
    for key, value in metrics.items():
        writer.add_scalar(f"{tag}/{key}", value, step)
