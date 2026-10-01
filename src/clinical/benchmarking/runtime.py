import os
from pathlib import Path


def get_runtime_output_root(repo_root: Path) -> Path:
    """Return a writable runtime output directory.

    Kaggle:
        /kaggle/working/FusionMedAI_outputs
    Local:
        <repo_root>/datasets/clinical/metadata/modeling
    """
    kaggle_root = Path("/kaggle/working")
    if kaggle_root.exists() and os.access(kaggle_root, os.W_OK):
        output_root = kaggle_root / "FusionMedAI_outputs"
    else:
        output_root = repo_root / "datasets" / "clinical" / "metadata" / "modeling"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root
