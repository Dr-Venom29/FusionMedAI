import os
from pathlib import Path


def get_runtime_output_root(repo_root: Path, component: str | None = None) -> Path:
    """Return a writable runtime output directory.

    Kaggle:
        /kaggle/working/FusionMedAI_outputs/<component>
    Local:
        <repo_root>/experiments/clinical/<component>
    """
    kaggle_root = Path("/kaggle/working")
    if kaggle_root.exists() and os.access(kaggle_root, os.W_OK):
        output_root = kaggle_root / "FusionMedAI_outputs"
    else:
        output_root = repo_root / "experiments" / "clinical"
    if component:
        output_root = output_root / component
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root
