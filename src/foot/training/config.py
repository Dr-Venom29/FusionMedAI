import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
import torch

from src.foot.config import (
    PROJECT_ROOT,
    DATASET_ROOT,
    PROCESSED_SPLITS_DIR,
    FOOT_EXPERIMENTS_DIR,
    SEED,
    NUM_CLASSES,
    CLASS_NAMES,
    IMAGE_SIZE,
    OBSERVED_DATASET_MEAN,
    OBSERVED_DATASET_STD,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    DEVICE
)

@dataclass
class BaselineConfig:
    """
    Frozen Experimental Contract & Configuration for Phase 10.4 Baseline Training.
    """
    # System & Paths
    project_root: Path = PROJECT_ROOT
    dataset_root: Path = DATASET_ROOT
    splits_dir: Path = PROCESSED_SPLITS_DIR
    experiments_dir: Path = FOOT_EXPERIMENTS_DIR
    
    # Reproducibility
    seed: int = SEED
    
    # Dataset Parameters
    num_classes: int = NUM_CLASSES
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))
    image_size: int = IMAGE_SIZE
    normalization_mean: List[float] = field(default_factory=lambda: list(OBSERVED_DATASET_MEAN))
    normalization_std: List[float] = field(default_factory=lambda: list(OBSERVED_DATASET_STD))
    
    # Architecture & Model
    model_name: str = "resnet50"  # Default baseline architecture (10.4.4)
    pretrained: bool = True
    dropout_rate: float = 0.2
    
    # Training Hyperparameters
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS if torch.cuda.is_available() else 0
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    patience: int = PATIENCE
    device: str = DEVICE
    use_amp: bool = torch.cuda.is_available()
    
    # Loss Configuration ('unweighted' = Baseline A, 'weighted' = Sensitivity Baseline)
    loss_type: str = "unweighted"
    class_weights: List[float] = field(default_factory=lambda: [1.0870, 1.0669, 1.0000, 1.0731])
    
    # Optional Explicit Path Overrides
    custom_experiment_dir: Optional[Path] = None
    
    @property
    def experiment_dir(self) -> Path:
        if self.custom_experiment_dir is not None:
            path = Path(self.custom_experiment_dir)
        else:
            exp_name = f"baseline_{self.model_name}_{self.loss_type}"
            path = self.experiments_dir / exp_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def checkpoint_dir(self) -> Path:
        path = self.experiment_dir / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path
