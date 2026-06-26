import sys
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import (
    TRAIN_SPLIT_CSV,
    VAL_SPLIT_CSV,
    TEST_SPLIT_CSV,
    TRAIN_IMAGES,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    DROP_LAST
)
from src.data.dataset import RetinaDataset
from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms
)

def create_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    persistent_workers: bool = PERSISTENT_WORKERS,
    drop_last: bool = DROP_LAST
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns PyTorch DataLoaders for the Train, Validation, and Test splits.
    
    Args:
        batch_size: Number of samples per batch.
        num_workers: How many subprocesses to use for data loading.
        pin_memory: If True, copy Tensors into CUDA pinned memory before returning them.
        persistent_workers: If True, keep data loading workers alive after a dataset has been consumed.
        drop_last: If True, drop the last incomplete batch (training only).
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for (train, validation, test) splits.
        
    Raises:
        FileNotFoundError: If split CSVs or image directories are missing.
    """
    # 1. Pre-creation validation checks
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
        
    csv_paths = {
        "Train": TRAIN_SPLIT_CSV,
        "Validation": VAL_SPLIT_CSV,
        "Test": TEST_SPLIT_CSV
    }
    
    for split_name, path in csv_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot create dataloaders: {split_name} split CSV file not found at '{path}'."
            )
            
    if not TRAIN_IMAGES.exists() or not TRAIN_IMAGES.is_dir():
        raise FileNotFoundError(
            f"Cannot create dataloaders: Image directory not found at '{TRAIN_IMAGES}'."
        )
        
    # 2. Instantiate datasets
    train_dataset = RetinaDataset(
        csv_file=TRAIN_SPLIT_CSV,
        image_dir=TRAIN_IMAGES,
        transform=get_train_transforms()
    )
    
    val_dataset = RetinaDataset(
        csv_file=VAL_SPLIT_CSV,
        image_dir=TRAIN_IMAGES,
        transform=get_val_transforms()
    )
    
    test_dataset = RetinaDataset(
        csv_file=TEST_SPLIT_CSV,
        image_dir=TRAIN_IMAGES,
        transform=get_test_transforms()
    )
    
    # 3. Construct PyTorch DataLoaders
    # Validation and test splits should never shuffle and should not drop the last batch.
    # persistent_workers can only be True if num_workers > 0.
    persistent_workers = persistent_workers and num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers
    )
    
    return train_loader, val_loader, test_loader
