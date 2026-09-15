import sys
import random
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.config import (
    TRAIN_SPLIT_CSV,
    VAL_SPLIT_CSV,
    TEST_SPLIT_CSV,
    RAW_DATA,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED
)
from src.foot.data.dataset import FootDFUDataset
from src.foot.data.transforms import (
    get_foot_train_transforms,
    get_foot_val_transforms,
    get_foot_test_transforms
)

def seed_worker(worker_id: int) -> None:
    """
    Ensures deterministic data loading across PyTorch multi-process DataLoader workers.
    Sets numpy and Python random seeds derived from the initial worker seed.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_foot_dataloaders(
    train_csv: Union[str, Path] = TRAIN_SPLIT_CSV,
    val_csv: Union[str, Path] = VAL_SPLIT_CSV,
    test_csv: Union[str, Path] = TEST_SPLIT_CSV,
    image_dir: Union[str, Path] = RAW_DATA,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    seed: int = SEED,
    drop_last: bool = False,
    persistent_workers: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns PyTorch DataLoaders for Train, Validation, and Test splits of the Foot DFU dataset.
    
    Requirements Enforced:
    1. Deterministic Validation/Test: shuffle=False, no augmentation transforms applied.
    2. Shuffling for Training: shuffle=True for train loader only.
    3. Reproducible Workers: Generator initialized with fixed seed and seed_worker worker_init_fn.
    4. Configurable Batch Size & Workers.
    """
    # 1. Instantiate Transforms
    train_transforms = get_foot_train_transforms()
    val_transforms = get_foot_val_transforms()
    test_transforms = get_foot_test_transforms()
    
    # 2. Instantiate Datasets
    train_dataset = FootDFUDataset(
        csv_file=train_csv,
        img_dir=image_dir,
        transform=train_transforms,
        is_train=True
    )
    
    val_dataset = FootDFUDataset(
        csv_file=val_csv,
        img_dir=image_dir,
        transform=val_transforms,
        is_train=False
    )
    
    test_dataset = FootDFUDataset(
        csv_file=test_csv,
        img_dir=image_dir,
        transform=test_transforms,
        is_train=False
    )
    
    # 3. Configure Reproducible PyTorch Generator
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Handle persistent_workers constraint
    use_persistent = persistent_workers if num_workers > 0 else False
    
    # 4. Construct DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle train split strictly
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=drop_last,
        persistent_workers=use_persistent
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic validation evaluation
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
        persistent_workers=use_persistent
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic held-out test evaluation
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
        persistent_workers=use_persistent
    )
    
    return train_loader, val_loader, test_loader


def _test_foot_dataloaders():
    """Unit test for create_foot_dataloaders."""
    print("Testing create_foot_dataloaders...")
    train_loader, val_loader, test_loader = create_foot_dataloaders(
        batch_size=16,
        num_workers=0,  # Single-process for unit testing
        pin_memory=False
    )
    
    print(f" - Train Loader Batches: {len(train_loader):,} (Dataset size: {len(train_loader.dataset):,})")
    print(f" - Val Loader Batches:   {len(val_loader):,} (Dataset size: {len(val_loader.dataset):,})")
    print(f" - Test Loader Batches:  {len(test_loader):,} (Dataset size: {len(test_loader.dataset):,})")
    
    # Verify Train Batch
    images, labels = next(iter(train_loader))
    print(f" - Train Batch 0 -> Images: {images.shape}, Labels: {labels.shape}")
    assert images.shape == torch.Size([16, 3, 224, 224]), "Train batch image shape incorrect"
    assert labels.shape == torch.Size([16]), "Train batch label shape incorrect"
    assert labels.min() >= 0 and labels.max() <= 3, "Train label out of bounds [0..3]"
    
    # Verify Val Batch
    val_images, val_labels = next(iter(val_loader))
    print(f" - Val Batch 0   -> Images: {val_images.shape}, Labels: {val_labels.shape}")
    assert val_images.shape == torch.Size([16, 3, 224, 224]), "Val batch image shape incorrect"
    assert val_labels.shape == torch.Size([16]), "Val batch label shape incorrect"
    
    # Verify Test Batch
    test_images, test_labels = next(iter(test_loader))
    print(f" - Test Batch 0  -> Images: {test_images.shape}, Labels: {test_labels.shape}")
    assert test_images.shape == torch.Size([16, 3, 224, 224]), "Test batch image shape incorrect"
    assert test_labels.shape == torch.Size([16]), "Test batch label shape incorrect"
    
    print("create_foot_dataloaders verified successfully!")

if __name__ == "__main__":
    _test_foot_dataloaders()
