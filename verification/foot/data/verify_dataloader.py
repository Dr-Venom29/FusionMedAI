import sys
import json
import random
from pathlib import Path
import torch
import numpy as np

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
from src.foot.data.dataloader import create_foot_dataloaders

def verify_foot_dataloaders():
    print("==================================================")
    print("Running Verification — Foot DFU DataLoaders")
    print("==================================================")
    
    # 1. Test Dataloader Creation
    print("1. Testing create_foot_dataloaders instantiation...")
    train_loader, val_loader, test_loader = create_foot_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        seed=SEED
    )
    
    print(f" - Train Loader: {len(train_loader):,} batches ({len(train_loader.dataset):,} images)")
    print(f" - Val Loader:   {len(val_loader):,} batches ({len(val_loader.dataset):,} images)")
    print(f" - Test Loader:  {len(test_loader):,} batches ({len(test_loader.dataset):,} images)")
    
    assert len(train_loader.dataset) == 8038, "Train dataset size mismatch"
    assert len(val_loader.dataset) == 1006, "Val dataset size mismatch"
    assert len(test_loader.dataset) == 1006, "Test dataset size mismatch"
    
    # 2. Test Batch Fetching & Tensor Layout
    print("\n2. Testing batch fetching & tensor layout...")
    images, labels = next(iter(train_loader))
    print(f" - Train Batch 0 -> Images: {images.shape}, Labels: {labels.shape}")
    assert images.shape == torch.Size([BATCH_SIZE, 3, 224, 224]), "Train batch image shape mismatch"
    assert labels.shape == torch.Size([BATCH_SIZE]), "Train batch label shape mismatch"
    assert images.dtype == torch.float32, "Expected float32 images"
    assert labels.dtype == torch.int64, "Expected int64 labels"
    assert labels.min() >= 0 and labels.max() <= 3, "Labels out of bounds [0..3]"
    
    # 3. Test Validation Determinism (0% Augmentation Leakage)
    print("\n3. Testing Validation Determinism...")
    val_images_pass1, val_labels_pass1 = next(iter(val_loader))
    val_images_pass2, val_labels_pass2 = next(iter(val_loader))
    
    val_diff = torch.abs(val_images_pass1 - val_images_pass2).max().item()
    val_lbl_diff = torch.abs(val_labels_pass1 - val_labels_pass2).max().item()
    print(f" - Validation Max Pixel Difference across passes: {val_diff}")
    assert val_diff == 0.0, "Validation loader must be 100% deterministic"
    assert val_lbl_diff == 0.0, "Validation labels must be 100% deterministic"
    
    # 4. Test Seeding Reproducibility
    print("\n4. Testing Seeding Reproducibility...")
    torch.manual_seed(SEED)
    tr1_loader, _, _ = create_foot_dataloaders(batch_size=16, num_workers=0, seed=SEED)
    img1, lbl1 = next(iter(tr1_loader))
    
    torch.manual_seed(SEED)
    tr2_loader, _, _ = create_foot_dataloaders(batch_size=16, num_workers=0, seed=SEED)
    img2, lbl2 = next(iter(tr2_loader))
    
    seed_diff = torch.abs(img1 - img2).max().item()
    print(f" - Max pixel difference across identical SEED runs: {seed_diff}")
    assert seed_diff == 0.0, "DataLoader seeding reproducibility failed!"
    
    print("\nAcceptance Checklist:")
    print(" [PASS] Train, Val, and Test DataLoaders successfully created")
    print(" [PASS] Batch tensor shapes verified ([B, 3, 224, 224] float32, [B] int64)")
    print(" [PASS] Validation loader verified as 100% deterministic (0% augmentation leakage)")
    print(" [PASS] Seeding reproducibility verified (0.0 pixel variance across identical runs)")
    print("\nFoot DataLoaders Verification COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    verify_foot_dataloaders()
