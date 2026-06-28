import sys
from pathlib import Path
import pandas as pd
import torch

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    TRAIN_SPLIT_CSV,
    VAL_SPLIT_CSV,
    TEST_SPLIT_CSV,
    NUM_CLASSES
)
from src.data.dataloader import create_dataloaders

def test_train_dataloader() -> None:
    print("Test 1: Train DataLoader verification...")
    # Set num_workers=0 to prevent multiprocessing overhead/issues on Windows during verification
    train_loader, val_loader, test_loader = create_dataloaders(num_workers=0)
    
    # Check dataset size against split CSV row count
    expected_train = len(pd.read_csv(TRAIN_SPLIT_CSV))
    train_dataset = train_loader.dataset
    print(f"  Train Dataset size: {len(train_dataset)} (expected: {expected_train})")
    if len(train_dataset) != expected_train:
        raise ValueError(
            f"Failed: Train dataset size mismatch. Expected {expected_train}, got {len(train_dataset)}"
        )
        
    # Check first batch loading
    print("  Fetching first batch from Train DataLoader...")
    batch_x, batch_y = next(iter(train_loader))
    
    print(f"  Batch X shape: {batch_x.shape} (expected: ({BATCH_SIZE}, 3, {IMAGE_SIZE}, {IMAGE_SIZE}))")
    print(f"  Batch Y shape: {batch_y.shape} (expected: ({BATCH_SIZE},))")
    
    if batch_x.shape != (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Failed: Train batch image shape mismatch: {batch_x.shape}")
        
    if batch_y.shape != (BATCH_SIZE,):
        raise ValueError(f"Failed: Train batch labels shape mismatch: {batch_y.shape}")
        
    if batch_x.dtype != torch.float32:
        raise TypeError(f"Failed: Expected image batch dtype torch.float32, got {batch_x.dtype}")
        
    if not batch_y.dtype.is_floating_point and batch_y.dtype != torch.int64:
        raise TypeError(f"Failed: Expected label batch integer type (torch.int64), got {batch_y.dtype}")
        
    if batch_y.min() < 0 or batch_y.max() >= NUM_CLASSES:
        raise ValueError(f"Failed: Train labels out of bounds: min={batch_y.min()}, max={batch_y.max()}")
        
    if not torch.isfinite(batch_x).all():
        raise ValueError("Failed: Train batch contains non-finite values (NaN/Inf).")
        
    print("[OK] Train DataLoader passed all checks successfully.")

def test_val_test_dataloaders() -> None:
    print("\nTest 2: Validation and Test DataLoaders verification...")
    train_loader, val_loader, test_loader = create_dataloaders(num_workers=0)
    
    # 1. Validation checks
    expected_val = len(pd.read_csv(VAL_SPLIT_CSV))
    val_dataset = val_loader.dataset
    print(f"  Validation Dataset size: {len(val_dataset)} (expected: {expected_val})")
    if len(val_dataset) != expected_val:
        raise ValueError(
            f"Failed: Validation dataset size mismatch. Expected {expected_val}, got {len(val_dataset)}"
        )
        
    val_x, val_y = next(iter(val_loader))
    if val_x.shape != (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Failed: Validation batch image shape mismatch: {val_x.shape}")
    if val_y.shape != (BATCH_SIZE,):
        raise ValueError(f"Failed: Validation batch labels shape mismatch: {val_y.shape}")
        
    if val_y.min() < 0 or val_y.max() >= NUM_CLASSES:
        raise ValueError(f"Failed: Validation labels out of bounds: min={val_y.min()}, max={val_y.max()}")
        
    print("[OK] Validation DataLoader passed all checks successfully.")
    
    # 2. Test checks
    expected_test = len(pd.read_csv(TEST_SPLIT_CSV))
    test_dataset = test_loader.dataset
    print(f"  Test Dataset size: {len(test_dataset)} (expected: {expected_test})")
    if len(test_dataset) != expected_test:
        raise ValueError(
            f"Failed: Test dataset size mismatch. Expected {expected_test}, got {len(test_dataset)}"
        )
        
    test_x, test_y = next(iter(test_loader))
    if test_x.shape != (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Failed: Test batch image shape mismatch: {test_x.shape}")
    if test_y.shape != (BATCH_SIZE,):
        raise ValueError(f"Failed: Test batch labels shape mismatch: {test_y.shape}")
        
    if test_y.min() < 0 or test_y.max() >= NUM_CLASSES:
        raise ValueError(f"Failed: Test labels out of bounds: min={test_y.min()}, max={test_y.max()}")
        
    print("[OK] Test DataLoader passed all checks successfully.")

def test_full_iteration() -> None:
    print("\nTest 3: Full iteration check (Validation DataLoader)...")
    _, val_loader, _ = create_dataloaders(num_workers=0)
    expected_val = len(pd.read_csv(VAL_SPLIT_CSV))
    
    batch_count = 0
    sample_count = 0
    for x, y in val_loader:
        batch_count += 1
        sample_count += len(x)
        if not torch.isfinite(x).all():
            raise ValueError(f"Failed: Found non-finite values in validation batch {batch_count}")
            
    print(f"  Iterated successfully over {batch_count} batches containing {sample_count} total samples.")
    if sample_count != expected_val:
        raise ValueError(
            f"Failed: Iterated count ({sample_count}) does not match dataset size ({expected_val})."
        )
        
    print("[OK] Full iteration test passed.")

def main() -> int:
    try:
        test_train_dataloader()
        test_val_test_dataloaders()
        test_full_iteration()
        print("\n==========================================")
        print("ALL DATALOADER TESTS PASSED SUCCESSFULLY!")
        print("==========================================")
        return 0
    except Exception as e:
        print(f"\n[FAIL] DataLoader verification failed: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
