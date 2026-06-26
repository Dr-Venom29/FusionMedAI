import sys
import time
from pathlib import Path
from typing import Tuple
import pandas as pd
import torch

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import (
    TRAIN_SPLIT_CSV,
    VAL_SPLIT_CSV,
    TEST_SPLIT_CSV,
    TRAIN_IMAGES,
    EXPECTED_TRAIN_SAMPLES,
    NUM_CLASSES,
    ID_COLUMN,
    LABEL_COLUMN,
    VALID_LABELS,
    BATCH_SIZE,
    IMAGE_SIZE
)
from src.data.dataloader import create_dataloaders

def verify_csvs() -> Tuple[bool, dict]:
    print("Verifying split CSVs...")
    df_dict = {}
    csv_paths = {
        "Train": TRAIN_SPLIT_CSV,
        "Validation": VAL_SPLIT_CSV,
        "Test": TEST_SPLIT_CSV
    }
    
    for split_name, path in csv_paths.items():
        if not path.exists():
            print(f"✗ {split_name} CSV not found at '{path}'")
            return False, {}
            
        try:
            df = pd.read_csv(path)
            if len(df) == 0:
                print(f"✗ {split_name} CSV is empty")
                return False, {}
                
            required = [ID_COLUMN, LABEL_COLUMN]
            for col in required:
                if col not in df.columns:
                    print(f"✗ {split_name} CSV is missing required column '{col}'")
                    return False, {}
                    
            df_dict[split_name] = df
            print(f"  [OK] {split_name} CSV: {len(df)} rows loaded successfully.")
        except Exception as e:
            print(f"✗ Failed to read {split_name} CSV: {e}")
            return False, {}
            
    return True, df_dict

def verify_split_integrity(df_dict: dict) -> bool:
    print("\nVerifying split integrity...")
    train_df = df_dict["Train"]
    val_df = df_dict["Validation"]
    test_df = df_dict["Test"]
    
    train_ids = set(train_df[ID_COLUMN])
    val_ids = set(val_df[ID_COLUMN])
    test_ids = set(test_df[ID_COLUMN])
    
    # 1. Mutually exclusive checks
    if not train_ids.isdisjoint(val_ids):
        print("✗ Train and Validation splits overlap!")
        return False
    if not train_ids.isdisjoint(test_ids):
        print("✗ Train and Test splits overlap!")
        return False
    if not val_ids.isdisjoint(test_ids):
        print("✗ Validation and Test splits overlap!")
        return False
    print("  [OK] Splits are mutually exclusive (zero overlap in ID codes).")
    
    # 2. Row count sum check
    total_samples = len(train_df) + len(val_df) + len(test_df)
    if total_samples != EXPECTED_TRAIN_SAMPLES:
        print(f"✗ Total split samples ({total_samples}) != EXPECTED_TRAIN_SAMPLES ({EXPECTED_TRAIN_SAMPLES})")
        return False
    print(f"  [OK] Total split samples sum matches expected: {total_samples}")
    return True

def compute_dist_dict(df: pd.DataFrame) -> dict:
    counts = df[LABEL_COLUMN].value_counts().sort_index()
    total = len(df)
    dist = {}
    for cls in sorted(VALID_LABELS):
        count = int(counts.get(cls, 0))
        pct = (count / total) * 100
        dist[cls] = f"{count} ({pct:.2f}%)"
    return dist

def print_class_distributions(df_dict: dict) -> bool:
    print("\nComputing class distributions...")
    try:
        # Load the original full CSV to get baseline distribution
        original_csv_path = TRAIN_SPLIT_CSV.parents[2] / "raw" / "aptos2019" / "train.csv"
        if not original_csv_path.exists():
            print(f"✗ Original CSV not found at '{original_csv_path}' for distribution baseline.")
            return False
            
        orig_df = pd.read_csv(original_csv_path)
        
        orig_dist = compute_dist_dict(orig_df)
        train_dist = compute_dist_dict(df_dict["Train"])
        val_dist = compute_dist_dict(df_dict["Validation"])
        test_dist = compute_dist_dict(df_dict["Test"])
        
        print(f"\n{'Class':<6} | {'Original':<20} | {'Train':<20} | {'Val':<20} | {'Test':<20}")
        print("-" * 95)
        for cls in sorted(VALID_LABELS):
            print(f"{cls:<6} | {orig_dist[cls]:<20} | {train_dist[cls]:<20} | {val_dist[cls]:<20} | {test_dist[cls]:<20}")
        print()
        return True
    except Exception as e:
        print(f"✗ Failed to compute class distributions: {e}")
        return False

def verify_dataset_loader(df_dict: dict) -> Tuple[bool, dict, Tuple]:
    print("Verifying Dataset, Transforms, and DataLoader integration...")
    try:
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(num_workers=0)
        
        # 1. Dataset length matches CSV rows
        train_ds = train_loader.dataset
        val_ds = val_loader.dataset
        test_ds = test_loader.dataset
        
        if len(train_ds) != len(df_dict["Train"]):
            print(f"✗ Train Dataset size ({len(train_ds)}) != CSV row count ({len(df_dict['Train'])})")
            return False, {}, ()
        if len(val_ds) != len(df_dict["Validation"]):
            print(f"✗ Validation Dataset size ({len(val_ds)}) != CSV row count ({len(df_dict['Validation'])})")
            return False, {}, ()
        if len(test_ds) != len(df_dict["Test"]):
            print(f"✗ Test Dataset size ({len(test_ds)}) != CSV row count ({len(df_dict['Test'])})")
            return False, {}, ()
            
        print("  [OK] Dataset lengths match their corresponding split CSV sizes.")
        
        # 2. Check complete batch loader and transforms
        print("  Fetching a training batch...")
        train_iter = iter(train_loader)
        images, labels = next(train_iter)
        
        # Assert batch shapes
        expected_img_shape = (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        if images.shape != expected_img_shape:
            print(f"✗ Unexpected image batch shape: {images.shape} (expected {expected_img_shape})")
            return False, {}, ()
        if labels.shape != (BATCH_SIZE,):
            print(f"✗ Unexpected label batch shape: {labels.shape} (expected ({BATCH_SIZE},))")
            return False, {}, ()
            
        # Assert dtypes
        if images.dtype != torch.float32:
            print(f"✗ Unexpected image batch dtype: {images.dtype} (expected torch.float32)")
            return False, {}, ()
        if labels.dtype != torch.int64:
            print(f"✗ Unexpected label batch dtype: {labels.dtype} (expected torch.int64)")
            return False, {}, ()
            
        # Assert labels are within range
        if labels.min() < 0 or labels.max() >= NUM_CLASSES:
            print(f"✗ Labels out of valid range [0, {NUM_CLASSES - 1}]: min={labels.min()}, max={labels.max()}")
            return False, {}, ()
            
        # Assert finiteness
        if not torch.isfinite(images).all():
            print("✗ Image batch contains non-finite values (NaN/Inf).")
            return False, {}, ()
            
        # Assert device type is CPU
        if images.device.type != "cpu" or labels.device.type != "cpu":
            print(f"✗ DataLoader returned tensors on device {images.device.type} instead of cpu.")
            return False, {}, ()
            
        print("  [OK] DataLoader batch loaded successfully with correct shapes, CPU device, and finite float32 values.")
        
        val_x, val_y = next(iter(val_loader))
        test_x, test_y = next(iter(test_loader))
        if val_x.shape != expected_img_shape or test_x.shape != expected_img_shape:
            print("✗ Validation or Test batch image shape mismatch.")
            return False, {}, ()
            
        print("  [OK] Validation and Test DataLoader batches loaded with correct shape.")
        
        status_dict = {
            "dataset": True,
            "transforms": True,
            "image_loading": True,
            "dataloader": True,
            "batch_shapes": True
        }
        return True, status_dict, (train_loader, val_loader, test_loader)
        
    except Exception as e:
        print(f"✗ Failed dataset and dataloader verification: {e}")
        return False, {}, ()

def verify_end_to_end_iteration(train_loader, val_loader, test_loader) -> Tuple[bool, dict]:
    print("Verifying end-to-end iteration (iterating splits)...")
    results = {}
    loaders = {
        "Train": train_loader,
        "Validation": val_loader,
        "Test": test_loader
    }
    
    try:
        for name, loader in loaders.items():
            start_time = time.perf_counter()
            batch_count = 0
            sample_count = 0
            
            for x, y in loader:
                batch_count += 1
                sample_count += len(x)
                
            elapsed = time.perf_counter() - start_time
            throughput = sample_count / elapsed if elapsed > 0 else 0.0
            
            results[name] = {
                "samples": sample_count,
                "batches": batch_count,
                "time": elapsed,
                "throughput": throughput
            }
            print(f"  {name:<10}: Loaded {sample_count:<4} samples in {batch_count:<3} batches. "
                  f"Time: {elapsed:.2f}s | Throughput: {throughput:.1f} samples/sec")
                  
        return True, results
    except Exception as e:
        print(f"✗ End-to-end iteration verification failed with error: {e}")
        return False, {}

def main() -> int:
    # Track status of each check
    checks = {
        "Train CSV": False,
        "Validation CSV": False,
        "Test CSV": False,
        "Split Integrity": False,
        "Class Distribution": False,
        "Dataset": False,
        "Transforms": False,
        "Image Loading": False,
        "DataLoader": False,
        "Batch Shapes": False,
        "Full Iteration": False,
        "Throughput": False
    }
    
    print("==========================================")
    print("STARTING END-TO-END PIPELINE VERIFICATION")
    print("==========================================")
    
    # 1. Verify CSVs
    csv_ok, df_dict = verify_csvs()
    if csv_ok:
        checks["Train CSV"] = True
        checks["Validation CSV"] = True
        checks["Test CSV"] = True
    else:
        return 1
        
    # 2. Verify Split Integrity
    integrity_ok = verify_split_integrity(df_dict)
    if integrity_ok:
        checks["Split Integrity"] = True
        
    # 3. Verify Class Distributions
    dist_ok = print_class_distributions(df_dict)
    if dist_ok:
        checks["Class Distribution"] = True
        
    # 4. Dataset & DataLoader integration
    loader_ok, sub_status, loaders = verify_dataset_loader(df_dict)
    if loader_ok:
        checks["Dataset"] = sub_status["dataset"]
        checks["Transforms"] = sub_status["transforms"]
        checks["Image Loading"] = sub_status["image_loading"]
        checks["DataLoader"] = sub_status["dataloader"]
        checks["Batch Shapes"] = sub_status["batch_shapes"]
    else:
        return 1
        
    # 5. End-to-end iteration verification
    train_loader, val_loader, test_loader = loaders
    iteration_ok, iteration_results = verify_end_to_end_iteration(train_loader, val_loader, test_loader)
    if iteration_ok:
        checks["Full Iteration"] = True
        checks["Throughput"] = True
        
    # Final summary panel
    all_passed = all(checks.values())
    
    print("\n==========================================")
    print("END-TO-END PIPELINE VERIFICATION SUMMARY")
    print("==========================================")
    print(f"Train CSV ............................. {'PASS' if checks['Train CSV'] else 'FAIL'}")
    print(f"Validation CSV ........................ {'PASS' if checks['Validation CSV'] else 'FAIL'}")
    print(f"Test CSV .............................. {'PASS' if checks['Test CSV'] else 'FAIL'}")
    print()
    print(f"Split Integrity ....................... {'PASS' if checks['Split Integrity'] else 'FAIL'}")
    print(f"Class Distribution .................... {'PASS' if checks['Class Distribution'] else 'FAIL'}")
    print()
    print(f"Dataset ............................... {'PASS' if checks['Dataset'] else 'FAIL'}")
    print(f"Transforms ............................ {'PASS' if checks['Transforms'] else 'FAIL'}")
    print(f"Image Loading ......................... {'PASS' if checks['Image Loading'] else 'FAIL'}")
    print()
    print(f"DataLoader ............................ {'PASS' if checks['DataLoader'] else 'FAIL'}")
    print(f"Batch Shapes .......................... {'PASS' if checks['Batch Shapes'] else 'FAIL'}")
    print(f"Full Iteration ........................ {'PASS' if checks['Full Iteration'] else 'FAIL'}")
    print()
    print(f"Throughput ............................ {'PASS' if checks['Throughput'] else 'FAIL'}")
    print("==========================================")
    
    if all_passed:
        print("ALL PIPELINE CHECKS PASSED")
        print("==========================================")
        return 0
    else:
        print("✗ PIPELINE VERIFICATION ENCOUNTERED FAILURES.")
        print("==========================================")
        return 1

if __name__ == "__main__":
    sys.exit(main())
