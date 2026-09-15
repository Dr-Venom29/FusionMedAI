import os
import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.config import (
    SEED,
    NUM_CLASSES,
    CLASS_NAMES,
    IMAGE_SIZE,
    OBSERVED_DATASET_MEAN,
    OBSERVED_DATASET_STD,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    PROCESSED_SPLITS_DIR
)

def verify_baseline_contract():
    print("==================================================")
    print("Verifying Baseline Experimental Contract")
    print("==================================================")
    
    errors = []
    
    # 1. Verify Dataset Split Manifests
    train_csv = PROCESSED_SPLITS_DIR / "train.csv"
    val_csv = PROCESSED_SPLITS_DIR / "val.csv"
    test_csv = PROCESSED_SPLITS_DIR / "test.csv"
    index_csv = PROCESSED_SPLITS_DIR / "index.csv"
    
    for split_p in [train_csv, val_csv, test_csv, index_csv]:
        if not split_p.exists():
            errors.append(f"Missing frozen split CSV: {split_p}")
        else:
            print(f" [PASS] Frozen split manifest verified: {split_p.name}")
            
    # 2. Verify Hyperparameter Contract Values
    if SEED != 42:
        errors.append(f"Expected SEED=42, found {SEED}")
    else:
        print(" [PASS] Random Seed 42 verified.")
        
    if NUM_CLASSES != 4 or len(CLASS_NAMES) != 4:
        errors.append(f"Expected 4 Wagner classes, found {NUM_CLASSES}")
    else:
        print(" [PASS] 4-Class Wagner Mapping verified.")
        
    if IMAGE_SIZE != 224:
        errors.append(f"Expected IMAGE_SIZE=224, found {IMAGE_SIZE}")
    else:
        print(" [PASS] Resolution 224x224 verified.")
        
    if OBSERVED_DATASET_MEAN != [0.4937, 0.3630, 0.3272]:
        errors.append(f"Incorrect dataset mean: {OBSERVED_DATASET_MEAN}")
    else:
        print(" [PASS] Observed RGB dataset mean [0.4937, 0.3630, 0.3272] verified.")
        
    if BATCH_SIZE != 32:
        errors.append(f"Expected BATCH_SIZE=32, found {BATCH_SIZE}")
    else:
        print(" [PASS] Batch Size 32 verified.")
        
    if LEARNING_RATE != 1e-4 or WEIGHT_DECAY != 1e-4:
        errors.append(f"Incorrect optimizer learning rate or weight decay: LR={LEARNING_RATE}, WD={WEIGHT_DECAY}")
    else:
        print(" [PASS] AdamW Optimizer settings (LR=1e-4, WD=1e-4) verified.")
        
    if EPOCHS != 20 or PATIENCE != 10:
        errors.append(f"Incorrect epochs/patience: EPOCHS={EPOCHS}, PATIENCE={PATIENCE}")
    else:
        print(" [PASS] Training Epochs (20) & Patience (10) verified.")
        
    # 3. Verify Contract Documentation File
    contract_doc = Path(__file__).resolve().parents[3] / "research" / "foot" / "Volume_04_Baseline_Framework" / "01_Baseline_Experimental_Contract.md"
    if not contract_doc.exists():
        errors.append(f"Missing Volume 04 Baseline Contract doc: {contract_doc}")
    else:
        print(" [PASS] Research Volume 04 Contract documentation verified.")

    if errors:
        print("\n[FAIL] Baseline Contract Verification Errors:")
        for err in errors:
            print(f" - {err}")
        sys.exit(1)
    else:
        print("\n==================================================")
        print("BASELINE EXPERIMENTAL CONTRACT: PASSED")
        print("==================================================")

if __name__ == "__main__":
    verify_baseline_contract()
