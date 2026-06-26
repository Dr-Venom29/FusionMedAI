import os
import sys
import json
import logging
import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import (
    TRAIN_CSV,
    TRAIN_IMAGES,
    SEED,
    VALID_LABELS,
    EXPECTED_TRAIN_SAMPLES,
    PROCESSED_SPLITS_DIR,
    ID_COLUMN,
    LABEL_COLUMN
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def validate_dataset(df: pd.DataFrame) -> None:
    """Performs rigorous checks on the dataset before splitting."""
    logger.info("Starting dataset validation...")
    
    # 1. Check required columns exist
    required_cols = [ID_COLUMN, LABEL_COLUMN]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Validation failed: Required column '{col}' is missing.")
            
    # 2. Check no missing values
    missing_id_code = df[ID_COLUMN].isna().sum()
    missing_diagnosis = df[LABEL_COLUMN].isna().sum()
    if missing_id_code > 0 or missing_diagnosis > 0:
        raise ValueError(
            f"Validation failed: Missing values found. "
            f"'{ID_COLUMN}' missing: {missing_id_code}, '{LABEL_COLUMN}' missing: {missing_diagnosis}."
        )
        
    # 3. Check exact label set matches expected
    actual_labels = set(df[LABEL_COLUMN].unique())
    if actual_labels != VALID_LABELS:
        raise ValueError(
            f"Validation failed: Label set {actual_labels} does not match expected {VALID_LABELS}."
        )
        
    # 4. Check duplicate IDs
    if len(df[ID_COLUMN].unique()) != len(df):
        duplicate_count = len(df) - len(df[ID_COLUMN].unique())
        raise ValueError(f"Validation failed: Found {duplicate_count} duplicate '{ID_COLUMN}' values.")
        
    # 5. Check total row count against expected size
    if len(df) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError(
            f"Validation failed: Row count ({len(df)}) does not match EXPECTED_TRAIN_SAMPLES ({EXPECTED_TRAIN_SAMPLES})."
        )
        
    # 6. Verify existence of image files for all ID_COLUMN values
    logger.info("Verifying image files exist for all records...")
    if not TRAIN_IMAGES.exists() or not TRAIN_IMAGES.is_dir():
        raise FileNotFoundError(f"Validation failed: Train image directory '{TRAIN_IMAGES}' does not exist.")
        
    existing_stems = {image.stem for image in TRAIN_IMAGES.glob("*.png")}
    missing_images = [id_code for id_code in df[ID_COLUMN] if id_code not in existing_stems]
    if missing_images:
        raise FileNotFoundError(
            f"Validation failed: {len(missing_images)} image files are missing from {TRAIN_IMAGES}. "
            f"First 5 missing: {missing_images[:5]}"
        )
        
    logger.info("Dataset validation passed successfully.")

def compute_distribution(df: pd.DataFrame) -> dict:
    """Computes sample count and percentage for each class."""
    counts = df[LABEL_COLUMN].value_counts().sort_index()
    total = len(df)
    dist = {}
    for cls in sorted(VALID_LABELS):
        count = int(counts.get(cls, 0))
        percentage = (count / total) * 100
        dist[str(cls)] = {
            "count": count,
            "percentage": round(percentage, 2)
        }
    return dist

def verify_and_log_splits(original_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Verifies that splits are disjoint and logs the class distribution comparisons."""
    logger.info("Verifying split integrity...")
    
    # 1. Size match
    total_split_size = len(train_df) + len(val_df) + len(test_df)
    if total_split_size != len(original_df):
        raise ValueError(f"Split size mismatch: {total_split_size} != {len(original_df)}")
        
    # 2. No overlap check
    train_ids = set(train_df[ID_COLUMN])
    val_ids = set(val_df[ID_COLUMN])
    test_ids = set(test_df[ID_COLUMN])
    
    if not train_ids.isdisjoint(val_ids):
        raise ValueError("Train and Validation splits overlap!")
    if not train_ids.isdisjoint(test_ids):
        raise ValueError("Train and Test splits overlap!")
    if not val_ids.isdisjoint(test_ids):
        raise ValueError("Validation and Test splits overlap!")
    
    logger.info("Integrity check passed: Splits are mutually exclusive and sum to the total dataset.")
    
    # 3. Class distributions
    dist_original = compute_distribution(original_df)
    dist_train = compute_distribution(train_df)
    dist_val = compute_distribution(val_df)
    dist_test = compute_distribution(test_df)
    
    logger.info("=== Class Distribution Comparison ===")
    logger.info(f"{'Class':<6} | {'Original':<20} | {'Train':<20} | {'Val':<20} | {'Test':<20}")
    logger.info("-" * 95)
    for cls in sorted(VALID_LABELS):
        cls_str = str(cls)
        orig_str = f"{dist_original[cls_str]['count']} ({dist_original[cls_str]['percentage']:.2f}%)"
        train_str = f"{dist_train[cls_str]['count']} ({dist_train[cls_str]['percentage']:.2f}%)"
        val_str = f"{dist_val[cls_str]['count']} ({dist_val[cls_str]['percentage']:.2f}%)"
        test_str = f"{dist_test[cls_str]['count']} ({dist_test[cls_str]['percentage']:.2f}%)"
        logger.info(f"{cls:<6} | {orig_str:<20} | {train_str:<20} | {val_str:<20} | {test_str:<20}")
    
    return {
        "original": dist_original,
        "train": dist_train,
        "val": dist_val,
        "test": dist_test
    }

def verify_saved_files(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Reloads saved files and verifies their readability, structure, and content."""
    logger.info("Reloading and verifying saved files...")
    
    train_path = PROCESSED_SPLITS_DIR / "train.csv"
    val_path = PROCESSED_SPLITS_DIR / "val.csv"
    test_path = PROCESSED_SPLITS_DIR / "test.csv"
    
    loaded_train = pd.read_csv(train_path)
    loaded_val = pd.read_csv(val_path)
    loaded_test = pd.read_csv(test_path)
    
    # Verify index-reset dataframes match exactly
    pd.testing.assert_frame_equal(train_df, loaded_train)
    pd.testing.assert_frame_equal(val_df, loaded_val)
    pd.testing.assert_frame_equal(test_df, loaded_test)
    
    logger.info("Verification of saved files passed successfully.")

def main() -> int:
    start_time = time.perf_counter()
    try:
        # Load dataset
        if not TRAIN_CSV.exists():
            logger.error(f"Train CSV file not found at {TRAIN_CSV}")
            return 1
            
        logger.info(f"Loading dataset from {TRAIN_CSV}...")
        df = pd.read_csv(TRAIN_CSV)
        
        # Validate dataset
        validate_dataset(df)
        
        # Split dataset
        logger.info("Performing Two-Stage Stratified Split...")
        # Stage 1: 80% Train, 20% Temp
        train_df, temp_df = train_test_split(
            df,
            test_size=0.20,
            random_state=SEED,
            stratify=df[LABEL_COLUMN]
        )
        # Stage 2: 10% Validation, 10% Test (split Temp in half)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=SEED,
            stratify=temp_df[LABEL_COLUMN]
        )
        
        # Sort splits by ID_COLUMN and reset index for deterministic saving and reproducibility
        logger.info(f"Sorting splits by '{ID_COLUMN}' for reproducibility...")
        train_df = train_df.sort_values(ID_COLUMN).reset_index(drop=True)
        val_df = val_df.sort_values(ID_COLUMN).reset_index(drop=True)
        test_df = test_df.sort_values(ID_COLUMN).reset_index(drop=True)
        
        logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Verify splits and print distribution
        distributions = verify_and_log_splits(df, train_df, val_df, test_df)
        
        # Save splits
        PROCESSED_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving splits to {PROCESSED_SPLITS_DIR}...")
        train_df.to_csv(PROCESSED_SPLITS_DIR / "train.csv", index=False)
        val_df.to_csv(PROCESSED_SPLITS_DIR / "val.csv", index=False)
        test_df.to_csv(PROCESSED_SPLITS_DIR / "test.csv", index=False)
        
        # Save summary statistics
        stats = {
            "version": "1.0",
            "created_by": "split_dataset.py",
            "total_samples": len(df),
            "train_samples": len(train_df),
            "validation_samples": len(val_df),
            "test_samples": len(test_df),
            "random_state": SEED,
            "stratified": True,
            "class_distributions": distributions
        }
        
        stats_path = PROCESSED_SPLITS_DIR / "split_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved split statistics to {stats_path}")
        
        # Verify saved files
        verify_saved_files(train_df, val_df, test_df)
        
        elapsed = time.perf_counter() - start_time
        logger.info(f"Stratified split completed successfully in {elapsed:.2f} seconds.")
        return 0
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
