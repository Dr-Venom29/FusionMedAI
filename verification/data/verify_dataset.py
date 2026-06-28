import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import (
    METADATA_DIR,
    METADATA_VALIDATION_DIR,
    METADATA_STATISTICS_DIR,
    PROJECT_ROOT,
    TRAIN_CSV,
    TEST_CSV,
    TRAIN_IMAGES,
    TEST_IMAGES,
    VALID_LABELS,
)

def main():
    start_time = time.time()
    
    METADATA_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    
    train_csv_path = TRAIN_CSV
    test_csv_path = TEST_CSV
    train_images_dir = TRAIN_IMAGES
    test_images_dir = TEST_IMAGES
    
    log_lines = []
    
    def log_print(message=""):
        print(message)
        log_lines.append(message)
        
    log_print("==========================================")
    log_print("APTOS DATASET VERIFICATION")
    log_print("==========================================")
    log_print()
    
    errors = []
    
    # 1. Check CSV existence
    train_csv_found = train_csv_path.exists()
    test_csv_found = test_csv_path.exists()
    
    if not train_csv_found:
        errors.append("train.csv not found")
    if not test_csv_found:
        errors.append("test.csv not found")
        
    # 2. Check image folders existence
    train_images_found = train_images_dir.exists() and train_images_dir.is_dir()
    test_images_found = test_images_dir.exists() and test_images_dir.is_dir()
    
    if not train_images_found:
        errors.append("train_images folder missing")
    if not test_images_found:
        errors.append("test_images folder missing")
        
    # Read CSV
    df_train = None
    csv_rows = 0
    duplicate_ids_count = 0
    missing_labels_count = 0
    invalid_labels_count = 0
    invalid_labels_list = []
    duplicate_ids_list = []
    ids_are_unique = True
    
    if train_csv_found:
        try:
            df_train = pd.read_csv(train_csv_path)
            csv_rows = len(df_train)
            
            # Check required columns
            required_cols = ["id_code", "diagnosis"]
            for col in required_cols:
                if col not in df_train.columns:
                    errors.append(f"Required column '{col}' missing in train.csv")
            
            if all(col in df_train.columns for col in required_cols):
                # Uniqueness check (confirm explicitly)
                ids_are_unique = len(df_train["id_code"].unique()) == len(df_train)
                if not ids_are_unique:
                    errors.append("ID codes in train.csv are not unique")
                
                # Duplicate IDs
                dup_series = df_train["id_code"][df_train["id_code"].duplicated(keep="first")]
                duplicate_ids_list = dup_series.unique().tolist()
                duplicate_ids_count = len(dup_series)
                
                # Check for unreadable CSV values (numeric check)
                if not pd.api.types.is_numeric_dtype(df_train["diagnosis"]):
                    errors.append("diagnosis column in train.csv is not numeric")
                
                # Missing Labels
                missing_labels_count = df_train["diagnosis"].isna().sum()
                if missing_labels_count > 0:
                    errors.append(f"{missing_labels_count} missing labels in diagnosis column")
                    
                # Invalid Labels
                invalid_rows = df_train[~df_train["diagnosis"].isin(VALID_LABELS) & df_train["diagnosis"].notna()]
                invalid_labels_count = len(invalid_rows)
                if invalid_labels_count > 0:
                    errors.append(f"{invalid_labels_count} invalid labels in diagnosis column (outside {VALID_LABELS})")
                    for idx, row in invalid_rows.iterrows():
                        invalid_labels_list.append({
                            "id_code": row["id_code"],
                            "invalid_label": row["diagnosis"]
                        })
        except Exception as e:
            errors.append(f"Failed to read/parse train.csv: {e}")
    else:
        ids_are_unique = False
        
    # Verify test set consistency
    test_csv_rows = 0
    test_duplicate_ids_count = 0
    test_missing_images_count = 0
    test_duplicate_ids_list = []
    test_missing_images_list = []
    test_ids_are_unique = True
    
    if test_csv_found:
        try:
            df_test = pd.read_csv(test_csv_path)
            test_csv_rows = len(df_test)
            
            if "id_code" not in df_test.columns:
                errors.append("Required column 'id_code' missing in test.csv")
            else:
                # Uniqueness check for test set
                test_ids_are_unique = len(df_test["id_code"].unique()) == len(df_test)
                if not test_ids_are_unique:
                    errors.append("ID codes in test.csv are not unique")
                    
                # Duplicate IDs in test set
                test_dup_series = df_test["id_code"][df_test["id_code"].duplicated(keep="first")]
                test_duplicate_ids_list = test_dup_series.unique().tolist()
                test_duplicate_ids_count = len(test_dup_series)
                
                # Check test set images
                if test_images_found:
                    test_files = list(test_images_dir.iterdir())
                    test_png_files = [f for f in test_files if f.is_file() and f.suffix.lower() == ".png"]
                    test_dir_files_set = {f.name for f in test_png_files}
                    for idx, row in df_test.iterrows():
                        id_code = row["id_code"]
                        expected_filename = f"{id_code}.png"
                        if expected_filename not in test_dir_files_set:
                            test_missing_images_list.append(id_code)
                    test_missing_images_count = len(test_missing_images_list)
                    if test_missing_images_count > 0:
                        errors.append(f"{test_missing_images_count} images missing from test_images directory")
        except Exception as e:
            errors.append(f"Failed to read/parse test.csv: {e}")
    else:
        test_ids_are_unique = False
            
    # Count images in folders
    train_images_count = 0
    test_images_count = 0
    duplicate_filenames_count = 0
    missing_images_count = 0
    corrupted_images_count = 0
    rgb_images_count = 0
    non_png_images_count = 0
    
    missing_images_list = []
    corrupted_images_list = []
    image_sizes = []
    
    if train_images_found:
        train_files = list(train_images_dir.iterdir())
        png_files = [f for f in train_files if f.is_file() and f.suffix.lower() == ".png"]
        train_images_count = len(png_files)
        
        # Check non-png files
        non_png_files = [f for f in train_files if f.is_file() and f.suffix.lower() != ".png"]
        non_png_images_count = len(non_png_files)
        if non_png_images_count > 0:
            errors.append(f"Found {non_png_images_count} files with non-PNG extensions in train_images")
            
        # Case-insensitive duplicate filenames check
        filename_counts = {}
        for f in png_files:
            name_lower = f.name.lower()
            filename_counts[name_lower] = filename_counts.get(name_lower, 0) + 1
        duplicate_filenames = [name for name, count in filename_counts.items() if count > 1]
        duplicate_filenames_count = len(duplicate_filenames)
        if duplicate_filenames_count > 0:
            errors.append(f"{duplicate_filenames_count} duplicate filenames found")
            
        # Match with CSV to find missing images
        if df_train is not None and "id_code" in df_train.columns:
            dir_files_set = {f.name for f in png_files}
            for idx, row in df_train.iterrows():
                id_code = row["id_code"]
                expected_filename = f"{id_code}.png"
                if expected_filename not in dir_files_set:
                    missing_images_list.append(id_code)
            missing_images_count = len(missing_images_list)
            if missing_images_count > 0:
                errors.append(f"{missing_images_count} images missing from train_images directory")
                
        # Completely verify all PNG images (mode, corruption, size)
        for f in png_files:
            try:
                with Image.open(f) as img:
                    w, h = img.size
                    image_sizes.append({"filename": f.name, "width": w, "height": h})
                    if w <= 0 or h <= 0:
                        errors.append(f"Image {f.name} has invalid size: {w}x{h}")
                    if img.mode == "RGB":
                        rgb_images_count += 1
                    else:
                        errors.append(f"Image {f.name} has non-RGB mode: {img.mode}")
                    img.verify()
            except Exception:
                corrupted_images_list.append(f.name)
        
        corrupted_images_count = len(corrupted_images_list)
        if corrupted_images_count > 0:
            errors.append(f"{corrupted_images_count} corrupted images found")
            
    if test_images_found:
        test_files = [f for f in test_images_dir.iterdir() if f.is_file() and f.suffix.lower() == ".png"]
        test_images_count = len(test_files)
        
    # Write CSV output files
    df_missing = pd.DataFrame(missing_images_list, columns=["id_code"])
    df_missing.to_csv(METADATA_VALIDATION_DIR / "missing_images.csv", index=False)
    
    df_corrupted = pd.DataFrame(corrupted_images_list, columns=["filename"])
    df_corrupted.to_csv(METADATA_VALIDATION_DIR / "corrupted_images.csv", index=False)
    
    df_duplicates = pd.DataFrame(duplicate_ids_list, columns=["id_code"])
    df_duplicates.to_csv(METADATA_VALIDATION_DIR / "duplicate_ids.csv", index=False)
    
    df_invalid = pd.DataFrame(invalid_labels_list, columns=["id_code", "invalid_label"])
    df_invalid.to_csv(METADATA_VALIDATION_DIR / "invalid_labels.csv", index=False)
    
    df_sizes = pd.DataFrame(image_sizes)
    if df_sizes.empty:
        df_sizes = pd.DataFrame(columns=["filename", "width", "height"])
    df_sizes.to_csv(METADATA_STATISTICS_DIR / "image_sizes.csv", index=False)
    
    # Save test set diagnostic files if issues exist
    df_test_missing = pd.DataFrame(test_missing_images_list, columns=["id_code"])
    df_test_missing.to_csv(METADATA_VALIDATION_DIR / "missing_test_images.csv", index=False)
    
    df_test_duplicates = pd.DataFrame(test_duplicate_ids_list, columns=["id_code"])
    df_test_duplicates.to_csv(METADATA_VALIDATION_DIR / "duplicate_test_ids.csv", index=False)
    
    # Calculate image statistics
    min_width = max_width = min_height = max_height = 0
    mean_width = mean_height = 0.0
    if image_sizes:
        widths = [x["width"] for x in image_sizes]
        heights = [x["height"] for x in image_sizes]
        min_width = int(min(widths))
        max_width = int(max(widths))
        min_height = int(min(heights))
        max_height = int(max(heights))
        mean_width = float(sum(widths) / len(widths))
        mean_height = float(sum(heights) / len(heights))
    
    # Print summary tables
    log_print("CSV")
    log_print("------------------------")
    log_print(f"{'Train CSV':<21}{'PASS' if train_csv_found else 'FAIL'}")
    log_print(f"{'Test CSV':<21}{'PASS' if test_csv_found else 'FAIL'}")
    log_print()
    
    log_print("Folders")
    log_print("------------------------")
    log_print(f"{'Train Images':<21}{'PASS' if train_images_found else 'FAIL'}")
    log_print(f"{'Test Images':<21}{'PASS' if test_images_found else 'FAIL'}")
    log_print()
    
    log_print("Integrity")
    log_print("------------------------")
    log_print(f"{'CSV Rows':<21}{csv_rows}")
    log_print(f"{'Images':<21}{train_images_count}")
    log_print(f"{'Missing Images':<21}{missing_images_count}")
    log_print(f"{'Duplicate IDs':<21}{duplicate_ids_count}")
    log_print(f"{'Corrupted Images':<21}{corrupted_images_count}")
    log_print()
    
    log_print("Labels")
    log_print("------------------------")
    log_print(f"{'Missing Labels':<21}{missing_labels_count}")
    log_print(f"{'Invalid Labels':<21}{invalid_labels_count}")
    log_print()
    
    verification_passed = len(errors) == 0
    
    log_print("Overall Status")
    log_print()
    log_print("PASS" if verification_passed else "FAIL")
    log_print()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    log_print(f"Verification completed in {elapsed_time:.2f} seconds")
    
    # Save log file
    with open(PROJECT_ROOT / "logs" / "verification.log", "w") as log_file:
        log_file.write("\n".join(log_lines) + "\n")
        
    # JSON Report
    report_data = {
        "dataset": "APTOS2019",
        "verification_date": datetime.now().strftime("%Y-%m-%d"),
        "csv_rows": int(csv_rows),
        "train_images": int(train_images_count),
        "test_images": int(test_images_count),
        "missing_images": int(missing_images_count),
        "duplicate_ids": int(duplicate_ids_count),
        "duplicate_filenames": int(duplicate_filenames_count),
        "missing_labels": int(missing_labels_count),
        "invalid_labels": int(invalid_labels_count),
        "corrupted_images": int(corrupted_images_count),
        "rgb_images": int(rgb_images_count),
        "verification_passed": bool(verification_passed),
        "image_statistics": {
            "min_width": min_width,
            "max_width": max_width,
            "min_height": min_height,
            "max_height": max_height,
            "mean_width": round(mean_width, 2),
            "mean_height": round(mean_height, 2)
        }
    }
    
    with open(METADATA_VALIDATION_DIR / "verification_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
        
    if not verification_passed:
        print()
        print("Collected Errors:")
        for err in errors:
            print(f"✗ {err}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
