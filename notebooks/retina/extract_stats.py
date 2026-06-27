import os
import sys
import time
import json
import logging
import hashlib
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import (
    METADATA_DIR,
    TRAIN_IMAGES,
    IMAGE_STATISTICS_CSV,
    EDA_STATISTICS_CSV,
    EDA_STATISTICS_PARQUET,
    EDA_SUMMARY_JSON,
    DARK_IMAGES_CSV,
    BRIGHT_IMAGES_CSV,
    BLURRY_IMAGES_CSV,
    SMALL_IMAGES_CSV,
    LARGE_IMAGES_CSV,
    RESULTS_DIR,
    REPORTS_DIR,
    RETINA_RESULTS_DIR,
    RETINA_FIGURES_DIR,
    RETINA_RESEARCH_DIR,
    EDA_VERSION,
    OUTLIER_PERCENTILE,
    MAX_ANALYSIS_DIM
)

# Import tqdm with a custom fallback function if not installed
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

def setup_logging() -> None:
    """Sets up unified logging to both console and a log file in the metadata directory."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    log_file = METADATA_DIR / "eda.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

def create_output_directories() -> None:
    """Ensures all output results, reports, and figure directories exist."""
    for directory in [RESULTS_DIR, REPORTS_DIR, RETINA_RESULTS_DIR, RETINA_FIGURES_DIR, RETINA_RESEARCH_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    logging.info("Created and verified all output directory structures.")

def compute_brightness(img_gray: np.ndarray) -> float:
    """Computes grayscale brightness as the mean intensity."""
    return float(np.mean(img_gray))

def compute_blur(img_gray: np.ndarray) -> float:
    """Computes blur score using the variance of the Laplacian (Blur Estimate)."""
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

def process_single_image(
    img_id: str, 
    filename: str, 
    train_images_dir: Path, 
    max_analysis_dim: int
) -> Tuple[str, float, float, Tuple[np.ndarray, np.ndarray, int], str, str]:
    """
    Loads a single image, verifies channels, resizes, and extracts brightness, blur, and RGB stats.
    Returns status: 'success', 'missing', 'corrupted', or 'invalid_channels', along with a pixel SHA256 hash.
    """
    img_path = train_images_dir / filename
    if not img_path.exists():
        return img_id, np.nan, np.nan, (np.zeros(3), np.zeros(3), 0), "missing", ""
        
    # Load image with OpenCV (BGR format)
    img_bgr = cv2.imread(str(img_path))
    
    # Verify read
    if img_bgr is None:
        return img_id, np.nan, np.nan, (np.zeros(3), np.zeros(3), 0), "corrupted", ""
        
    # Verify shape and channels
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        return img_id, np.nan, np.nan, (np.zeros(3), np.zeros(3), 0), "invalid_channels", ""
        
    # Compute exact pixel checksum for duplicate detection (on raw loaded pixels to avoid scaling artifacts)
    pixel_hash = hashlib.sha256(img_bgr.tobytes()).hexdigest()
    
    # Get dimensions
    h, w, _ = img_bgr.shape
    
    # Downscale in-memory for fast analysis
    if max(h, w) > max_analysis_dim:
        scale = max_analysis_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_bgr_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_bgr_resized = img_bgr
        
    # Convert to Grayscale for brightness and blur
    img_gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
    
    brightness = compute_brightness(img_gray)
    blur = compute_blur(img_gray)
    
    # Channel stats accumulation (BGR to RGB)
    img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float64) / 255.0
    hr, wr, _ = img_normalized.shape
    n_pixels = hr * wr
    channel_sum = img_normalized.sum(axis=(0, 1))
    channel_sq_sum = (img_normalized ** 2).sum(axis=(0, 1))
    
    return img_id, brightness, blur, (channel_sum, channel_sq_sum, n_pixels), "success", pixel_hash

def compute_dataset_fingerprint(df: pd.DataFrame, train_images_dir: Path) -> str:
    """Computes a strong SHA-256 fingerprint based on filename, exact filesize_bytes, width, height, and label."""
    hasher = hashlib.sha256()
    # Sort by id_code for determinism
    sorted_df = df.sort_values("id_code")
    for _, row in sorted_df.iterrows():
        img_path = train_images_dir / row["filename"]
        filesize_bytes = os.path.getsize(img_path) if img_path.exists() else 0
        entry = f"{row['filename']}:{filesize_bytes}:{row['width']}:{row['height']}:{row['diagnosis']}\n".encode("utf-8")
        hasher.update(entry)
    return hasher.hexdigest()

def compute_thresholds(df: pd.DataFrame) -> Tuple[float, float, float, float, float]:
    """Computes statistical thresholds based on outlier percentile configuration."""
    brightness_5pct = float(df["brightness_score"].quantile(OUTLIER_PERCENTILE))
    brightness_95pct = float(df["brightness_score"].quantile(1.0 - OUTLIER_PERCENTILE))
    blur_5pct = float(df["blur_score"].quantile(OUTLIER_PERCENTILE))
    filesize_5pct = float(df["filesize_kb"].quantile(OUTLIER_PERCENTILE))
    filesize_95pct = float(df["filesize_kb"].quantile(1.0 - OUTLIER_PERCENTILE))
    
    return brightness_5pct, brightness_95pct, blur_5pct, filesize_5pct, filesize_95pct

def save_outliers(df: pd.DataFrame, f_5pct: float, f_95pct: float) -> None:
    """Saves outlier tables based on statistical thresholds, including top 20 outliers."""
    # Standard outliers
    df[df["is_dark"]].to_csv(DARK_IMAGES_CSV, index=False)
    df[df["is_bright"]].to_csv(BRIGHT_IMAGES_CSV, index=False)
    df[df["is_blurry"]].to_csv(BLURRY_IMAGES_CSV, index=False)
    df[df["filesize_kb"] < f_5pct].to_csv(SMALL_IMAGES_CSV, index=False)
    df[df["filesize_kb"] > f_95pct].to_csv(LARGE_IMAGES_CSV, index=False)
    
    # Top 20 Outlier lists
    df.sort_values("brightness_score", ascending=True).head(20).to_csv(METADATA_DIR / "top_20_darkest.csv", index=False)
    df.sort_values("blur_score", ascending=True).head(20).to_csv(METADATA_DIR / "top_20_blurriest.csv", index=False)
    df.sort_values("filesize_kb", ascending=False).head(20).to_csv(METADATA_DIR / "top_20_largest.csv", index=False)
    df.sort_values("filesize_kb", ascending=True).head(20).to_csv(METADATA_DIR / "top_20_smallest.csv", index=False)
    logging.info("Saved all standard and Top 20 outlier tables.")

def write_summary(
    dataset_hash: str, 
    total_imgs: int, 
    counters: Dict[str, int], 
    rgb_mean: List[float], 
    rgb_std: List[float], 
    class_rgb: Dict[str, Dict[str, List[float]]],
    thresholds: Tuple[float, float, float, float, float], 
    imbalance_stats: Dict[str, Any],
    quality_stats: Dict[str, float],
    duration: float
) -> None:
    """Saves a JSON summary containing schema, versions, imbalance statistics, and rounded statistics."""
    summary_data = {
        "schema_version": "1.0",
        "eda_version": EDA_VERSION,
        "phase": "EDA",
        "dataset": "APTOS2019",
        "dataset_fingerprint": dataset_hash,
        "total_training_images": total_imgs,
        "missing_images": counters["missing"],
        "corrupted_images": counters["corrupted"],
        "invalid_channel_images": counters["invalid_channels"],
        "quality_metrics": {
            "dark_images_percentage": round(quality_stats["dark_pct"], 4),
            "bright_images_percentage": round(quality_stats["bright_pct"], 4),
            "blurry_images_percentage": round(quality_stats["blurry_pct"], 4)
        },
        "rgb_mean": [round(x, 6) for x in rgb_mean],
        "rgb_std": [round(x, 6) for x in rgb_std],
        "class_rgb_statistics": class_rgb,
        "class_imbalance": imbalance_stats,
        "thresholds": {
            "brightness_5pct": round(thresholds[0], 6),
            "brightness_95pct": round(thresholds[1], 6),
            "blur_5pct": round(thresholds[2], 6),
            "filesize_5pct": round(thresholds[3], 6),
            "filesize_95pct": round(thresholds[4], 6)
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "opencv_version": cv2.__version__
        },
        "execution_time_seconds": round(duration, 2)
    }
    
    with open(EDA_SUMMARY_JSON, "w") as f:
        json.dump(summary_data, f, indent=2)

def main() -> None:
    setup_logging()
    logging.info("==================================================")
    logging.info(f"STARTING CONCURRENT EDA FEATURE EXTRACTION (v{EDA_VERSION})")
    logging.info("==================================================")
    
    start_time = time.perf_counter()
    
    # Ensure folders exist
    create_output_directories()
    
    # Load base metadata using config path
    if not IMAGE_STATISTICS_CSV.exists():
        raise FileNotFoundError(f"Base metadata not found: {IMAGE_STATISTICS_CSV}. Please run verify_dataset.py first.")
        
    df = pd.read_csv(IMAGE_STATISTICS_CSV)
    logging.info(f"Loaded base metadata containing {len(df)} images.")
    
    # Initialize structures for parallel results
    results: Dict[str, Tuple[float, float]] = {}
    total_pixels = 0
    total_channel_sum = np.zeros(3, dtype=np.float64) # R, G, B
    total_channel_sq_sum = np.zeros(3, dtype=np.float64) # R, G, B
    
    # Structures for class-wise RGB stats
    class_pixels = {c: 0 for c in range(5)}
    class_sum = {c: np.zeros(3, dtype=np.float64) for c in range(5)}
    class_sq_sum = {c: np.zeros(3, dtype=np.float64) for c in range(5)}
    
    counters: Dict[str, int] = {
        "missing": 0,
        "corrupted": 0,
        "invalid_channels": 0,
    }
    
    # Hash mapping to detect visual duplicates
    pixel_hashes: Dict[str, List[Tuple[str, str]]] = {} # hash -> list of (id_code, filename)
    
    max_workers = min(16, (os.cpu_count() or 4) * 2)
    logging.info(f"Processing images using ThreadPoolExecutor ({max_workers} workers)...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_image, 
                row["id_code"], 
                row["filename"], 
                TRAIN_IMAGES, 
                MAX_ANALYSIS_DIM
            ): (row["id_code"], row["diagnosis"])
            for _, row in df.iterrows()
        }
        
        # Monitor progress with tqdm
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Features"):
            img_id, diag = futures[future]
            try:
                res_id, brightness, blur, rgb_stats, status, pixel_hash = future.result()
                results[res_id] = (brightness, blur)
                
                if status == "success":
                    channel_sum, channel_sq_sum, n_pixels = rgb_stats
                    total_pixels += n_pixels
                    total_channel_sum += channel_sum
                    total_channel_sq_sum += channel_sq_sum
                    
                    # Accumulate class-wise stats
                    class_pixels[diag] += n_pixels
                    class_sum[diag] += channel_sum
                    class_sq_sum[diag] += channel_sq_sum
                    
                    # Save pixel hash for duplicate checks
                    if pixel_hash:
                        # Find corresponding row to get filename
                        filename_val = df[df["id_code"] == res_id]["filename"].values[0]
                        pixel_hashes.setdefault(pixel_hash, []).append((res_id, filename_val))
                else:
                    counters[status] += 1
            except Exception as e:
                logging.error(f"Failed to process image {img_id}: {e}")
                results[img_id] = (np.nan, np.nan)
                counters["corrupted"] += 1
                
    # Map parallel results back to original DataFrame
    df["brightness_score"] = df["id_code"].map(lambda x: results[x][0]).astype(np.float32)
    df["blur_score"] = df["id_code"].map(lambda x: results[x][1]).astype(np.float32)
    
    # Handle missing values
    missing_brightness = df["brightness_score"].isna().sum()
    missing_blur = df["blur_score"].isna().sum()
    if missing_brightness > 0:
        df["brightness_score"] = df["brightness_score"].fillna(df["brightness_score"].median())
    if missing_blur > 0:
        df["blur_score"] = df["blur_score"].fillna(df["blur_score"].median())
        
    # 4. Compute Statistical Thresholds
    b_5pct, b_95pct, bl_5pct, f_5pct, f_95pct = compute_thresholds(df)
    
    # 5. Set Quality Flags
    df["is_dark"] = df["brightness_score"] < b_5pct
    df["is_bright"] = df["brightness_score"] > b_95pct
    df["is_blurry"] = df["blur_score"] < bl_5pct
    
    # Compute quality flag percentages
    quality_stats = {
        "dark_pct": float(df["is_dark"].mean() * 100),
        "bright_pct": float(df["is_bright"].mean() * 100),
        "blurry_pct": float(df["is_blurry"].mean() * 100)
    }
    
    # 6. Compute continuous Quality Score Q in [0, 1]
    ref_blur = float(df["blur_score"].quantile(0.75))
    if ref_blur == 0:
        ref_blur = 1.0
        
    q_brightness = np.clip(1.0 - np.abs(df["brightness_score"] - 120.0) / 120.0, 0.0, 1.0)
    q_blur = np.clip(df["blur_score"] / ref_blur, 0.0, 1.0)
    q_resolution = np.clip((df["width"] * df["height"]) / (1024.0 * 1024.0), 0.0, 1.0)
    
    df["quality_score"] = (0.4 * q_brightness + 0.3 * q_blur + 0.3 * q_resolution).astype(np.float32)
    logging.info(f"Computed continuous Quality Score Q (mean: {df['quality_score'].mean():.4f}).")
    
    # Save CSV of quality flags with specific reason tag
    logging.info("Generating quality_flags.csv...")
    reasons_list = []
    for _, row in df.iterrows():
        reasons = []
        if row["is_dark"]: reasons.append("dark")
        if row["is_bright"]: reasons.append("bright")
        if row["is_blurry"]: reasons.append("blurry")
        reasons_list.append(",".join(reasons) if reasons else "pass")
    
    q_flags_df = df.copy()
    q_flags_df["flag_reason"] = reasons_list
    q_flags_df = q_flags_df[["id_code", "filename", "diagnosis", "brightness_score", "blur_score", "is_dark", "is_bright", "is_blurry", "quality_score", "flag_reason"]]
    q_flags_df.to_csv(METADATA_DIR / "quality_flags.csv", index=False)
    
    # 7. Save Outlier Lists (including top 20 lists)
    save_outliers(df, f_5pct, f_95pct)
    
    # 8. Visual Duplicate Detection Report
    logging.info("Scanning for visual duplicate images...")
    duplicate_rows = []
    for phash, occurrences in pixel_hashes.items():
        if len(occurrences) > 1:
            # We found duplicate occurrences
            for idx1 in range(len(occurrences)):
                for idx2 in range(idx1 + 1, len(occurrences)):
                    id1, fn1 = occurrences[idx1]
                    id2, fn2 = occurrences[idx2]
                    duplicate_rows.append({
                        "id_code_1": id1,
                        "filename_1": fn1,
                        "id_code_2": id2,
                        "filename_2": fn2,
                        "pixel_hash": phash
                    })
    if duplicate_rows:
        dup_df = pd.DataFrame(duplicate_rows)
        dup_df.to_csv(METADATA_DIR / "duplicate_images.csv", index=False)
        logging.info(f"Detected {len(dup_df)} duplicate pairs. Saved to duplicate_images.csv.")
    else:
        # Save empty placeholder with headers
        pd.DataFrame(columns=["id_code_1", "filename_1", "id_code_2", "filename_2", "pixel_hash"]).to_csv(METADATA_DIR / "duplicate_images.csv", index=False)
        logging.info("No visual duplicate images detected.")
        
    # 9. Compute Class-wise RGB mean & std
    class_rgb_stats = {}
    for c in range(5):
        c_pixels = class_pixels[c]
        if c_pixels > 0:
            c_mean = (class_sum[c] / c_pixels).tolist()
            c_var = (class_sq_sum[c] / c_pixels) - (class_sum[c] / c_pixels) ** 2
            c_std = np.sqrt(np.clip(c_var, 0, None)).tolist()
            class_rgb_stats[f"Class_{c}"] = {
                "mean": [round(x, 6) for x in c_mean],
                "std": [round(x, 6) for x in c_std]
            }
        else:
            class_rgb_stats[f"Class_{c}"] = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
            
    # 10. Class Imbalance Analysis & Recommendation
    class_counts = df["diagnosis"].value_counts().sort_index()
    majority_count = int(class_counts.max())
    minority_count = int(class_counts.min())
    imbalance_ratio = float(majority_count / minority_count)
    
    recommended_loss = "CrossEntropyLoss"
    if imbalance_ratio > 1.5:
        recommended_loss = "WeightedCrossEntropyLoss"  # focal loss is also a valid recommendation
        
    imbalance_stats = {
        "imbalance_ratio": round(imbalance_ratio, 4),
        "majority_class_count": majority_count,
        "minority_class_count": minority_count,
        "recommended_loss_function": recommended_loss
    }
    
    # 11. Compute Global Dataset RGB Mean & Standard Deviation
    if total_pixels > 0:
        rgb_mean = (total_channel_sum / total_pixels).tolist()
        rgb_var = (total_channel_sq_sum / total_pixels) - (total_channel_sum / total_pixels) ** 2
        rgb_std = np.sqrt(np.clip(rgb_var, 0, None)).tolist()
    else:
        rgb_mean = [0.0, 0.0, 0.0]
        rgb_std = [1.0, 1.0, 1.0]
        
    # 12. Compute Dataset Reproducibility Fingerprint (Stronger hash incorporating bytes & diagnosis)
    dataset_hash = compute_dataset_fingerprint(df, TRAIN_IMAGES)
    
    # 13. Track Execution time
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # 14. Prepare Summary JSON
    write_summary(
        dataset_hash, len(df), counters, rgb_mean, rgb_std, class_rgb_stats,
        (b_5pct, b_95pct, bl_5pct, f_5pct, f_95pct), imbalance_stats, quality_stats, duration
    )
    
    # 15. Serialization to CSV
    df.to_csv(EDA_STATISTICS_CSV, index=False)
    logging.info(f"Saved complete statistics CSV to {EDA_STATISTICS_CSV}")
    
    # 16. Serialization to Parquet
    try:
        import pyarrow
        df.to_parquet(EDA_STATISTICS_PARQUET, index=False)
        logging.info(f"Saved complete statistics Parquet to {EDA_STATISTICS_PARQUET}")
    except ImportError:
        logging.info("pyarrow is not installed. Skipping Parquet export.")
        
    logging.info("==================================================")
    logging.info(f"FEATURE EXTRACTION COMPLETED IN {duration:.2f} SECONDS")
    logging.info(f"Summary metrics: missing={counters['missing']}, corrupted={counters['corrupted']}, invalid_channels={counters['invalid_channels']}")
    logging.info("==================================================")

if __name__ == "__main__":
    main()
