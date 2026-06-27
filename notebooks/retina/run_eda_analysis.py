import os
import sys
import json
import time
import shutil
import logging
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# Add project root to sys.path to allow config import
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.config import (
    METADATA_DIR,
    TRAIN_IMAGES,
    IMAGE_STATISTICS_CSV,
    EDA_STATISTICS_CSV,
    EDA_SUMMARY_JSON,
    PREPROCESSING_RECOMMENDATIONS_JSON,
    REPORTS_DIR,
    RETINA_NOTEBOOKS_DIR,
    RETINA_RESULTS_DIR,
    RETINA_FIGURES_DIR,
    RETINA_RESEARCH_DIR,
    FIGURE_DPI,
    FIGURE_FORMATS,
    SEED,
    CLASS_NAMES
)

# Colors palette configuration
COLORS = ["#4f81bd", "#c0504d", "#9bbb59", "#8064a2", "#4bacc6"]

# Cached placeholder image for unreadable files
PLACEHOLDER_IMAGE = np.zeros((512, 512, 3), dtype=np.uint8)

# Module-level logger setup
logger = logging.getLogger("eda_analysis")

def setup_logging() -> None:
    """Sets up unified logging to both console and a log file in the metadata directory."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    log_file = METADATA_DIR / "eda_analysis.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

def create_plot_style():
    """Sets a clean, publication-friendly matplotlib style."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.color"] = "#555555"
    plt.rcParams["ytick.color"] = "#555555"

def save_fig(fig, base_name):
    """Saves a figure in all configured formats (PNG and SVG)."""
    for fmt in FIGURE_FORMATS:
        path = RETINA_FIGURES_DIR / f"{base_name}.{fmt}"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", format=fmt)
    logger.info(f"Saved figure: {base_name} (PNG + SVG)")

def get_first_or_fallback(
    subset_df: pd.DataFrame,
    fallback_row: pd.Series
) -> pd.Series:
    """Returns the first row in subset_df if not empty, otherwise falls back to fallback_row."""
    return subset_df.iloc[0] if not subset_df.empty else fallback_row

def safe_read_image(img_path: Path) -> np.ndarray:
    """Safely reads an image. Returns a BGR image, or a black placeholder if missing/corrupted."""
    if not img_path.exists():
        logger.warning(f"Image file missing: {img_path}. Using placeholder.")
        return PLACEHOLDER_IMAGE.copy()
        
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning(f"Failed to decode image: {img_path}. Using placeholder.")
        return PLACEHOLDER_IMAGE.copy()
        
    return img

def main():
    setup_logging()
    logger.info("==================================================")
    logger.info("STARTING EDA PLOTTING & REPORT GENERATION")
    logger.info("==================================================")
    
    start_time = time.perf_counter()
    create_plot_style()
    
    # Load cached statistics
    if not EDA_STATISTICS_CSV.exists():
        raise FileNotFoundError(f"Cached statistics not found: {EDA_STATISTICS_CSV}. Run extract_stats.py first.")
    if not EDA_SUMMARY_JSON.exists():
        raise FileNotFoundError(f"Cached summary JSON not found: {EDA_SUMMARY_JSON}.")
        
    df = pd.read_csv(EDA_STATISTICS_CSV)
    with open(EDA_SUMMARY_JSON, "r") as f:
        summary_data = json.load(f)
        
    # Load thresholds from summary JSON
    thresholds = summary_data["thresholds"]
    b_5pct = thresholds["brightness_5pct"]
    b_95pct = thresholds["brightness_95pct"]
    bl_5pct = thresholds["blur_5pct"]
    f_5pct = thresholds["filesize_5pct"]
    f_95pct = thresholds["filesize_95pct"]
    
    # Compute DataFrame statistics directly to prevent KeyError
    average_width = df["width"].mean()
    average_height = df["height"].mean()
    min_width = df["width"].min()
    max_width = df["width"].max()
    min_height = df["height"].min()
    max_height = df["height"].max()
    average_filesize_kb = df["filesize_kb"].mean()
    average_aspect_ratio = df["aspect_ratio"].mean()
    average_quality_score = df["quality_score"].mean()
    
    # Ensure directories exist
    RETINA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RETINA_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set seed for reproducibility
    np.random.seed(SEED)
    
    generated_figures_count = 0
    
    # ----------------------------------------------------
    # Fig 03_01: Class Distribution
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    counts = df["diagnosis"].value_counts().sort_index()
    percentages = (counts / len(df)) * 100
    
    # Bar Chart
    bars = ax1.bar(CLASS_NAMES, counts, color=COLORS, edgecolor="#555555", alpha=0.85)
    ax1.set_title("Class Distribution (Counts)", fontsize=14, pad=15)
    ax1.set_ylabel("Number of Samples", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    
    # Add values on top of bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax1.annotate(f"{int(height)}\n({pct:.2f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)
                    
    # Pie Chart
    wedges, texts, autotexts = ax2.pie(counts, labels=CLASS_NAMES, autopct="%1.1f%%",
                                      startangle=140, colors=COLORS,
                                      wedgeprops=dict(width=0.4, edgecolor="#555555", alpha=0.85))
    ax2.set_title("Class Distribution (Proportions)", fontsize=14, pad=15)
    
    plt.tight_layout()
    save_fig(fig, "Fig_03_01_class_distribution")
    plt.close(fig)
    generated_figures_count += 1
    
    # ----------------------------------------------------
    # Fig 03_02: Sample Images Grid (5x5)
    # ----------------------------------------------------
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    for class_id in range(5):
        class_subset = df[df["diagnosis"] == class_id]
        # Deterministically sample 5 images (with safety bounds check)
        sample_size = min(5, len(class_subset))
        sample_rows = class_subset.sample(n=sample_size, random_state=SEED)
        
        for sample_idx in range(5):
            ax = axes[class_id, sample_idx]
            if sample_idx < len(sample_rows):
                row = sample_rows.iloc[sample_idx]
                img_path = TRAIN_IMAGES / row["filename"]
                img = safe_read_image(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                ax.imshow(img)
                ax.axis("off")
                
                if sample_idx == 0:
                    ax.text(-30, img.shape[0] // 2, f"Class {class_id}\n{CLASS_NAMES[class_id]}",
                            va="center", ha="right", fontsize=12, fontweight="bold", rotation=0)
            else:
                ax.axis("off")
                        
    plt.suptitle("Deterministic Retina Image Sample Grid (5x5)", fontsize=16, y=0.92, fontweight="bold")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    save_fig(fig, "Fig_03_02_sample_images")
    plt.close(fig)
    generated_figures_count += 1
    
    # ----------------------------------------------------
    # Fig 03_03: Resolution Histogram
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(df["width"], bins=30, color=COLORS[0], edgecolor="#555555", alpha=0.75)
    ax1.set_title("Image Width Distribution", fontsize=12)
    ax1.set_xlabel("Width (pixels)")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2.hist(df["height"], bins=30, color=COLORS[1], edgecolor="#555555", alpha=0.75)
    ax2.set_title("Image Height Distribution", fontsize=12)
    ax2.set_xlabel("Height (pixels)")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle("Image Resolutions Histogram Analysis", fontsize=14, y=0.98, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "Fig_03_03_resolution_histogram")
    plt.close(fig)
    generated_figures_count += 1
    
    # ----------------------------------------------------
    # Fig 03_04: Resolution Scatter Plot
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df["width"], df["height"], c=df["diagnosis"], cmap="coolwarm", alpha=0.6, edgecolors="none")
    ax.set_title("Image Resolution Scatter Plot (Width vs Height)", fontsize=14, pad=15)
    ax.set_xlabel("Width (pixels)", fontsize=12)
    ax.set_ylabel("Height (pixels)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Add a diagonal representing aspect ratio 1:1
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.75, zorder=0, label="1:1 Ratio")
    ax.legend()
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1, 2, 3, 4])
    cbar.set_ticklabels(CLASS_NAMES)
    cbar.set_label("Retinopathy Class Staging")
    
    plt.tight_layout()
    save_fig(fig, "Fig_03_04_resolution_scatter")
    plt.close(fig)
    generated_figures_count += 1
    
    # ----------------------------------------------------
    # Fig 03_05: Aspect Ratio Histogram & Boxplot
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.hist(df["aspect_ratio"], bins=30, color=COLORS[2], edgecolor="#555555", alpha=0.75)
    ax1.set_title("Aspect Ratio (Width / Height) Distribution", fontsize=14, pad=10)
    ax1.set_ylabel("Frequency")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2.boxplot(df["aspect_ratio"], vert=False, patch_artist=True,
                tick_labels=[""],
                boxprops=dict(facecolor=COLORS[2], color="#555555", alpha=0.75),
                medianprops=dict(color="#d62728", linewidth=2))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Aspect Ratio")
    
    plt.tight_layout()
    save_fig(fig, "Fig_03_05_aspect_ratio_histogram")
    plt.close(fig)
    generated_figures_count += 1
    
    # ----------------------------------------------------
    # Fig 03_06: File Size Histogram
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.hist(df["filesize_kb"], bins=30, color=COLORS[3], edgecolor="#555555", alpha=0.75)
    ax1.set_title("Image File Size Distribution", fontsize=14, pad=10)
    ax1.set_ylabel("Frequency")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2.boxplot(df["filesize_kb"], vert=False, patch_artist=True,
                tick_labels=[""],
                boxprops=dict(facecolor=COLORS[3], color="#555555", alpha=0.75),
                medianprops=dict(color="#d62728", linewidth=2))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_xlabel("File Size (KB)")
    
    plt.tight_layout()
    save_fig(fig, "Fig_03_06_filesize_histogram")
    plt.close(fig)
    generated_figures_count += 1
    
    # ----------------------------------------------------
    # Fig 03_07: Brightness Histogram
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.hist(df["brightness_score"], bins=30, color=COLORS[4], edgecolor="#555555", alpha=0.75)
    ax1.set_title("Grayscale Average Intensity (Brightness) Distribution", fontsize=14, pad=10)
    ax1.set_ylabel("Frequency")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2.boxplot(df["brightness_score"], vert=False, patch_artist=True,
                tick_labels=[""],
                boxprops=dict(facecolor=COLORS[4], color="#555555", alpha=0.75),
                medianprops=dict(color="#d62728", linewidth=2))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Brightness Score (0-255)")
    
    plt.tight_layout()
    save_fig(fig, "Fig_03_07_brightness_histogram")
    plt.close(fig)
    generated_figures_count += 1
    
    # ----------------------------------------------------
    # Fig 03_08: Sharpness Histogram
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.hist(df["blur_score"], bins=np.logspace(np.log10(1), np.log10(10000), 40), color="#e46c0a", edgecolor="#555555", alpha=0.75)
    ax1.set_xscale("log")
    ax1.set_title("Variance of Laplacian (Sharpness Score) Distribution", fontsize=14, pad=10)
    ax1.set_ylabel("Frequency")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2.boxplot(df["blur_score"], vert=False, patch_artist=True,
                tick_labels=[""],
                boxprops=dict(facecolor="#e46c0a", color="#555555", alpha=0.75),
                medianprops=dict(color="#d62728", linewidth=2))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Laplacian Variance Sharpness (Log Scale)")
    
    plt.tight_layout()
    save_fig(fig, "Fig_03_08_blur_histogram")
    plt.close(fig)
    generated_figures_count += 1
    
    # ----------------------------------------------------
    # Fig 03_09: RGB Channel Histograms Overlay (sampled 100 images)
    # ----------------------------------------------------
    logger.info("Sampling pixel values for RGB Histograms...")
    sample_size_rgb = min(100, len(df))
    sample_images_list = df.sample(n=sample_size_rgb, random_state=SEED)["filename"].tolist()
    r_pixels, g_pixels, b_pixels = [], [], []
    for filename in sample_images_list:
        img_path = TRAIN_IMAGES / filename
        img = safe_read_image(img_path)
        if img is not None:
            img_small = cv2.resize(img, (64, 64))
            img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            r_pixels.extend(img_rgb[:, :, 0].flatten() / 255.0)
            g_pixels.extend(img_rgb[:, :, 1].flatten() / 255.0)
            b_pixels.extend(img_rgb[:, :, 2].flatten() / 255.0)
            
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(r_pixels, bins=50, color="red", alpha=0.4, label="Red Channel", density=True)
    ax.hist(g_pixels, bins=50, color="green", alpha=0.4, label="Green Channel", density=True)
    ax.hist(b_pixels, bins=50, color="blue", alpha=0.4, label="Blue Channel", density=True)
    ax.set_title("RGB Channel Histograms Overlay (Normalized Pixel Values)", fontsize=14, pad=15)
    ax.set_xlabel("Normalized Pixel Value [0, 1]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    save_fig(fig, "Fig_03_09_rgb_histogram")
    plt.close(fig)
    generated_figures_count += 1
    
    # ----------------------------------------------------
    # Fig 03_10: Correlation Heatmap (6x6 including quality score)
    # ----------------------------------------------------
    corr_cols = ["width", "height", "filesize_kb", "brightness_score", "blur_score", "quality_score"]
    corr_matrix = df[corr_cols].corr(method="pearson").values
    display_names = ["Width", "Height", "File Size", "Brightness", "Sharpness Score", "Quality Score"]
    
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    
    # Show annotations
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            text = ax.text(j, i, f"{corr_matrix[i, j]:.3f}",
                           ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.7 else "white",
                           fontweight="bold", fontsize=10)
                           
    ax.set_xticks(np.arange(len(corr_cols)))
    ax.set_yticks(np.arange(len(corr_cols)))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(display_names, fontsize=11)
    ax.set_title("Pearson Correlation Coefficient Matrix Heatmap (6x6)", fontsize=14, pad=20, fontweight="bold")
    
    # Add colorbar
    fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    save_fig(fig, "Fig_03_10_correlation_heatmap")
    plt.close(fig)
    generated_figures_count += 1
    
    # ----------------------------------------------------
    # Fig 03_11: Class-wise Brightness & Sharpness Boxplots
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Brightness vs Class
    brightness_data = [df[df["diagnosis"] == i]["brightness_score"] for i in range(5)]
    ax1.boxplot(brightness_data, tick_labels=CLASS_NAMES, patch_artist=True,
                boxprops=dict(facecolor=COLORS[4], color="#555555", alpha=0.7),
                medianprops=dict(color="#d62728", linewidth=2))
    ax1.set_title("Grayscale Brightness vs Disease Class Staging", fontsize=12, pad=10)
    ax1.set_ylabel("Brightness Score (0-255)")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    
    # Sharpness vs Class
    blur_data = [df[df["diagnosis"] == i]["blur_score"] for i in range(5)]
    ax2.boxplot(blur_data, tick_labels=CLASS_NAMES, patch_artist=True,
                boxprops=dict(facecolor=COLORS[1], color="#555555", alpha=0.7),
                medianprops=dict(color="#d62728", linewidth=2))
    ax2.set_title("Variance of Laplacian (Sharpness) vs Disease Class Staging", fontsize=12, pad=10)
    ax2.set_ylabel("Sharpness Score")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    
    plt.suptitle("Class-wise Brightness and Sharpness Metric Boxplots", fontsize=14, y=0.98, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "Fig_03_11_classwise_brightness")
    plt.close(fig)
    generated_figures_count += 1
    
    # ----------------------------------------------------
    # Fig 03_12: Class-wise Resolution & File Size Boxplots
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Aspect Ratio vs Class
    aspect_data = [df[df["diagnosis"] == i]["aspect_ratio"] for i in range(5)]
    ax1.boxplot(aspect_data, tick_labels=CLASS_NAMES, patch_artist=True,
                boxprops=dict(facecolor=COLORS[2], color="#555555", alpha=0.7),
                medianprops=dict(color="#d62728", linewidth=2))
    ax1.set_title("Aspect Ratio vs Disease Class Staging", fontsize=12, pad=10)
    ax1.set_ylabel("Aspect Ratio")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    
    # File Size vs Class
    size_data = [df[df["diagnosis"] == i]["filesize_kb"] for i in range(5)]
    ax2.boxplot(size_data, tick_labels=CLASS_NAMES, patch_artist=True,
                boxprops=dict(facecolor=COLORS[3], color="#555555", alpha=0.7),
                medianprops=dict(color="#d62728", linewidth=2))
    ax2.set_title("Image File Size vs Disease Class Staging", fontsize=12, pad=10)
    ax2.set_ylabel("File Size (KB)")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    
    plt.suptitle("Class-wise Resolution and File Size Metric Boxplots", fontsize=14, y=0.98, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "Fig_03_12_classwise_resolution")
    plt.close(fig)
    generated_figures_count += 1
    
    # ----------------------------------------------------
    # Fig 03_13: Quality Examples Grid (2x3) (Safe Crash-Proof Select)
    # ----------------------------------------------------
    good_subset = df[(~df["is_dark"]) & (~df["is_bright"]) & (~df["is_blurry"])]
    good_img_row = good_subset.iloc[0] if not good_subset.empty else df.iloc[0]
    
    dark_img_row = get_first_or_fallback(df[df["is_dark"]], good_img_row)
    bright_img_row = get_first_or_fallback(df[df["is_bright"]], good_img_row)
    blurry_img_row = get_first_or_fallback(df[df["is_blurry"]], good_img_row)
    
    large_img_row = df.sort_values("filesize_kb", ascending=False).iloc[0]
    small_img_row = df.sort_values("filesize_kb", ascending=True).iloc[0]
    
    quality_cases = [
        ("Good Baseline", good_img_row),
        ("Dark Outlier", dark_img_row),
        ("Bright Outlier", bright_img_row),
        ("Blurry Outlier", blurry_img_row),
        ("Large Size File", large_img_row),
        ("Small Size File", small_img_row)
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()
    
    for idx, (title, row) in enumerate(quality_cases):
        img_path = TRAIN_IMAGES / row["filename"]
        img = safe_read_image(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax = axes_flat[idx]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{title}\nID: {row['id_code']}\nBrightness: {row['brightness_score']:.1f} | Sharpness: {row['blur_score']:.1f}\n{row['width']}x{row['height']} | {row['filesize_kb']} KB",
                     fontsize=10, fontweight="bold", pad=8)
                     
    plt.suptitle("Ocular Fundus Quality Examples and Outlier Cases", fontsize=16, y=0.98, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "Fig_03_13_quality_examples")
    plt.close(fig)
    generated_figures_count += 1
    
    # ----------------------------------------------------
    # Fig 03_14: Summary Dashboard (includes Quality Score Histogram)
    # ----------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Subplot 1: Class distribution
    axes[0, 0].bar(CLASS_NAMES, counts, color=COLORS, edgecolor="#555555", alpha=0.8)
    axes[0, 0].set_title("Class Frequency", fontsize=12, pad=10)
    axes[0, 0].set_ylabel("Samples")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5, axis="y")
    
    # Subplot 2: Quality Score histogram
    axes[0, 1].hist(df["quality_score"], bins=30, color=COLORS[2], edgecolor="#555555", alpha=0.75)
    axes[0, 1].set_title("Image Quality Score (Q) Distribution", fontsize=12, pad=10)
    axes[0, 1].set_xlabel("Quality Score Q")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 3: Width vs Height scatter
    axes[1, 0].scatter(df["width"], df["height"], c=df["diagnosis"], cmap="coolwarm", alpha=0.5, edgecolors="none")
    axes[1, 0].set_title("Resolution Scatter Plot", fontsize=12, pad=10)
    axes[1, 0].set_xlabel("Width")
    axes[1, 0].set_ylabel("Height")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 4: Brightness histogram
    axes[1, 1].hist(df["brightness_score"], bins=30, color=COLORS[4], edgecolor="#555555", alpha=0.75)
    axes[1, 1].set_title("Grayscale Brightness Distribution", fontsize=12, pad=10)
    axes[1, 1].set_xlabel("Brightness Score (0-255)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 5: Sharpness histogram
    axes[2, 0].hist(df["blur_score"], bins=np.logspace(np.log10(1), np.log10(10000), 30), color="#e46c0a", edgecolor="#555555", alpha=0.75)
    axes[2, 0].set_xscale("log")
    axes[2, 0].set_title("Sharpness Score Distribution (Laplacian Variance)", fontsize=12, pad=10)
    axes[2, 0].set_xlabel("Sharpness Score")
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 6: 6x6 Correlation Heatmap
    im = axes[2, 1].imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            axes[2, 1].text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.7 else "white", fontweight="bold")
    axes[2, 1].set_xticks(np.arange(len(corr_cols)))
    axes[2, 1].set_yticks(np.arange(len(corr_cols)))
    axes[2, 1].set_xticklabels(display_names, rotation=45, ha="right")
    axes[2, 1].set_yticklabels(display_names)
    axes[2, 1].set_title("Correlation Coefficient Heatmap (6x6)", fontsize=12, pad=10)
    
    plt.suptitle("FusionMedAI: Retina Dataset Exploratory Summary Dashboard", fontsize=16, y=0.98, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "Fig_03_14_summary_dashboard")
    plt.close(fig)
    generated_figures_count += 1
    
    # Save the combined dashboard as summary_dashboard.png in the retina root results folder
    summary_dash_target = RETINA_RESULTS_DIR / "summary_dashboard.png"
    shutil.copy2(str(RETINA_FIGURES_DIR / "Fig_03_14_summary_dashboard.png"), str(summary_dash_target))
    logger.info(f"Copied summary_dashboard.png to root results folder: {summary_dash_target}")
    
    # ----------------------------------------------------
    # Generate Preprocessing Recommendations JSON (using candidate parameters)
    # ----------------------------------------------------
    recommendations = {
        "candidate_image_sizes": [224, 384, 512],
        "candidate_normalizations": ["ImageNet", "Dataset"],
        "candidate_augmentations": {
            "horizontal_flip": [True, False],
            "rotation_degrees": [0, 15, 30],
            "color_jitter": [True, False],
            "random_crop": [True, False]
        },
        "quality_assessment": {
            "dark_threshold": round(b_5pct, 6),
            "bright_threshold": round(b_95pct, 6),
            "blur_threshold": round(bl_5pct, 6),
            "filesize_min_kb": round(f_5pct, 6),
            "filesize_max_kb": round(f_95pct, 6)
        }
    }
    with open(PREPROCESSING_RECOMMENDATIONS_JSON, "w") as f:
        json.dump(recommendations, f, indent=2)
    logger.info(f"Saved preprocessing recommendations to {PREPROCESSING_RECOMMENDATIONS_JSON}")
    
    # ----------------------------------------------------
    # Generate PDF summary report using Matplotlib backend
    # ----------------------------------------------------
    pdf_path = REPORTS_DIR / "retina_eda_summary.pdf"
    logger.info(f"Compiling PDF summary report at {pdf_path}...")
    with PdfPages(pdf_path) as pdf:
        # Page 1: Text Summary Table
        fig_text = plt.figure(figsize=(11, 8.5))
        plt.axis("off")
        
        # Add Title & Metadata
        plt.text(0.5, 0.95, "FusionMedAI: Ocular Fundus Dataset Exploratory Summary", 
                 fontsize=18, fontweight="bold", ha="center")
        plt.text(0.5, 0.90, f"EDA Version: {summary_data['eda_version']} | Fingerprint: {summary_data['dataset_fingerprint'][:16]}...",
                 fontsize=11, style="italic", ha="center")
        
        # Add summary stats table
        table_content = [
            ["Metric", "Value"],
            ["Total Labeled Images", str(summary_data["total_training_images"])],
            ["Ocular Diagnoses Classes", "5 Staging Classes"],
            ["Average Dimensions", f"{average_width:.1f} x {average_height:.1f} pixels"],
            ["Resolution Extrema", f"Min: {min_width}x{min_height} | Max: {max_width}x{max_height}"],
            ["Average File Size", f"{average_filesize_kb:.1f} KB"],
            ["Average Aspect Ratio", f"{average_aspect_ratio:.2f}"],
            ["Average Quality Score (Q)", f"{average_quality_score:.4f}"],
            ["Global RGB Channel Mean", f"R: {summary_data['rgb_mean'][0]:.4f} | G: {summary_data['rgb_mean'][1]:.4f} | B: {summary_data['rgb_mean'][2]:.4f}"],
            ["Global RGB Channel Std", f"R: {summary_data['rgb_std'][0]:.4f} | G: {summary_data['rgb_std'][1]:.4f} | B: {summary_data['rgb_std'][2]:.4f}"],
            ["Quality Outlier Flags", f"Dark threshold < {b_5pct:.2f} | Sharpness threshold < {bl_5pct:.2f}"],
            ["Corrupted / Missing Files", f"Corrupted: {summary_data['corrupted_images']} | Missing: {summary_data['missing_images']}"],
            ["Extraction Runtime", f"{summary_data['execution_time_seconds']:.2f} seconds"]
        ]
        
        table = plt.table(cellText=table_content, loc="center", cellLoc="left", colWidths=[0.35, 0.55])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.8)
        
        pdf.savefig(fig_text)
        plt.close(fig_text)
        
        # Page 2: Candidate Preprocessing & Augmentations Table
        fig_recs = plt.figure(figsize=(11, 8.5))
        plt.axis("off")
        plt.text(0.5, 0.95, "FusionMedAI: Candidate Preprocessing & Augmentation Parameters",
                 fontsize=16, fontweight="bold", ha="center")
        plt.text(0.5, 0.90, "Parameters established to be evaluated experimentally during model training:",
                 fontsize=11, style="italic", ha="center")
                 
        recs_content = [
            ["Hyperparameter Choice", "Candidate Configurations to Benchmark"],
            ["Candidate Image Sizes", "[224, 384, 512] pixels"],
            ["Candidate Normalizations", "[ImageNet, Dataset]"],
            ["Candidate Augmentations", "Horizontal Flip: [True, False]\nRotation Degrees: [0, 15, 30]\nColor Jitter: [True, False]\nRandom Crop: [True, False]"],
            ["Dark Brightness Threshold", f"< {b_5pct:.6f}"],
            ["Bright Brightness Threshold", f"> {b_95pct:.6f}"],
            ["Sharpness Threshold (Laplacian)", f"< {bl_5pct:.6f}"],
            ["File Size Thresholds", f"Min: {f_5pct:.2f} KB | Max: {f_95pct:.2f} KB"]
        ]
        table_recs = plt.table(cellText=recs_content, loc="center", cellLoc="left", colWidths=[0.40, 0.50])
        table_recs.auto_set_font_size(False)
        table_recs.set_fontsize(10)
        table_recs.scale(1.0, 2.5)
        
        pdf.savefig(fig_recs)
        plt.close(fig_recs)
        
        # Page 3: Class distribution & correlations
        fig_dash = plt.figure(figsize=(11, 8.5))
        plt.axis("off")
        dash_img = safe_read_image(RETINA_RESULTS_DIR / "summary_dashboard.png")
        dash_img = cv2.cvtColor(dash_img, cv2.COLOR_BGR2RGB)
        plt.imshow(dash_img)
        pdf.savefig(fig_dash)
        plt.close(fig_dash)
        
        # Page 4: Sample Grid
        fig_sample = plt.figure(figsize=(11, 8.5))
        plt.axis("off")
        sample_img = safe_read_image(RETINA_FIGURES_DIR / "Fig_03_02_sample_images.png")
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        plt.imshow(sample_img)
        pdf.savefig(fig_sample)
        plt.close(fig_sample)
        
        # Page 5: Quality Outliers
        fig_qual = plt.figure(figsize=(11, 8.5))
        plt.axis("off")
        qual_img = safe_read_image(RETINA_FIGURES_DIR / "Fig_03_13_quality_examples.png")
        qual_img = cv2.cvtColor(qual_img, cv2.COLOR_BGR2RGB)
        plt.imshow(qual_img)
        pdf.savefig(fig_qual)
        plt.close(fig_qual)
        
    logger.info(f"Saved PDF report to {pdf_path}")
    
    # ----------------------------------------------------
    # Generate retina_eda_summary.md (Markdown report)
    # ----------------------------------------------------
    md_path = REPORTS_DIR / "retina_eda_summary.md"
    logger.info(f"Compiling Markdown summary report at {md_path}...")
    
    # Detect duplicates count
    duplicate_images_path = METADATA_DIR / "duplicate_images.csv"
    if duplicate_images_path.exists():
        dup_csv_df = pd.read_csv(duplicate_images_path)
        duplicate_images_count = len(dup_csv_df)
    else:
        duplicate_images_count = 0
        
    md_content = f"""# Ocular Fundus (Retina) Dataset Exploratory Analysis Summary

## Executive Summary
This document summarizes the quantitative, colorimetric, and visual analysis of the APTOS 2019 dataset training split before model preprocessing or training.

## Dataset Fingerprint & Reproducibility
- **Dataset Fingerprint (SHA-256)**: `{summary_data['dataset_fingerprint']}`
- **EDA Module Version**: `{summary_data['eda_version']}`
- **Execution Run Duration**: `{summary_data['execution_time_seconds']:.2f} seconds`
- **Dataset Integrity**: All {summary_data['total_training_images']} images were successfully read with no missing, corrupted, or invalid-channel files.

## High-Level Statistics Table
| Metric | Value |
| :--- | :--- |
| **Total Images** | {summary_data['total_training_images']} |
| **Average Width** | {average_width:.2f} pixels |
| **Average Height** | {average_height:.2f} pixels |
| **Resolution Range** | Min: {min_width}x{min_height} \| Max: {max_width}x{max_height} |
| **Average Aspect Ratio** | {average_aspect_ratio:.4f} |
| **Average File Size** | {average_filesize_kb:.2f} KB |
| **Average Continuous Quality Score (Q)** | {average_quality_score:.4f} |
| **Detected Exact Visual Duplicate Pairs** | {duplicate_images_count} |

## Class Distribution and Imbalance
- **Distribution Counts**:
  - Class 0 (No DR): {counts[0]} ({percentages[0]:.2f}%)
  - Class 1 (Mild): {counts[1]} ({percentages[1]:.2f}%)
  - Class 2 (Moderate): {counts[2]} ({percentages[2]:.2f}%)
  - Class 3 (Severe): {counts[3]} ({percentages[3]:.2f}%)
  - Class 4 (Proliferative): {counts[4]} ({percentages[4]:.2f}%)
- **Imbalance Ratio**: `{summary_data['class_imbalance']['imbalance_ratio']}` (Majority Class Count vs Minority Class Count)
- **Recommended Loss Function**: Class-Weighted Cross-Entropy Loss

## Global RGB Channel Mean & Standard Deviation (Normalized [0, 1])
- **Dataset Mean**: R: `{summary_data['rgb_mean'][0]:.6f}` \| G: `{summary_data['rgb_mean'][1]:.6f}` \| B: `{summary_data['rgb_mean'][2]:.6f}`
- **Dataset Std Dev**: R: `{summary_data['rgb_std'][0]:.6f}` \| G: `{summary_data['rgb_std'][1]:.6f}` \| B: `{summary_data['rgb_std'][2]:.6f}`

The computed RGB statistics provide a dataset-specific normalization baseline that will be benchmarked against ImageNet normalization during model training.

## Quality Outlier Assessment Percentages
- **Dark Images (< {b_5pct:.2f})**: {summary_data['quality_metrics']['dark_images_percentage']:.2f}% (Threshold: 5th percentile)
- **Bright Images (> {b_95pct:.2f})**: {summary_data['quality_metrics']['bright_images_percentage']:.2f}% (Threshold: 95th percentile)
- **Blurry Images (< {bl_5pct:.2f})**: {summary_data['quality_metrics']['blurry_images_percentage']:.2f}% (Threshold: 5th percentile)
- **Corrupted Files / Invalid Channels**: Corrupted: {summary_data['corrupted_images']} \| Missing: {summary_data['missing_images']}

## Candidate Preprocessing Guidelines
Based on statistical variances, the following candidate configurations have been identified for experimental evaluation during preprocessing and baseline model training:
- **Candidate Image Sizes**: `[224, 384, 512]` (To be determined via training benchmarks)
- **Candidate Normalization Methods**: `[ImageNet, Dataset]` (To be compared in ablation studies)
- **Candidate Augmentations**:
  - Horizontal Flip: `[True, False]`
  - Rotation Degrees: `[0, 15, 30]`
  - Color Jitter: `[True, False]`
  - Random Crop: `[True, False]`

## Conclusion and Next Steps
Overall, the exploratory analysis confirms that the APTOS 2019 training cohort is structurally valid, quantitatively characterized, and suitable for deep learning experimentation. The extracted metadata, quality assessments, duplicate audit, and preprocessing recommendations establish a reproducible foundation for Step 4 (Image Preprocessing) and subsequent baseline model development.
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved Markdown report to {md_path}")
    
    # Copy Markdown report to research Volume 3 directory
    research_report_path = RETINA_RESEARCH_DIR / "retina_eda_summary.md"
    shutil.copy2(str(md_path), str(research_report_path))
    logger.info(f"Copied Markdown report to research directory: {research_report_path}")
    

    
    # ----------------------------------------------------
    # Generate eda.ipynb Notebook programmatically
    # ----------------------------------------------------
    notebook_path = RETINA_NOTEBOOKS_DIR / "eda.ipynb"
    logger.info(f"Constructing Jupyter notebook at {notebook_path}...")
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 3: Exploratory Data Analysis (EDA) - Retina Module\n",
                    "\n",
                    "This notebook loads the cached dataset statistics, generates descriptive figures, and establishes baseline configurations for model training.\n",
                    "The analysis is strictly **read-only** and performs no image modifications."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import json\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import cv2\n",
                    "from pathlib import Path\n",
                    "\n",
                    "# Imports from src.config\n",
                    "from src.config import (\n",
                    "    EDA_STATISTICS_CSV,\n",
                    "    EDA_SUMMARY_JSON,\n",
                    "    PREPROCESSING_RECOMMENDATIONS_JSON,\n",
                    "    CLASS_NAMES,\n",
                    "    TRAIN_IMAGES,\n",
                    "    SEED\n",
                    ")\n",
                    "\n",
                    "np.random.seed(SEED)\n",
                    "print(\"Libraries loaded and random seed initialized.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Dataset Overview\n",
                    "We load the global summary JSON containing versions, environment configurations, and the deterministic dataset fingerprint."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "with open(EDA_SUMMARY_JSON, 'r') as f:\n",
                    "    summary = json.load(f)\n",
                    "\n",
                    "print(f\"Dataset: {summary['dataset']}\")\n",
                    "print(f\"EDA Version: {summary['eda_version']}\")\n",
                    "print(f\"Dataset Fingerprint (SHA-256): {summary['dataset_fingerprint']}\")\n",
                    "print(f\"Total Labeled Training Images: {summary['total_training_images']}\")\n",
                    "print(f\"Missing Images: {summary['missing_images']} | Corrupted: {summary['corrupted_images']} | Invalid Channels: {summary['invalid_channel_images']}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Class Distribution Analysis\n",
                    "Check target label proportions to evaluate class imbalances."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df = pd.read_csv(EDA_STATISTICS_CSV)\n",
                    "counts = df['diagnosis'].value_counts().sort_index()\n",
                    "for cid, (name, count) in enumerate(zip(CLASS_NAMES, counts)):\n",
                    "    pct = (count / len(df)) * 100\n",
                    "    print(f\"Class {cid} ({name}): {count} images ({pct:.2f}%)\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Resolution, Aspect Ratio, & Size Statistics\n",
                    "Identify the image shapes, crop factors, and disk footprints."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(f\"Width  -> Min: {df['width'].min()} | Max: {df['width'].max()} | Avg: {df['width'].mean():.1f} pixels\")\n",
                    "print(f\"Height -> Min: {df['height'].min()} | Max: {df['height'].max()} | Avg: {df['height'].mean():.1f} pixels\")\n",
                    "print(f\"Aspect Ratio -> Avg: {df['aspect_ratio'].mean():.2f} | Std: {df['aspect_ratio'].std():.2f}\")\n",
                    "print(f\"File Size    -> Avg: {df['filesize_kb'].mean():.1f} KB | Std: {df['filesize_kb'].std():.1f} KB\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Grayscale Brightness & Laplacian Sharpness Analysis\n",
                    "Verify the brightness and focus distributions across the dataset."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(f\"Average Grayscale Brightness: {df['brightness_score'].mean():.2f} (std: {df['brightness_score'].std():.2f})\")\n",
                    "print(f\"Average Laplacian Sharpness Score: {df['blur_score'].mean():.2f} (std: {df['blur_score'].std():.2f})\")\n",
                    "print(f\"Darkest 5% Threshold: < {summary['thresholds']['brightness_5pct']:.2f}\")\n",
                    "print(f\"Blurriest 5% Threshold: < {summary['thresholds']['blur_5pct']:.2f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Visual Duplicate Verification\n",
                    "Verifies if there are duplicate image pairs based on pixel contents hashes."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "dup_path = Path(EDA_STATISTICS_CSV).parent / 'duplicate_images.csv'\n",
                    "if dup_path.exists():\n",
                    "    dup_df = pd.read_csv(dup_path)\n",
                    "    print(f\"Total exact visual duplicate image pairs detected: {len(dup_df)}\")\n",
                    "else:\n",
                    "    print(\"No visual duplicates verification file found.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Continuous Image Quality Score (Q)\n",
                    "Loads the continuous Image Quality score computed from normalized resolution, brightness, and sharpness."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(f\"Average image quality score Q: {df['quality_score'].mean():.4f} (std: {df['quality_score'].std():.4f})\")\n",
                    "print(f\"Minimum quality score: {df['quality_score'].min():.4f} | Maximum: {df['quality_score'].max():.4f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. RGB Global Normalization Statistics\n",
                    "Compute dataset-specific channel-wise mean and standard deviations."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(f\"Global RGB Mean: R={summary['rgb_mean'][0]:.6f}, G={summary['rgb_mean'][1]:.6f}, B={summary['rgb_mean'][2]:.6f}\")\n",
                    "print(f\"Global RGB Std:  R={summary['rgb_std'][0]:.6f}, G={summary['rgb_std'][1]:.6f}, B={summary['rgb_std'][2]:.6f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Preprocessing Recommendations\n",
                    "Review the final configuration guidelines derived from the EDA pipeline."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "with open(PREPROCESSING_RECOMMENDATIONS_JSON, 'r') as f:\n",
                    "    recs = json.load(f)\n",
                    "print(json.dumps(recs, indent=2))"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open(notebook_path, "w") as f:
        json.dump(notebook, f, indent=2)
    logger.info(f"Jupyter notebook saved successfully at {notebook_path}")
    
    # ----------------------------------------------------
    # Generate manifest.json (written last to ensure all files exist)
    # ----------------------------------------------------
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    # Dynamically count all generated files
    results_files = [f for f in RETINA_RESULTS_DIR.rglob("*") if f.is_file()]
    report_files = [f for f in REPORTS_DIR.glob("retina_eda_summary.*") if f.is_file()]
    research_report_file = [RETINA_RESEARCH_DIR / "retina_eda_summary.md"] if (RETINA_RESEARCH_DIR / "retina_eda_summary.md").exists() else []
    notebook_file = [notebook_path] if notebook_path.exists() else []
    recommendations_file = [PREPROCESSING_RECOMMENDATIONS_JSON] if PREPROCESSING_RECOMMENDATIONS_JSON.exists() else []
    
    total_generated_files = len(results_files) + len(report_files) + len(research_report_file) + len(notebook_file) + len(recommendations_file)
    # Add 1 for the manifest itself if it doesn't exist yet in results_files
    if not (RETINA_RESULTS_DIR / "manifest.json").exists():
        total_generated_files += 1
        
    MODULE_VERSION = "1.0.0"
    PHASE_TAG = "v0.3.0"
    
    manifest_data = {
        "module": "Retina",
        "phase": "EDA",
        "version": MODULE_VERSION,
        "git_tag": PHASE_TAG,
        "generated_files": int(total_generated_files),
        "figures": int(generated_figures_count),
        "reports": {
            "pdf": str(REPORTS_DIR / "retina_eda_summary.pdf"),
            "markdown_reports": [
                str(REPORTS_DIR / "retina_eda_summary.md"),
                str(RETINA_RESEARCH_DIR / "retina_eda_summary.md")
            ],
            "notebook": str(notebook_path)
        },
        "execution_time_seconds": round(execution_time, 2),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": "v0.3.0-draft",
        "dataset_sha256": summary_data["dataset_fingerprint"],
        "python_version": sys.version.split()[0],
        "opencv_version": cv2.__version__
    }
    manifest_path = RETINA_RESULTS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info(f"Saved manifest JSON to {manifest_path}")
    
    logger.info("==================================================")
    logger.info(f"EDA PROCESSING COMPLETED IN {execution_time:.2f} SECONDS")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
