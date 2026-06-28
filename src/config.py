from pathlib import Path
import os
import re
import torch

# ==========================
# Project Paths
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "datasets"

RAW_DATA = DATASET_ROOT / "raw" / "aptos2019"

PROCESSED_DATA = DATASET_ROOT / "processed"

PROCESSED_SPLITS_DIR = PROCESSED_DATA / "splits"

TRAIN_SPLIT_CSV = PROCESSED_SPLITS_DIR / "train.csv"

VAL_SPLIT_CSV = PROCESSED_SPLITS_DIR / "val.csv"

TEST_SPLIT_CSV = PROCESSED_SPLITS_DIR / "test.csv"

INTERIM_DATA = DATASET_ROOT / "interim"

METADATA_DIR = DATASET_ROOT / "metadata"

TRAIN_IMAGES = RAW_DATA / "train_images"

TEST_IMAGES = RAW_DATA / "test_images"

TRAIN_CSV = RAW_DATA / "train.csv"

TEST_CSV = RAW_DATA / "test.csv"

# ==========================
# Reproducibility
# ==========================

SEED = 42

EXPECTED_TRAIN_SAMPLES = 3662

# ==========================
# Dataset
# ==========================

ID_COLUMN = "id_code"

LABEL_COLUMN = "diagnosis"

NUM_CLASSES = 5

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative"
]

VALID_LABELS = {0, 1, 2, 3, 4}

# ==========================
# Image Preprocessing
# ==========================

IMAGE_SIZE = 224

# Temporary ImageNet normalization.
# Replace with dataset-specific statistics once computed.
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

ROTATION_DEGREES = 15
FLIP_PROBABILITY = 0.5

# Color Jitter parameters
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.1
COLOR_JITTER_HUE = 0.05

# ==========================
# Training Hyperparameters
# ==========================

BATCH_SIZE = 32

# On Windows, num_workers > 0 can sometimes cause issues during development or slower startup,
# but keeping it configurable.
NUM_WORKERS = 4

PIN_MEMORY = torch.cuda.is_available()

PERSISTENT_WORKERS = False

DROP_LAST = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_AMP = True

MODEL_NAME = "efficientnet_b0"

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

# Added for Step 4
EPOCHS = 20
PATIENCE = 10
EARLY_STOPPING_MONITOR = "qwk"  # Optimize validation QWK

# Resolve run version dynamically
RUN_VERSION = os.environ.get("FUSIONMED_RUN_VERSION")
RUN_NAME = os.environ.get("FUSIONMED_RUN_NAME", MODEL_NAME)

if not RUN_VERSION:
    experiments_base = PROJECT_ROOT / "experiments"
    existing_versions = []
    if experiments_base.exists():
        for path in experiments_base.iterdir():
            if path.is_dir():
                match = re.match(r"^v(\d+)_", path.name)
                if match:
                    existing_versions.append(int(match.group(1)))
    if existing_versions:
        # Default to latest version for evaluation/testing/inference
        latest_version = max(existing_versions)
        RUN_VERSION = f"v{latest_version:03d}"
    else:
        RUN_VERSION = "v001"

# ==========================
# Run Directory Setup
# ==========================

RUN_DIR = PROJECT_ROOT / "experiments" / f"{RUN_VERSION}_{RUN_NAME}"

CHECKPOINT_DIR = RUN_DIR / "checkpoints"
TENSORBOARD_DIR = RUN_DIR / "tensorboard"
CURVES_DIR = RUN_DIR / "curves"
CONFUSION_MATRIX_DIR = RUN_DIR / "confusion_matrix"
ROC_DIR = RUN_DIR / "roc"

# Checkpoint paths
BEST_CHECKPOINT = CHECKPOINT_DIR / "best_model.pt"
LAST_CHECKPOINT = CHECKPOINT_DIR / "last_model.pt"

# Run outputs
RUN_HISTORY_CSV = RUN_DIR / "history.csv"
RUN_HISTORY_JSON = RUN_DIR / "history.json"
RUN_CONFIG_JSON = RUN_DIR / "config.json"
RUN_PREDICTIONS_CSV = RUN_DIR / "predictions.csv"

# Global tracking
BASELINE_CSV = PROJECT_ROOT / "experiments" / "baseline_results.csv"
EXPERIMENT_LOG_EXCEL = PROJECT_ROOT / "experiments" / "experiment_log.xlsx"

# Auto-create run folders
for directory in [RUN_DIR, CHECKPOINT_DIR, TENSORBOARD_DIR, CURVES_DIR, CONFUSION_MATRIX_DIR, ROC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================
# Phase 3 Directories & Files
# ==========================

RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESEARCH_DIR = PROJECT_ROOT / "research"

# Retina-specific paths
RETINA_NOTEBOOKS_DIR = NOTEBOOKS_DIR / "retina"
RETINA_RESULTS_DIR = RESULTS_DIR / "retina"
RETINA_FIGURES_DIR = RETINA_RESULTS_DIR / "figures"
RETINA_RESEARCH_DIR = RESEARCH_DIR / "Volume_03_Exploratory_Data_Analysis"

# Metadata Subdirectories
METADATA_QUALITY_DIR = METADATA_DIR / "quality"
METADATA_VALIDATION_DIR = METADATA_DIR / "validation"
METADATA_STATISTICS_DIR = METADATA_DIR / "statistics"
METADATA_EDA_DIR = METADATA_DIR / "eda"
METADATA_RECOMMENDATIONS_DIR = METADATA_DIR / "recommendations"

# Metadata additions
IMAGE_STATISTICS_CSV = METADATA_STATISTICS_DIR / "image_statistics.csv"
EDA_STATISTICS_CSV = METADATA_EDA_DIR / "eda_statistics.csv"
EDA_STATISTICS_PARQUET = METADATA_EDA_DIR / "eda_statistics.parquet"
EDA_SUMMARY_JSON = METADATA_EDA_DIR / "eda_summary.json"
PREPROCESSING_RECOMMENDATIONS_JSON = METADATA_RECOMMENDATIONS_DIR / "preprocessing_recommendations.json"

# Quality Outlier CSV Paths
DARK_IMAGES_CSV = METADATA_QUALITY_DIR / "dark_images.csv"
BRIGHT_IMAGES_CSV = METADATA_QUALITY_DIR / "bright_images.csv"
BLURRY_IMAGES_CSV = METADATA_QUALITY_DIR / "blurry_images.csv"
SMALL_IMAGES_CSV = METADATA_STATISTICS_DIR / "small_images.csv"
LARGE_IMAGES_CSV = METADATA_STATISTICS_DIR / "large_images.csv"

# ==========================
# Phase 3 EDA Parameters
# ==========================
EDA_VERSION = "1.0.0"
FIGURE_DPI = 300
FIGURE_FORMATS = ["png", "svg"]
OUTLIER_PERCENTILE = 0.05
MAX_ANALYSIS_DIM = 512
