from pathlib import Path
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
# Training Hyperparameters (Future Use)
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

# Metadata additions
IMAGE_STATISTICS_CSV = METADATA_DIR / "image_statistics.csv"
EDA_STATISTICS_CSV = METADATA_DIR / "eda_statistics.csv"
EDA_STATISTICS_PARQUET = METADATA_DIR / "eda_statistics.parquet"
EDA_SUMMARY_JSON = METADATA_DIR / "eda_summary.json"
PREPROCESSING_RECOMMENDATIONS_JSON = METADATA_DIR / "preprocessing_recommendations.json"

# Quality Outlier CSV Paths
DARK_IMAGES_CSV = METADATA_DIR / "dark_images.csv"
BRIGHT_IMAGES_CSV = METADATA_DIR / "bright_images.csv"
BLURRY_IMAGES_CSV = METADATA_DIR / "blurry_images.csv"
SMALL_IMAGES_CSV = METADATA_DIR / "small_images.csv"
LARGE_IMAGES_CSV = METADATA_DIR / "large_images.csv"

# ==========================
# Phase 3 EDA Parameters
# ==========================
EDA_VERSION = "1.0.0"
FIGURE_DPI = 300
FIGURE_FORMATS = ["png", "svg"]
OUTLIER_PERCENTILE = 0.05
MAX_ANALYSIS_DIM = 512
