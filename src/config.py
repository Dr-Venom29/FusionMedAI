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
