from pathlib import Path
import torch

# ==========================
# Project Paths
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "datasets"

RAW_DATA = DATASET_ROOT / "raw" / "aptos2019"

PROCESSED_DATA = DATASET_ROOT / "processed"

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

# ==========================
# Dataset
# ==========================

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
# Training Hyperparameters (Future Use)
# ==========================

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 4

PIN_MEMORY = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_AMP = True

MODEL_NAME = "efficientnet_b0"

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4
