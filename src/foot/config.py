from pathlib import Path
import os
import re
import torch

# ==========================
# Project & Modality Paths
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "foot"

RAW_DATA = DATASET_ROOT / "raw"

PROCESSED_DATA = DATASET_ROOT / "processed"

PROCESSED_SPLITS_DIR = PROCESSED_DATA / "splits"

TRAIN_SPLIT_CSV = PROCESSED_SPLITS_DIR / "train.csv"

VAL_SPLIT_CSV = PROCESSED_SPLITS_DIR / "val.csv"

TEST_SPLIT_CSV = PROCESSED_SPLITS_DIR / "test.csv"

INTERIM_DATA = DATASET_ROOT / "interim"

METADATA_DIR = DATASET_ROOT / "metadata"

# ==========================
# Reproducibility
# ==========================

SEED = 42

# ==========================
# Foot DFU Dataset (Wagner 4-Class)
# ==========================

ID_COLUMN = "id_code"

LABEL_COLUMN = "wagner_grade"

NUM_CLASSES = 4

CLASS_NAMES = [
    "Grade 1",  # Superficial ulcer (full skin thickness, no subcutaneous involvement)
    "Grade 2",  # Deep ulcer (penetrating to tendon, ligament, or capsule, without bone involvement)
    "Grade 3",  # Deep ulcer with abscess, osteomyelitis, or joint sepsis
    "Grade 4"   # Localized gangrene (forefoot or heel)
]

VALID_LABELS = {0, 1, 2, 3}

# ==========================
# Image Preprocessing & Observed Statistics
# ==========================

IMAGE_SIZE = 224

# Measured from dataset during Phase 10.1.E
OBSERVED_DATASET_MEAN = [0.4937, 0.3630, 0.3272]
OBSERVED_DATASET_STD = [0.1744, 0.1632, 0.1551]

NORMALIZATION_MEAN = OBSERVED_DATASET_MEAN
NORMALIZATION_STD = OBSERVED_DATASET_STD

# Candidate Augmentations (To be evaluated in Step 10.2 - Data Pipeline)
CANDIDATE_ROTATION_DEGREES = 15
CANDIDATE_FLIP_PROBABILITY = 0.5

# ==========================
# Training Hyperparameters
# ==========================

BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True

MODEL_NAME = "efficientnet_b0"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
PATIENCE = 10

# ==========================
# Experiment & Metadata Directories
# ==========================

FOOT_EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "foot"
FOOT_RESULTS_DIR = PROJECT_ROOT / "results" / "foot"
FOOT_NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks" / "foot"
FOOT_RESEARCH_DIR = PROJECT_ROOT / "research" / "foot"

METADATA_QUALITY_DIR = METADATA_DIR / "quality"
METADATA_VALIDATION_DIR = METADATA_DIR / "validation"
METADATA_STATISTICS_DIR = METADATA_DIR / "statistics"

for directory in [
    DATASET_ROOT, RAW_DATA, PROCESSED_DATA, PROCESSED_SPLITS_DIR, INTERIM_DATA, METADATA_DIR,
    FOOT_EXPERIMENTS_DIR, FOOT_RESULTS_DIR, FOOT_NOTEBOOKS_DIR, FOOT_RESEARCH_DIR,
    METADATA_QUALITY_DIR, METADATA_VALIDATION_DIR, METADATA_STATISTICS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)
