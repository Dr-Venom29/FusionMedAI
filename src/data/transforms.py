import sys
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import (
    IMAGE_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    ROTATION_DEGREES,
    FLIP_PROBABILITY,
    COLOR_JITTER_BRIGHTNESS,
    COLOR_JITTER_CONTRAST,
    COLOR_JITTER_SATURATION,
    COLOR_JITTER_HUE
)

def get_train_transforms() -> transforms.Compose:
    """
    Returns the transformation pipeline for training data.
    Includes data augmentation to prevent overfitting.
    """
    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=InterpolationMode.BILINEAR
        ),
        transforms.RandomRotation(
            degrees=(-ROTATION_DEGREES, ROTATION_DEGREES)
        ),
        transforms.RandomHorizontalFlip(
            p=FLIP_PROBABILITY
        ),
        transforms.ColorJitter(
            brightness=COLOR_JITTER_BRIGHTNESS,
            contrast=COLOR_JITTER_CONTRAST,
            saturation=COLOR_JITTER_SATURATION,
            hue=COLOR_JITTER_HUE
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=NORMALIZATION_MEAN,
            std=NORMALIZATION_STD
        )
    ])

def get_val_transforms() -> transforms.Compose:
    """
    Returns the deterministic transformation pipeline for validation data.
    No augmentation is applied.
    """
    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=NORMALIZATION_MEAN,
            std=NORMALIZATION_STD
        )
    ])

def get_test_transforms() -> transforms.Compose:
    """
    Returns the deterministic transformation pipeline for test data.
    Must behave identically to the validation pipeline.
    """
    return get_val_transforms()
