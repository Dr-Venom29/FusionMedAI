import sys
from pathlib import Path
from typing import Optional, List
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.config import (
    IMAGE_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    CANDIDATE_ROTATION_DEGREES,
    CANDIDATE_FLIP_PROBABILITY
)

def get_foot_train_transforms(
    image_size: int = IMAGE_SIZE,
    use_augmentation: bool = True,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> transforms.Compose:
    """
    Returns the transformation pipeline for training Foot DFU images.
    
    Clinical & Data Engineering Evaluation:
    1. Resize: Ensures uniform (224, 224) input tensor dimensions with bilinear interpolation.
    2. Random Rotation (+/-15 degrees): Clinically justified. Handheld photography of diabetic foot ulcers
       exhibits natural variation in camera tilt and patient leg positioning.
    3. Random Horizontal Flip (p=0.5): Evaluated for foot laterality. Because Wagner classification assesses
       ulcer tissue depth, osteomyelitis, and gangrene rather than left vs. right foot anatomical laterality,
       horizontal reflection preserves the diagnostic class label while improving spatial robustness.
    4. Color Jitter (Brightness 0.2, Contrast 0.2, Saturation 0.1): Clinically justified. Simulates varying ambient
       lighting, flash illumination, and mobile camera sensor calibrations without distorting tissue color signatures.
    5. Normalization: Uses observed Foot dataset statistics (Mean [0.4937, 0.3630, 0.3272], Std [0.1744, 0.1632, 0.1551]).
    
    Args:
        image_size: Target height and width (default: 224).
        use_augmentation: If True, includes spatial and color augmentations. If False, returns basic pipeline.
        mean: Normalization mean per channel (default: OBSERVED_DATASET_MEAN).
        std: Normalization std per channel (default: OBSERVED_DATASET_STD).
        
    Returns:
        transforms.Compose: Composed torchvision transform pipeline.
    """
    norm_mean = mean if mean is not None else NORMALIZATION_MEAN
    norm_std = std if std is not None else NORMALIZATION_STD
    
    transform_list = [
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR
        )
    ]
    
    if use_augmentation:
        transform_list.extend([
            transforms.RandomRotation(
                degrees=(-CANDIDATE_ROTATION_DEGREES, CANDIDATE_ROTATION_DEGREES)
            ),
            transforms.RandomHorizontalFlip(
                p=CANDIDATE_FLIP_PROBABILITY
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.02
            )
        ])
        
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    return transforms.Compose(transform_list)


def get_foot_val_transforms(
    image_size: int = IMAGE_SIZE,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> transforms.Compose:
    """
    Returns deterministic transformation pipeline for validation Foot DFU images.
    Applies resolution standardization and normalization; no random augmentations applied.
    """
    norm_mean = mean if mean is not None else NORMALIZATION_MEAN
    norm_std = std if std is not None else NORMALIZATION_STD
    
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])


def get_foot_test_transforms(
    image_size: int = IMAGE_SIZE,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> transforms.Compose:
    """
    Returns deterministic transformation pipeline for test Foot DFU images.
    Behaves identically to validation pipeline.
    """
    return get_foot_val_transforms(image_size=image_size, mean=mean, std=std)


def _test_foot_transforms():
    """Unit test for foot transformation pipelines."""
    print("Testing Foot DFU Transformation Pipelines...")
    from PIL import Image
    import torch
    
    # Create dummy RGB image
    dummy_img = Image.new("RGB", (224, 224), color=(128, 90, 80))
    
    train_tf = get_foot_train_transforms()
    val_tf = get_foot_val_transforms()
    test_tf = get_foot_test_transforms()
    
    train_tensor = train_tf(dummy_img)
    val_tensor = val_tf(dummy_img)
    test_tensor = test_tf(dummy_img)
    
    print(f" - Train Tensor Shape: {train_tensor.shape}, dtype: {train_tensor.dtype}")
    print(f" - Val Tensor Shape:   {val_tensor.shape}, dtype: {val_tensor.dtype}")
    print(f" - Test Tensor Shape:  {test_tensor.shape}, dtype: {test_tensor.dtype}")
    
    assert train_tensor.shape == torch.Size([3, 224, 224]), "Train tensor shape invalid"
    assert val_tensor.shape == torch.Size([3, 224, 224]), "Val tensor shape invalid"
    assert test_tensor.shape == torch.Size([3, 224, 224]), "Test tensor shape invalid"
    
    print("Foot Transformation Pipelines verified successfully!")

if __name__ == "__main__":
    _test_foot_transforms()
