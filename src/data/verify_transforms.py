import sys
from pathlib import Path
import torch
from PIL import Image

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import IMAGE_SIZE
from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms
)

def create_dummy_image() -> Image.Image:
    """Creates a dummy RGB PIL Image with non-uniform colors for robust transforms testing."""
    # Let's create a 500x500 image with a gradient/pattern so rotation/jitter is highly visible
    img = Image.new("RGB", (500, 500))
    pixels = img.load()
    for x in range(500):
        for y in range(500):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    return img

def test_train_transforms(dummy_img: Image.Image) -> None:
    print("Test 1: Training pipeline verification...")
    train_transform = get_train_transforms()
    
    # 1. Output types, shapes, and properties checks
    out = train_transform(dummy_img)
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Failed: Expected torch.Tensor, got {type(out)}")
        
    expected_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
    if out.shape != expected_shape:
        raise ValueError(f"Failed: Expected shape {expected_shape}, got {out.shape}")
        
    if out.dtype != torch.float32:
        raise ValueError(f"Failed: Expected dtype torch.float32, got {out.dtype}")
        
    if not torch.isfinite(out).all():
        raise ValueError("Failed: Output tensor contains non-finite values (NaN or Inf).")
        
    print("[OK] Train pipeline returns finite float32 tensor of correct shape.")
    
    # 2. Assert variations (nondeterminism) in augmentations
    tensors = [train_transform(dummy_img) for _ in range(5)]
    any_diff = False
    for i in range(1, len(tensors)):
        if not torch.equal(tensors[0], tensors[i]):
            any_diff = True
            break
            
    if not any_diff:
        raise ValueError("Failed: Train pipeline produced identical outputs across multiple runs (expected variance).")
        
    print("[OK] Train pipeline produces non-deterministic augmented outputs.")

def test_val_transforms(dummy_img: Image.Image) -> None:
    print("\nTest 2: Validation pipeline verification...")
    val_transform = get_val_transforms()
    
    # 1. Output properties
    out = val_transform(dummy_img)
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Failed: Expected torch.Tensor, got {type(out)}")
        
    expected_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
    if out.shape != expected_shape:
        raise ValueError(f"Failed: Expected shape {expected_shape}, got {out.shape}")
        
    if out.dtype != torch.float32:
        raise ValueError(f"Failed: Expected dtype torch.float32, got {out.dtype}")
        
    if not torch.isfinite(out).all():
        raise ValueError("Failed: Output tensor contains non-finite values.")
        
    print("[OK] Validation pipeline returns finite float32 tensor of correct shape.")
    
    # 2. Assert determinism
    tensors = [val_transform(dummy_img) for _ in range(5)]
    for i in range(1, len(tensors)):
        if not torch.equal(tensors[0], tensors[i]):
            raise ValueError("Failed: Validation pipeline is not deterministic (outputs vary).")
            
    print("[OK] Validation pipeline is strictly deterministic.")

def test_test_transforms(dummy_img: Image.Image) -> None:
    print("\nTest 3: Test pipeline verification...")
    test_transform = get_test_transforms()
    val_transform = get_val_transforms()
    
    # 1. Output properties
    out = test_transform(dummy_img)
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Failed: Expected torch.Tensor, got {type(out)}")
        
    expected_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
    if out.shape != expected_shape:
        raise ValueError(f"Failed: Expected shape {expected_shape}, got {out.shape}")
        
    if out.dtype != torch.float32:
        raise ValueError(f"Failed: Expected dtype torch.float32, got {out.dtype}")
        
    if not torch.isfinite(out).all():
        raise ValueError("Failed: Output tensor contains non-finite values.")
        
    print("[OK] Test pipeline returns finite float32 tensor of correct shape.")
    
    # 2. Assert determinism
    tensors = [test_transform(dummy_img) for _ in range(5)]
    for i in range(1, len(tensors)):
        if not torch.equal(tensors[0], tensors[i]):
            raise ValueError("Failed: Test pipeline is not deterministic.")
            
    # 3. Assert identity with validation
    val_out = val_transform(dummy_img)
    if not torch.equal(out, val_out):
        raise ValueError("Failed: Test pipeline outputs differ from Validation pipeline outputs.")
        
    print("[OK] Test pipeline is deterministic and identical to Validation pipeline.")

def main() -> int:
    try:
        dummy_img = create_dummy_image()
        test_train_transforms(dummy_img)
        test_val_transforms(dummy_img)
        test_test_transforms(dummy_img)
        print("\n==========================================")
        print("ALL IMAGE TRANSFORMS TESTS PASSED SUCCESSFULLY!")
        print("==========================================")
        return 0
    except Exception as e:
        print(f"\n[FAIL] Transforms verification failed: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
