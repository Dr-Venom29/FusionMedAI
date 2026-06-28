import sys
import tempfile
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import TRAIN_SPLIT_CSV, TRAIN_IMAGES, ID_COLUMN, LABEL_COLUMN, VALID_LABELS
from src.data.dataset import RetinaDataset

def test_dataset_loading() -> None:
    print("Test 1: Basic instantiation and length verification...")
    # Load dataset size directly from CSV
    expected_length = len(pd.read_csv(TRAIN_SPLIT_CSV))
    
    dataset = RetinaDataset(csv_file=TRAIN_SPLIT_CSV, image_dir=TRAIN_IMAGES)
    
    if len(dataset) != expected_length:
        raise ValueError(
            f"Failed: Dataset length ({len(dataset)}) does not match expected length ({expected_length})."
        )
    print(f"[OK] Dataset length matches: {len(dataset)} samples.")
    
    print("\nTest 2: Sample loading without transform...")
    image, label = dataset[0]
    
    if not isinstance(image, Image.Image):
        raise TypeError(f"Failed: Expected PIL.Image.Image, got {type(image)}")
        
    if image.mode != "RGB":
        raise ValueError(f"Failed: Expected RGB mode, got {image.mode}")
        
    if not isinstance(label, int):
        raise TypeError(f"Failed: Expected label type int, got {type(label)}")
        
    if label not in VALID_LABELS:
        raise ValueError(f"Failed: Label {label} is not in valid labels {VALID_LABELS}")
        
    print(f"[OK] Sample loaded successfully: Image type={type(image)}, Mode={image.mode}, Label={label} (type={type(label)}).")
    
    print("\nTest 3: Sample loading with ToTensor transform...")
    dataset_with_transform = RetinaDataset(
        csv_file=TRAIN_SPLIT_CSV,
        image_dir=TRAIN_IMAGES,
        transform=transforms.ToTensor()
    )
    
    transformed_image, label = dataset_with_transform[0]
    
    if not isinstance(transformed_image, torch.Tensor):
        raise TypeError(f"Failed: Expected torch.Tensor, got {type(transformed_image)}")
        
    print(f"[OK] Transformed sample loaded successfully: Image type={type(transformed_image)}, Shape={transformed_image.shape}.")

def test_missing_image() -> None:
    print("\nTest 4: Missing image file error handling...")
    # Create temporary CSV file referencing a nonexistent image
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_csv_path = Path(temp_dir) / "fake_split.csv"
        
        # Build DataFrame with nonexistent ID
        fake_df = pd.DataFrame([
            {ID_COLUMN: "nonexistent_id_xyz123", LABEL_COLUMN: 2}
        ])
        fake_df.to_csv(temp_csv_path, index=False)
        
        # Instantiate dataset using TRAIN_IMAGES directory (where the nonexistent image won't exist)
        dataset = RetinaDataset(csv_file=temp_csv_path, image_dir=TRAIN_IMAGES)
        
        try:
            _ = dataset[0]
            raise RuntimeError("Failed: Accessing missing image did not raise FileNotFoundError.")
        except FileNotFoundError as e:
            print(f"[OK] Success: Raised FileNotFoundError as expected. Message: {e}")

def test_invalid_csv() -> None:
    print("\nTest 5: Invalid CSV columns validation...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_csv_path = Path(temp_dir) / "invalid_columns.csv"
        
        # Build DataFrame with wrong column names
        invalid_df = pd.DataFrame([
            {"wrong_id_col": "abc123", "wrong_label_col": 3}
        ])
        invalid_df.to_csv(temp_csv_path, index=False)
        
        try:
            _ = RetinaDataset(csv_file=temp_csv_path, image_dir=TRAIN_IMAGES)
            raise RuntimeError("Failed: Invalid columns did not raise ValueError.")
        except ValueError as e:
            print(f"[OK] Success: Raised ValueError as expected. Message: {e}")

def main() -> int:
    try:
        test_dataset_loading()
        test_missing_image()
        test_invalid_csv()
        print("\n==========================================")
        print("ALL RETINA_DATASET TESTS PASSED SUCCESSFULLY!")
        print("==========================================")
        return 0
    except Exception as e:
        print(f"\n[FAIL] Test execution failed: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
