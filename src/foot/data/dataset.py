import sys
from pathlib import Path
from typing import Union, Callable, Optional, Tuple, Dict, Any
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.config import RAW_DATA, ID_COLUMN, LABEL_COLUMN, CLASS_NAMES

class FootDFUDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Foot Ulcer (DFU) Wagner 4-Class Classification.
    
    Loads image paths and labels lazily from split CSV manifests (e.g. train.csv, val.csv, test.csv),
    guarantees 3-channel RGB image mode, applies torchvision transformation pipelines, and returns
    (image_tensor, wagner_grade_label) tuples or metadata-enriched tuples.
    """
    
    def __init__(
        self,
        csv_file: Union[str, Path],
        image_dir: Union[str, Path] = RAW_DATA,
        transform: Optional[Callable] = None,
        return_metadata: bool = False
    ) -> None:
        """
        Args:
            csv_file: Path to split CSV manifest (e.g. datasets/foot/processed/splits/train.csv).
            image_dir: Path to immutable raw image directory (default: datasets/foot/raw/).
            transform: Optional torchvision/albumentations transformation pipeline.
            return_metadata: If True, __getitem__ returns (image, label, metadata_dict).
                             If False, __getitem__ returns (image, label).
                             
        Raises:
            FileNotFoundError: If csv_file or image_dir does not exist.
            ValueError: If required columns ('id_code', 'wagner_grade', 'image_path') are missing.
        """
        self.csv_file = Path(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.return_metadata = return_metadata
        
        if not self.csv_file.exists():
            raise FileNotFoundError(f"Split CSV manifest not found: '{self.csv_file}'")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: '{self.image_dir}'")
            
        self.dataframe = pd.read_csv(self.csv_file)
        
        # Column validation
        required_cols = [ID_COLUMN, LABEL_COLUMN, "image_path"]
        for col in required_cols:
            if col not in self.dataframe.columns:
                raise ValueError(
                    f"CSV manifest '{self.csv_file}' is missing required column '{col}'."
                )

    def __len__(self) -> int:
        """Returns total number of image records in the dataset partition."""
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Union[Tuple[Any, int], Tuple[Any, int, Dict[str, Any]]]:
        """
        Retrieves the image and label at the specified index.
        
        Args:
            index: Row index in the manifest dataframe.
            
        Returns:
            Tuple[Any, int] if return_metadata is False:
                - image: PIL Image or transformed torch.Tensor [C, H, W]
                - label: Integer Wagner grade (0: Grade 1, 1: Grade 2, 2: Grade 3, 3: Grade 4)
            Tuple[Any, int, Dict[str, Any]] if return_metadata is True:
                - image: PIL Image or transformed torch.Tensor
                - label: Integer Wagner grade
                - metadata: Dict containing id_code, image_path, source_image_id, class_name
        """
        row = self.dataframe.iloc[index]
        id_code = str(row[ID_COLUMN])
        label = int(row[LABEL_COLUMN])
        rel_image_path = str(row["image_path"])
        
        abs_image_path = self.image_dir / rel_image_path
        if not abs_image_path.exists():
            raise FileNotFoundError(
                f"Image file missing on disk: '{abs_image_path}' for record index {index} (ID: {id_code})."
            )
            
        # Lazy image loading with guaranteed RGB mode conversion
        with Image.open(abs_image_path) as img:
            image = img.convert("RGB")
            
        # Apply torchvision / callable transformation pipeline
        if self.transform is not None:
            image = self.transform(image)
            
        if self.return_metadata:
            metadata = {
                "id_code": id_code,
                "image_path": rel_image_path,
                "source_image_id": str(row.get("source_image_id", "")),
                "wagner_grade": label,
                "class_name": CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else f"Grade {label+1}"
            }
            return image, label, metadata
            
        return image, label


def _test_foot_dataset():
    """Unit test function for FootDFUDataset."""
    print("Testing FootDFUDataset class...")
    from src.foot.config import TRAIN_SPLIT_CSV, VAL_SPLIT_CSV, TEST_SPLIT_CSV
    
    for split_name, csv_p in [("Train", TRAIN_SPLIT_CSV), ("Val", VAL_SPLIT_CSV), ("Test", TEST_SPLIT_CSV)]:
        if not csv_p.exists():
            print(f"Skipping test for {split_name}: {csv_p} not found.")
            continue
            
        dataset = FootDFUDataset(csv_file=csv_p, return_metadata=True)
        print(f"Loaded {split_name} dataset with {len(dataset):,} samples.")
        
        # Test item retrieval
        img, label, meta = dataset[0]
        assert isinstance(img, Image.Image), f"Expected PIL Image, got {type(img)}"
        assert isinstance(label, int), f"Expected int label, got {type(label)}"
        assert 0 <= label <= 3, f"Label {label} out of bounds [0..3]"
        assert "source_image_id" in meta, "Missing source_image_id in metadata"
        print(f" - Sample 0: ID={meta['id_code']}, Grade={label} ({meta['class_name']}), Size={img.size}, Mode={img.mode}")

    print("FootDFUDataset class verified successfully!")

if __name__ == "__main__":
    _test_foot_dataset()
