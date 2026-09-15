import sys
from pathlib import Path
from typing import Union, Callable, Optional, Tuple, Dict, Any
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.config import RAW_DATA, ID_COLUMN, LABEL_COLUMN, CLASS_NAMES, PROJECT_ROOT

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
        """
        super().__init__()
        self.csv_file = Path(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.return_metadata = return_metadata

        if not self.csv_file.exists():
            raise FileNotFoundError(f"Split CSV manifest missing at path: '{self.csv_file}'")

        self.dataframe = pd.read_csv(self.csv_file)

        # Validate required columns in manifest
        required_cols = [ID_COLUMN, LABEL_COLUMN, "image_path"]
        for col in required_cols:
            if col not in self.dataframe.columns:
                raise KeyError(f"Required column '{col}' missing from split CSV '{self.csv_file}'")

    def __len__(self) -> int:
        """Returns total number of samples in dataset split."""
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Union[Tuple[Any, int], Tuple[Any, int, Dict[str, Any]]]:
        """
        Loads and returns single sample at given index.
        
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
            # Kaggle / environment fallback search paths
            candidate_paths = [
                Path(rel_image_path),
                PROJECT_ROOT / rel_image_path,
                self.csv_file.parents[2] / "raw" / rel_image_path,
                self.csv_file.parents[1] / rel_image_path,
                self.csv_file.parents[2] / rel_image_path,
            ]
            found = False
            for cand in candidate_paths:
                if cand.exists():
                    abs_image_path = cand
                    found = True
                    break
            if not found:
                raise FileNotFoundError(
                    f"Image file missing on disk: '{abs_image_path}' for record index {index} (ID: {id_code})."
                )
            
        # Lazy image loading with guaranteed RGB mode conversion
        with Image.open(abs_image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_metadata:
            metadata = {
                "id_code": id_code,
                "image_path": rel_image_path,
                "abs_path": str(abs_image_path),
                "source_image_id": str(row.get("source_image_id", id_code)),
                "class_name": CLASS_NAMES.get(label, f"Grade_{label+1}")
            }
            return image, label, metadata

        return image, label


def _test_dataset():
    """Unit test for FootDFUDataset."""
    from src.foot.config import TRAIN_SPLIT_CSV
    print("Testing FootDFUDataset instantiation...")
    ds = FootDFUDataset(csv_file=TRAIN_SPLIT_CSV)
    print(f" - Train Dataset Length: {len(ds):,} samples")
    
    img, lbl, meta = ds.__getitem__(0) if hasattr(ds, '__getitem__') and ds.return_metadata else (*ds[0], {})
    print(f" - Sample 0 -> Image Type: {type(img)}, Label: {lbl}")
    assert lbl in [0, 1, 2, 3], f"Invalid Wagner grade label: {lbl}"
    print("FootDFUDataset verified successfully!")

if __name__ == "__main__":
    _test_dataset()
