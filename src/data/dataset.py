import sys
from pathlib import Path
from typing import Union, Callable, Optional, Tuple, Any
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import ID_COLUMN, LABEL_COLUMN

class RetinaDataset(Dataset):
    """
    A PyTorch Dataset for loading retina fundus images and their associated
    Diabetic Retinopathy (DR) diagnosis labels from a CSV split file.
    """
    
    def __init__(
        self,
        csv_file: Union[str, Path],
        image_dir: Union[str, Path],
        transform: Optional[Callable] = None
    ) -> None:
        """
        Args:
            csv_file: Path to the split CSV file (e.g., train.csv).
            image_dir: Path to the directory containing raw images.
            transform: Optional torchvision transform to apply to the image.
            
        Raises:
            ValueError: If the CSV file does not contain the required columns.
        """
        self.csv_file = Path(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        self.dataframe = pd.read_csv(self.csv_file)
        
        # Validate CSV columns
        if ID_COLUMN not in self.dataframe.columns:
            raise ValueError(
                f"CSV file '{self.csv_file}' is missing the required ID column '{ID_COLUMN}'."
            )
        if LABEL_COLUMN not in self.dataframe.columns:
            raise ValueError(
                f"CSV file '{self.csv_file}' is missing the required label column '{LABEL_COLUMN}'."
            )

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Retrieves the image and label at the specified index.
        
        Args:
            index: The index of the item to retrieve.
            
        Returns:
            Tuple[Any, int]: A tuple containing the loaded image (PIL Image or 
                             transformed object like a Tensor) and the integer label.
                             
        Raises:
            FileNotFoundError: If the image file associated with the record is missing.
        """
        row = self.dataframe.iloc[index]
        id_code = row[ID_COLUMN]
        label = int(row[LABEL_COLUMN])
        
        image_path = self.image_dir / f"{id_code}.png"
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found: '{image_path}' for record at index {index} (ID: {id_code})."
            )
            
        # Load image and guarantee RGB mode
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        
        # Apply transformation if provided
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
