"""
Common Image and Array Utilities for Quality Layer (Phase C11.2).
Provides zero-crash loading and decoding safeguards.
"""

from typing import Optional, Union, Tuple
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


def load_and_decode_image(
    image_input: Union[str, Path, np.ndarray, Image.Image]
) -> Tuple[bool, Optional[np.ndarray], str]:
    """
    Safely decode an image from a path, numpy array, or PIL Image.
    
    Returns:
        (success: bool, rgb_array: Optional[np.ndarray], status_reason: str)
    """
    if image_input is None:
        return False, None, "INPUT_IS_NONE"

    try:
        # Case 1: File path
        if isinstance(image_input, (str, Path)):
            path_obj = Path(image_input)
            if not path_obj.exists() or not path_obj.is_file():
                return False, None, "FILE_NOT_FOUND"
            
            # Read using OpenCV
            bgr_img = cv2.imread(str(path_obj))
            if bgr_img is None or bgr_img.size == 0:
                # Try fallback via PIL
                try:
                    pil_img = Image.open(path_obj).convert("RGB")
                    rgb_img = np.array(pil_img)
                    return True, rgb_img, "DECODE_SUCCESS_PIL"
                except Exception:
                    return False, None, "DECODE_FAILED"
            
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            return True, rgb_img, "DECODE_SUCCESS"

        # Case 2: PIL Image
        elif isinstance(image_input, Image.Image):
            rgb_img = np.array(image_input.convert("RGB"))
            return True, rgb_img, "PIL_CONVERT_SUCCESS"

        # Case 3: Numpy array
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                return False, None, "EMPTY_NUMPY_ARRAY"
            
            if image_input.ndim == 2:
                # Grayscale to RGB
                rgb_img = np.stack([image_input] * 3, axis=-1)
                return True, rgb_img, "GRAYSCALE_TO_RGB"
            elif image_input.ndim == 3 and image_input.shape[2] in (1, 3, 4):
                if image_input.shape[2] == 1:
                    rgb_img = np.repeat(image_input, 3, axis=2)
                elif image_input.shape[2] == 4:
                    rgb_img = image_input[:, :, :3]
                else:
                    rgb_img = image_input
                return True, rgb_img, "NUMPY_ARRAY_SUCCESS"
            else:
                return False, None, f"INVALID_ARRAY_SHAPE_{image_input.shape}"

        else:
            return False, None, f"UNSUPPORTED_TYPE_{type(image_input).__name__}"

    except Exception as e:
        return False, None, f"LOAD_EXCEPTION_{type(e).__name__}"
