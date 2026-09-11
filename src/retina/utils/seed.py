import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Sets the seed for python, numpy, and torch to ensure reproducible results.
    Configures PyTorch to use deterministic algorithms when running on GPU.
    
    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Guarantees deterministic behavior for convolution operators
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
