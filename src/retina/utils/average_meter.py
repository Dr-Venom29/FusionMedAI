class AverageMeter:
    """
    Computes and stores the average and current value of a tracked metric.
    Commonly used for tracking loss, accuracy, etc. across training/validation batches.
    """
    
    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()
        
    def reset(self) -> None:
        """Resets all internal statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        
    def update(self, val: float, n: int = 1) -> None:
        """
        Updates the tracker with a new value and its associated batch size/weight.
        
        Args:
            val: The current batch value.
            n: Number of samples in the batch.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
