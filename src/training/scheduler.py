import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = "cosine",
    epochs: int = 20,
    min_lr: float = 1e-6
) -> _LRScheduler:
    """
    Constructs and returns a learning rate scheduler.
    
    Args:
        optimizer: The optimizer to wrap.
        scheduler_type: Type of scheduler ('cosine').
        epochs: T_max value (normally matching total epochs).
        min_lr: Minimum learning rate floor for CosineAnnealing.
        
    Returns:
        _LRScheduler: Configured learning rate scheduler.
        
    Raises:
        ValueError: If scheduler_type is unsupported.
    """
    sched_type_lower = scheduler_type.lower()
    
    if sched_type_lower == "cosine" or sched_type_lower == "cosineannealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    else:
        raise ValueError(
            f"Unsupported scheduler type: '{scheduler_type}'. "
            f"Currently registered schedulers: ['cosine']"
        )
