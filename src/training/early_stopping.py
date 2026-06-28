class EarlyStopping:
    """
    Early stopping helper to monitor validation metrics and signal termination 
    if no improvement is observed for a configured number of patience epochs.
    """
    
    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 0.0
    ) -> None:
        """
        Args:
            patience: Number of epochs to wait without improvement.
            mode: Metric optimization target ('min' for loss, 'max' for QWK/F1).
            min_delta: Minimum change in the monitored value to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode.lower()
        self.min_delta = min_delta
        
        self.counter = 0
        self.early_stop = False
        
        if self.mode == "min":
            self.best_score = float('inf')
        elif self.mode == "max":
            self.best_score = float('-inf')
        else:
            raise ValueError(f"Unsupported early stopping mode: '{self.mode}'. Choose 'min' or 'max'.")
            
    def __call__(self, current_score: float) -> bool:
        """
        Registers the score of the current epoch.
        
        Args:
            current_score: Current validation metric score.
            
        Returns:
            bool: True if training should be halted, False otherwise.
        """
        if self.mode == "min":
            improvement = current_score < (self.best_score - self.min_delta)
        else:
            improvement = current_score > (self.best_score + self.min_delta)
            
        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
