import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(name: str = "FusionMedAI", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a standard logger that outputs to both the console and a file.
    
    Args:
        name: Name of the logger.
        log_file: Optional path to the log file.
        level: Logging level (e.g. logging.INFO).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if setup is called multiple times
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(level)
    
    # Formatter for log messages
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if file path is provided)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
