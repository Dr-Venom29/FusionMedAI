import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.foot.data.eda_statistical_profiling import run_dataset_statistical_profiling

if __name__ == "__main__":
    run_dataset_statistical_profiling()
