import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.foot.data.eda_sampling_strategy import run_sampling_strategy_eda

if __name__ == "__main__":
    run_sampling_strategy_eda()
