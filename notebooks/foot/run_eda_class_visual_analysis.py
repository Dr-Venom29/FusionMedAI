import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.foot.data.eda_class_visual_analysis import run_class_wise_visual_analysis

if __name__ == "__main__":
    run_class_wise_visual_analysis()
