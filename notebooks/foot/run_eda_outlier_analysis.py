import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.foot.data.eda_outlier_analysis import analyze_outliers

if __name__ == "__main__":
    analyze_outliers()
