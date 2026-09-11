import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.foot.data.eda_image_quality import run_image_quality_eda

if __name__ == "__main__":
    run_image_quality_eda()
