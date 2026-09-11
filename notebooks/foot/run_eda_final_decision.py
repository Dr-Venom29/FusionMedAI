import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.foot.data.eda_final_decision import run_final_eda_decision

if __name__ == "__main__":
    run_final_eda_decision()
