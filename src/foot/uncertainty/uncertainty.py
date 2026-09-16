import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.foot.uncertainty.run_uncertainty_experiment import run_uncertainty_experiment

def run_uncertainty_pipeline():
    return run_uncertainty_experiment()

if __name__ == "__main__":
    run_uncertainty_pipeline()
