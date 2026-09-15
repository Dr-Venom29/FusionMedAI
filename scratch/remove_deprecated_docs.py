from pathlib import Path

target_dir = Path("d:/FusionMedAI/research/foot/Volume_07_Probability_Calibration")
deprecated_files = [
    target_dir / "04_Validation_Fitting_Results.md",
    target_dir / "05_Test_Evaluation_Results.md"
]

for f in deprecated_files:
    if f.exists():
        f.unlink()
        print(f"Removed deprecated file: {f}")
    else:
        print(f"File not found: {f}")
