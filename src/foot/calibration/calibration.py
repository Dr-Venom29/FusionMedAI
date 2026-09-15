import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.foot.config import VAL_SPLIT_CSV, TEST_SPLIT_CSV, RAW_DATA
from src.foot.data.transforms import get_foot_test_transforms
from src.foot.models.factory import load_foot_final_model
from src.foot.calibration.metrics import compute_calibration_metrics
from src.foot.calibration.temperature_scaling import FootTemperatureScaler
from src.foot.calibration.vector_scaling import FootVectorScaler
from src.foot.calibration.reliability import (
    plot_reliability_diagram,
    plot_reliability_comparison,
    plot_confidence_distribution,
    plot_classwise_reliability
)
from src.foot.calibration.evaluation import (
    evaluate_calibrated_model,
    generate_calibration_results_dataframe,
    generate_classwise_calibration_dataframe
)

def extract_split_logits(model, csv_path: Path, transform) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts raw logits and ground-truth labels for a given split CSV."""
    df = pd.read_csv(csv_path)
    all_logits = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for idx, row in df.iterrows():
            rel_path = row["image_path"]
            label = int(row["wagner_grade"])
            
            full_path = Path(RAW_DATA) / rel_path
            if not full_path.exists():
                full_path = Path(__file__).resolve().parents[3] / "datasets" / "foot" / "raw" / rel_path
                
            pil_img = Image.open(full_path).convert("RGB")
            tensor_img = transform(pil_img).unsqueeze(0) # [1, 3, 224, 224]
            
            out = model(tensor_img)
            logits = out["logits"] if isinstance(out, dict) else out
            
            all_logits.append(logits.cpu())
            all_labels.append(torch.tensor([label]))
            
    return torch.cat(all_logits), torch.cat(all_labels)

def run_calibration_pipeline():
    print("=============================================================")
    print("Phase 10.7 — Foot Ulcer Probability Calibration Pipeline")
    print("=============================================================\n", flush=True)
    
    exp_dir = Path(__file__).resolve().parents[3] / "experiments" / "foot" / "calibration"
    final_model_dir = Path(__file__).resolve().parents[3] / "experiments" / "foot" / "final_model"
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = exp_dir / "temperature_scaling"
    vec_dir = exp_dir / "vector_scaling"
    temp_dir.mkdir(parents=True, exist_ok=True)
    vec_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Config export
    config = {
        "modality": "foot",
        "task": "4-class Wagner diabetic foot ulcer classification",
        "selected_model": "efficientnet_b3",
        "checkpoint": "experiments/foot/architecture_benchmark/efficientnet_b3/checkpoints/best_model.pt",
        "seed": 42,
        "input_resolution": [224, 224],
        "validation_samples": 1006,
        "test_samples": 1006,
        "binning_protocol": "10 fixed equal-width bins [0.0..1.0]",
        "calibration_methods": ["Uncalibrated", "Temperature Scaling", "Vector Scaling"],
        "selection_rule": "Primary: lower validation NLL. Predefined project parsimony tolerance: if validation NLL difference (Temp vs Vector) <= 0.005, prefer Temperature Scaling for 1-parameter model simplicity and strict argmax prediction preservation."
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    print("1. Loading frozen EfficientNet-B3 model...", flush=True)
    model = load_foot_final_model(device="cpu")
    transform = get_foot_test_transforms()
    
    print("2. Extracting raw validation logits (N=1,006)...", flush=True)
    val_logits, val_labels = extract_split_logits(model, VAL_SPLIT_CSV, transform)
    val_probs = F.softmax(val_logits, dim=1)
    
    print("3. Extracting raw test logits (N=1,006)...", flush=True)
    test_logits, test_labels = extract_split_logits(model, TEST_SPLIT_CSV, transform)
    test_probs = F.softmax(test_logits, dim=1)
    
    # Save raw arrays
    np.save(exp_dir / "validation_logits.npy", val_logits.numpy())
    np.save(exp_dir / "validation_labels.npy", val_labels.numpy())
    np.save(exp_dir / "test_logits.npy", test_logits.numpy())
    np.save(exp_dir / "test_labels.npy", test_labels.numpy())
    
    # Raw metrics
    raw_val_metrics = evaluate_calibrated_model(val_logits, val_labels, "Uncalibrated (Raw)")
    raw_test_metrics = evaluate_calibrated_model(test_logits, test_labels, "Uncalibrated (Raw)")
    
    with open(exp_dir / "raw_metrics.json", "w") as f:
        json.dump({"validation": raw_val_metrics, "test": raw_test_metrics}, f, indent=2)
        
    # -------------------------------------------------------------
    # Fitting Calibrators EXCLUSIVELY on Validation Set
    # -------------------------------------------------------------
    print("\n4. Fitting Temperature Scaler on Validation set...", flush=True)
    temp_scaler = FootTemperatureScaler(initial_temperature=1.0)
    opt_temp = temp_scaler.fit(val_logits, val_labels)
    print(f"   Optimal Validation Temperature T*: {opt_temp:.4f}", flush=True)
    
    temp_params = {
        "method": "temperature_scaling",
        "temperature": round(opt_temp, 4),
        "fit_split": "validation",
        "seed": 42
    }
    with open(temp_dir / "calibration_params.json", "w") as f:
        json.dump(temp_params, f, indent=2)
    torch.save({"temperature": opt_temp, "state_dict": temp_scaler.state_dict()}, temp_dir / "temperature_scaling.pt")
    
    print("5. Fitting Vector Scaler on Validation set...", flush=True)
    vec_scaler = FootVectorScaler(num_classes=4)
    opt_vec = vec_scaler.fit(val_logits, val_labels)
    print(f"   Optimal Validation Scaling Weights: {opt_vec['weights']}", flush=True)
    print(f"   Optimal Validation Bias Vector: {opt_vec['bias']}", flush=True)
    
    vec_params = {
        "method": "vector_scaling",
        "weights": [round(float(w), 4) for w in opt_vec["weights"]],
        "bias": [round(float(b), 4) for b in opt_vec["bias"]],
        "fit_split": "validation",
        "seed": 42
    }
    with open(vec_dir / "calibration_params.json", "w") as f:
        json.dump(vec_params, f, indent=2)
    torch.save({"weights": opt_vec["weights"], "bias": opt_vec["bias"], "state_dict": vec_scaler.state_dict()}, vec_dir / "vector_scaling.pt")
    
    # -------------------------------------------------------------
    # Computing Metrics on Validation & Held-Out Test Sets
    # -------------------------------------------------------------
    temp_val_logits = temp_scaler(val_logits)
    temp_test_logits = temp_scaler(test_logits)
    
    vec_val_logits = vec_scaler(val_logits)
    vec_test_logits = vec_scaler(test_logits)
    
    temp_val_metrics = evaluate_calibrated_model(temp_val_logits, val_labels, "Temperature Scaling")
    temp_test_metrics = evaluate_calibrated_model(temp_test_logits, test_labels, "Temperature Scaling")
    
    vec_val_metrics = evaluate_calibrated_model(vec_val_logits, val_labels, "Vector Scaling")
    vec_test_metrics = evaluate_calibrated_model(vec_test_logits, test_labels, "Vector Scaling")
    
    with open(temp_dir / "validation_metrics.json", "w") as f:
        json.dump(temp_val_metrics, f, indent=2)
    with open(temp_dir / "test_metrics.json", "w") as f:
        json.dump(temp_test_metrics, f, indent=2)
        
    with open(vec_dir / "validation_metrics.json", "w") as f:
        json.dump(vec_val_metrics, f, indent=2)
    with open(vec_dir / "test_metrics.json", "w") as f:
        json.dump(vec_test_metrics, f, indent=2)
        
    # Comparison artifact
    comparison = {
        "validation_evaluation": {
            "Raw": raw_val_metrics,
            "Temperature_Scaling": temp_val_metrics,
            "Vector_Scaling": vec_val_metrics
        },
        "test_evaluation": {
            "Raw": raw_test_metrics,
            "Temperature_Scaling": temp_test_metrics,
            "Vector_Scaling": vec_test_metrics
        }
    }
    with open(exp_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
        
    # Tabular CSV exports
    df_results = generate_calibration_results_dataframe(raw_test_metrics, temp_test_metrics, vec_test_metrics)
    df_results.to_csv(exp_dir / "calibration_results.csv", index=False)
    
    # -------------------------------------------------------------
    # Selection Rule: Validation NLL / Parsimony Tolerance
    # -------------------------------------------------------------
    val_nll_temp = temp_val_metrics["NLL"]
    val_nll_vec = vec_val_metrics["NLL"]
    
    if val_nll_temp <= val_nll_vec + 0.005:
        selected_method = "temperature_scaling"
        selected_test_logits = temp_test_logits
        selection_reason = "Selected via parsimony preference: validation NLL delta <= 0.005 tolerance threshold over vector scaling, using 1 parameter and strictly preserving prediction rank."
        final_calib_artifact = {
            "method": "temperature_scaling",
            "temperature": round(opt_temp, 4),
            "fit_split": "validation",
            "validation_nll": val_nll_temp,
            "test_nll": temp_test_metrics["NLL"],
            "test_ece": temp_test_metrics["ECE"],
            "selection_rationale": selection_reason,
            "seed": 42,
            "status": "selected"
        }
    else:
        selected_method = "vector_scaling"
        selected_test_logits = vec_test_logits
        selection_reason = "Selected via lower validation NLL exceeding the 0.005 parsimony tolerance threshold."
        final_calib_artifact = {
            "method": "vector_scaling",
            "weights": [round(float(w), 4) for w in opt_vec["weights"]],
            "bias": [round(float(b), 4) for b in opt_vec["bias"]],
            "fit_split": "validation",
            "validation_nll": val_nll_vec,
            "test_nll": vec_test_metrics["NLL"],
            "test_ece": vec_test_metrics["ECE"],
            "selection_rationale": selection_reason,
            "seed": 42,
            "status": "selected"
        }
        
    with open(final_model_dir / "calibration.json", "w") as f:
        json.dump(final_calib_artifact, f, indent=2)
        
    model_selection_artifact = {
        "model": "efficientnet_b3",
        "checkpoint": "experiments/foot/architecture_benchmark/efficientnet_b3/checkpoints/best_model.pt",
        "input_resolution": [224, 224],
        "calibration": final_calib_artifact
    }
    with open(final_model_dir / "model_selection.json", "w") as f:
        json.dump(model_selection_artifact, f, indent=2)
        
    # Classwise diagnostic export for selected method
    test_probs_selected = F.softmax(selected_test_logits, dim=1)
    df_classwise = generate_classwise_calibration_dataframe(test_probs_selected, test_labels, selected_method.replace("_", " ").title())
    df_classwise.to_csv(exp_dir / "classwise_calibration.csv", index=False)
    
    # -------------------------------------------------------------
    # Generating Reliability Diagrams & Figures
    # -------------------------------------------------------------
    print("\n6. Generating Reliability Diagrams & Figures...", flush=True)
    import matplotlib.pyplot as plt
    
    fig_raw = plot_reliability_diagram(test_probs, test_labels, title="Uncalibrated (Raw) Reliability Diagram", num_bins=10, save_path=exp_dir / "reliability_diagram.png")
    plt.close(fig_raw)
    
    method_probs_dict = {
        "Uncalibrated (Raw)": test_probs,
        f"Temperature Scaling (T*={opt_temp:.2f})": F.softmax(temp_test_logits, dim=1),
        "Vector Scaling": F.softmax(vec_test_logits, dim=1)
    }
    fig_dist = plot_confidence_distribution(method_probs_dict, save_path=exp_dir / "confidence_distribution.png")
    plt.close(fig_dist)
    
    # Save image in research volume images directory
    research_img_dir = Path(__file__).resolve().parents[3] / "research" / "foot" / "Volume_07_Probability_Calibration" / "images"
    research_img_dir.mkdir(parents=True, exist_ok=True)
    
    plot_reliability_comparison(method_probs_dict, test_labels, num_bins=10, save_path=research_img_dir / "fig1_reliability_comparison.png")
    plot_confidence_distribution(method_probs_dict, save_path=research_img_dir / "fig2_confidence_distribution.png")
    plot_classwise_reliability(test_probs_selected, test_labels, save_path=research_img_dir / "fig3_classwise_calibration.png")
    
    print("\n=============================================================")
    print("HELD-OUT TEST SET CALIBRATION EVALUATION SUMMARY:")
    print("-------------------------------------------------------------")
    print(f"Method A (Uncalibrated):           ECE={raw_test_metrics['ECE']:.4f}, NLL={raw_test_metrics['NLL']:.4f}, Brier={raw_test_metrics['Brier']:.4f}, Macro F1={raw_test_metrics['Macro F1']:.4f}")
    print(f"Method B (Temperature Scaling T*={opt_temp:.4f}): ECE={temp_test_metrics['ECE']:.4f}, NLL={temp_test_metrics['NLL']:.4f}, Brier={temp_test_metrics['Brier']:.4f}, Macro F1={temp_test_metrics['Macro F1']:.4f}")
    print(f"Method C (Vector Scaling):         ECE={vec_test_metrics['ECE']:.4f}, NLL={vec_test_metrics['NLL']:.4f}, Brier={vec_test_metrics['Brier']:.4f}, Macro F1={vec_test_metrics['Macro F1']:.4f}")
    print(f"\nSelected Calibration Method: {selected_method.upper()}")
    print("=============================================================", flush=True)

if __name__ == "__main__":
    run_calibration_pipeline()
