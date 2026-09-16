import sys
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.foot.config import VAL_SPLIT_CSV, TEST_SPLIT_CSV, RAW_DATA
from src.foot.data.dataloader import create_foot_dataloaders
from src.foot.models.factory import load_foot_final_model
from src.foot.calibration.vector_scaling import FootVectorScaler
from src.foot.uncertainty.mc_dropout import enable_foot_mc_dropout
from src.foot.uncertainty.metrics import compute_mc_uncertainty_metrics, compute_error_detection_metrics
from src.foot.uncertainty.convergence import run_mc_convergence_study
from src.foot.uncertainty.risk_coverage import compute_risk_coverage_curve, evaluate_risk_coverage
from src.foot.uncertainty.evaluation import (
    evaluate_classwise_uncertainty,
    evaluate_boundary_uncertainty,
    extract_high_uncertainty_samples,
    generate_uncertainty_explainability_overlays
)

def run_uncertainty_experiment():
    print("=============================================================")
    print("Foot Ulcer Prediction Uncertainty Experiment")
    print("=============================================================\n", flush=True)
    
    exp_dir = Path(__file__).resolve().parents[3] / "experiments" / "foot" / "uncertainty"
    final_model_dir = Path(__file__).resolve().parents[3] / "experiments" / "foot" / "final_model"
    fig_dir = exp_dir / "figures"
    research_img_dir = Path(__file__).resolve().parents[3] / "research" / "foot" / "Volume_08_Prediction_Uncertainty" / "images"
    explain_dir = exp_dir / "high_uncertainty_explainability"
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    research_img_dir.mkdir(parents=True, exist_ok=True)
    explain_dir.mkdir(parents=True, exist_ok=True)
    
    # Config
    config = {
        "modality": "foot",
        "task": "4-class Wagner diabetic foot ulcer classification",
        "model": "efficientnet_b3",
        "checkpoint": "experiments/foot/architecture_benchmark/efficientnet_b3/checkpoints/best_model.pt",
        "calibration": "vector_scaling",
        "calibration_artifact": "experiments/foot/final_model/calibration.json",
        "seed": 42,
        "validation_samples": 1006,
        "test_samples": 1006,
        "stochastic_passes_N": "dynamic_from_convergence",
        "pipeline_mode": "Option B (Stochastic Logits -> Frozen Vector Scaling -> Softmax -> Uncertainty)"
    }
    
    # -------------------------------------------------------------
    # [1/10] Loading Model
    # -------------------------------------------------------------
    print("[1/10] Loading frozen EfficientNet-B3 model...", flush=True)
    model = load_foot_final_model(device="cpu")
    print("       DONE\n", flush=True)
    
    # -------------------------------------------------------------
    # [2/10] Loading Calibrator
    # -------------------------------------------------------------
    print("[2/10] Loading frozen Vector Scaling calibrator...", flush=True)
    calib_json_path = final_model_dir / "calibration.json"
    with open(calib_json_path, "r") as f:
        calib_info = json.load(f)
        
    vector_scaler = FootVectorScaler(num_classes=4)
    vector_scaler.weights.data = torch.tensor(calib_info["weights"], dtype=torch.float32)
    vector_scaler.bias.data = torch.tensor(calib_info["bias"], dtype=torch.float32)
    vector_scaler.eval()
    print("       DONE\n", flush=True)
    
    # -------------------------------------------------------------
    # [3/10] Creating DataLoaders
    # -------------------------------------------------------------
    print("[3/10] Creating DataLoaders...", flush=True)
    train_loader, val_loader, test_loader = create_foot_dataloaders(batch_size=32, num_workers=0, pin_memory=False)
    print(f"       Train: {len(train_loader.dataset)}")
    print(f"       Val:   {len(val_loader.dataset)}")
    print(f"       Test:  {len(test_loader.dataset)}")
    print("       DONE\n", flush=True)
    
    test_csv_df = pd.read_csv(TEST_SPLIT_CSV)
    id_codes = test_csv_df["id_code"].values if "id_code" in test_csv_df.columns else np.array([f"sample_{i}" for i in range(len(test_csv_df))])

    # -------------------------------------------------------------
    # [4/10] MC Dropout Convergence
    # -------------------------------------------------------------
    print("[4/10] MC Dropout convergence", flush=True)
    conv_results = run_mc_convergence_study(model, vector_scaler, val_loader, pass_counts=[5, 10, 15, 20, 25, 30])
    
    config["stochastic_passes_N"] = conv_results["selected_pass_count"]
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(exp_dir / "convergence_study.json", "w") as f:
        json.dump(conv_results, f, indent=2)
        
    pd.DataFrame(conv_results["convergence_records"]).to_csv(exp_dir / "convergence_results.csv", index=False)
        
    # Plot Convergence Curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = [r["pass_count"] for r in conv_results["convergence_records"]]
    entropies = [r["mean_predictive_entropy"] for r in conv_results["convergence_records"]]
    
    ax.plot(counts, entropies, "o-", color="royalblue", label="Predictive Entropy H(p_bar)", linewidth=2)
    ax.set_xlabel("MC Dropout Pass Count N", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Entropy (nats)", fontsize=11, fontweight="bold")
    ax.set_title("MC Dropout Pass Count Convergence Study", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(fig_dir / "convergence_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(research_img_dir / "fig1_mc_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # -------------------------------------------------------------
    # [5/10] Held-Out Test MC Inference
    # -------------------------------------------------------------
    num_passes = conv_results["selected_pass_count"]
    print(f"[5/10] Held-out test MC inference (N*={num_passes})", flush=True)

    enable_foot_mc_dropout(model)

    test_stochastic_passes = []
    total_batches = len(test_loader)
    overall_start = time.perf_counter()

    with torch.no_grad():
        for pass_idx in range(num_passes):
            pass_start = time.perf_counter()
            pass_probs = []
            pass_labels = []

            for batch_idx, (images, labels) in enumerate(test_loader, start=1):
                logits = model(images)
                logits_tensor = logits["logits"] if isinstance(logits, dict) else logits
                scaled_logits = vector_scaler(logits_tensor)
                probs = F.softmax(scaled_logits, dim=1)

                pass_probs.append(probs)

                if pass_idx == 0:
                    pass_labels.append(labels)

                if batch_idx % 10 == 0 or batch_idx == total_batches:
                    elapsed = time.perf_counter() - pass_start
                    progress = batch_idx / total_batches * 100
                    print(
                        f"\r       Pass {pass_idx + 1:02d}/{num_passes} | "
                        f"Batch {batch_idx:03d}/{total_batches:03d} ({progress:5.1f}%) | "
                        f"Elapsed: {elapsed:6.1f}s",
                        end="",
                        flush=True
                    )

            test_stochastic_passes.append(torch.cat(pass_probs, dim=0))

            if pass_idx == 0:
                test_labels_tensor = torch.cat(pass_labels, dim=0)

            pass_time = time.perf_counter() - pass_start
            print(f"\n       Pass {pass_idx + 1:02d}/{num_passes} completed in {pass_time:.1f}s", flush=True)

    overall_time = time.perf_counter() - overall_start
    print(f"\n       DONE — {overall_time / 60:.2f} min\n", flush=True)

    mc_probs_tensor = torch.stack(test_stochastic_passes, dim=0) # [N*, 1006, 4]
    np.save(exp_dir / "mc_probabilities.npy", mc_probs_tensor.numpy())
    
    # -------------------------------------------------------------
    # [6/10] Error Detection
    # -------------------------------------------------------------
    print("[6/10] Error detection", flush=True)
    metrics = compute_mc_uncertainty_metrics(mc_probs_tensor)
    
    pred_means = metrics["predictive_mean"].numpy() # [1006, 4]
    pred_vars = metrics["predictive_variance"].numpy() # [1006]
    pred_ents = metrics["predictive_entropy"].numpy() # [1006]
    exp_ents = metrics["expected_entropy"].numpy() # [1006]
    mut_infos = metrics["mutual_information"].numpy() # [1006]
    
    test_labels_np = test_labels_tensor.numpy()
    test_preds_np = np.argmax(pred_means, axis=1)
    is_error = (test_preds_np != test_labels_np).astype(int) # 1 if error, 0 if correct
    
    metrics_var = compute_error_detection_metrics(pred_vars, is_error)
    metrics_ent = compute_error_detection_metrics(pred_ents, is_error)
    metrics_mi  = compute_error_detection_metrics(mut_infos, is_error)
    
    error_detection_summary = {
        "predictive_variance": metrics_var,
        "predictive_entropy": metrics_ent,
        "mutual_information": metrics_mi,
        "baseline_error_rate": round(float(np.mean(is_error)), 4)
    }
    with open(exp_dir / "error_detection_metrics.json", "w") as f:
        json.dump(error_detection_summary, f, indent=2)
        
    print(f"       Variance AUROC/AUPRC: {metrics_var['auroc']:.4f} / {metrics_var['auprc']:.4f}")
    print(f"       Entropy AUROC/AUPRC:  {metrics_ent['auroc']:.4f} / {metrics_ent['auprc']:.4f}")
    print(f"       MI AUROC/AUPRC:       {metrics_mi['auroc']:.4f} / {metrics_mi['auprc']:.4f}\n", flush=True)
    
    # -------------------------------------------------------------
    # [7/10] Classwise + Boundary Analysis
    # -------------------------------------------------------------
    print("[7/10] Classwise + boundary analysis", flush=True)
    ranks = (len(pred_vars) - np.argsort(np.argsort(pred_vars))).astype(int)
    
    df_uncertainty = pd.DataFrame({
        "sample_index": np.arange(len(test_labels_np)),
        "id_code": id_codes[:len(test_labels_np)],
        "true_label": test_labels_np,
        "predicted_label": test_preds_np,
        "is_error": is_error,
        "prob_grade1": pred_means[:, 0],
        "prob_grade2": pred_means[:, 1],
        "prob_grade3": pred_means[:, 2],
        "prob_grade4": pred_means[:, 3],
        "predictive_confidence": np.max(pred_means, axis=1),
        "predictive_variance": pred_vars,
        "predictive_entropy": pred_ents,
        "expected_entropy": exp_ents,
        "mutual_information": mut_infos,
        "uncertainty_rank": ranks
    })
    df_uncertainty.to_csv(exp_dir / "test_uncertainty.csv", index=False)
    df_uncertainty.to_csv(exp_dir / "uncertainty_results.csv", index=False)
    
    df_classwise = evaluate_classwise_uncertainty(df_uncertainty)
    df_classwise.to_csv(exp_dir / "classwise_uncertainty.csv", index=False)
    
    df_risk_cov = evaluate_risk_coverage(pred_vars, is_error)
    df_risk_cov.to_csv(exp_dir / "risk_coverage.csv", index=False)
    
    boundary_results = evaluate_boundary_uncertainty(df_uncertainty)
    print("       DONE\n", flush=True)
    
    # -------------------------------------------------------------
    # [8/10] High-Uncertainty Cases + Grad-CAM
    # -------------------------------------------------------------
    print("[8/10] High-uncertainty cases + Grad-CAM", flush=True)
    df_high_unc = extract_high_uncertainty_samples(df_uncertainty, top_k=20)
    df_high_unc.to_csv(exp_dir / "high_uncertainty_cases.csv", index=False)
    print(f"       {len(df_high_unc)} unique cases")
    
    # Explicit deterministic eval mode for explainability
    model.eval()
    gen_count = 0
    try:
        gen_overlays = generate_uncertainty_explainability_overlays(
            model=model,
            vector_scaler=vector_scaler,
            test_loader=test_loader,
            high_unc_df=df_high_unc,
            output_dir=explain_dir
        )
        gen_count = len(gen_overlays)
    except Exception as e:
        print(f"       Explainability warning: {e}")
        
    print(f"       Generated: {gen_count}/{len(df_high_unc)}\n", flush=True)
    
    # -------------------------------------------------------------
    # [9/10] Generating Figures
    # -------------------------------------------------------------
    print("[9/10] Generating figures", flush=True)
    coverages, risk_rates = compute_risk_coverage_curve(pred_vars, is_error)
    fig_rc, ax_rc = plt.subplots(figsize=(6, 5))
    ax_rc.plot(coverages * 100, risk_rates * 100, "o-", color="crimson", linewidth=2, label="Foot Model (Variance Rejection)")
    ax_rc.axhline(float(np.mean(is_error)) * 100, color="gray", linestyle="--", label="Baseline Test Error Rate")
    ax_rc.set_xlabel("Coverage (%)", fontsize=11, fontweight="bold")
    ax_rc.set_ylabel("Risk / Error Rate (%)", fontsize=11, fontweight="bold")
    ax_rc.set_title("Risk-Coverage Curve (Foot MC Variance)", fontsize=12, fontweight="bold")
    ax_rc.grid(True, linestyle=":", alpha=0.6)
    ax_rc.legend(loc="upper left")
    plt.tight_layout()
    fig_rc.savefig(exp_dir / "risk_coverage.png", dpi=300, bbox_inches="tight")
    fig_rc.savefig(fig_dir / "risk_coverage.png", dpi=300, bbox_inches="tight")
    fig_rc.savefig(research_img_dir / "fig2_risk_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig_rc)
    
    # Uncertainty Distribution (Correct vs Error)
    fig_dist, ax_dist = plt.subplots(figsize=(7, 4.5))
    correct_vars = pred_vars[is_error == 0]
    error_vars = pred_vars[is_error == 1]
    
    ax_dist.hist(correct_vars, bins=25, alpha=0.5, label="Correct Predictions", color="royalblue", edgecolor="black", density=True)
    ax_dist.hist(error_vars, bins=25, alpha=0.5, label="Incorrect Predictions", color="crimson", edgecolor="black", density=True)
    ax_dist.set_xlabel("Predictive Variance", fontsize=11, fontweight="bold")
    ax_dist.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax_dist.set_title("Predictive Variance Distribution (Correct vs Error)", fontsize=12, fontweight="bold")
    ax_dist.grid(True, linestyle=":", alpha=0.6)
    ax_dist.legend(loc="upper right")
    plt.tight_layout()
    fig_dist.savefig(exp_dir / "uncertainty_distributions.png", dpi=300, bbox_inches="tight")
    fig_dist.savefig(fig_dir / "uncertainty_distributions.png", dpi=300, bbox_inches="tight")
    fig_dist.savefig(research_img_dir / "fig3_uncertainty_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig_dist)
    print("       DONE\n", flush=True)
    
    # -------------------------------------------------------------
    # [10/10] Writing Final Artifacts
    # -------------------------------------------------------------
    print("[10/10] Writing final artifacts", flush=True)
    summary = {
        "modality": "foot",
        "model": "efficientnet_b3",
        "stochastic_passes": num_passes,
        "test_samples": len(test_labels_np),
        "test_accuracy": round(float(np.mean(test_preds_np == test_labels_np)), 4),
        "test_error_rate": round(float(np.mean(is_error)), 4),
        "mean_predictive_variance": round(float(np.mean(pred_vars)), 6),
        "mean_predictive_entropy": round(float(np.mean(pred_ents)), 4),
        "mean_expected_entropy": round(float(np.mean(exp_ents)), 4),
        "mean_mutual_information": round(float(np.mean(mut_infos)), 4),
        "error_detection": error_detection_summary,
        "grade2_grade3_boundary_analysis": boundary_results,
        "classwise_uncertainty": df_classwise.to_dict(orient="records")
    }
    
    with open(exp_dir / "uncertainty_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    final_unc_manifest = {
        "modality": "foot",
        "model": "efficientnet_b3",
        "checkpoint": "experiments/foot/architecture_benchmark/efficientnet_b3/checkpoints/best_model.pt",
        "calibration": "vector_scaling",
        "stochastic_passes_N": num_passes,
        "mean_predictive_variance": summary["mean_predictive_variance"],
        "mean_predictive_entropy": summary["mean_predictive_entropy"],
        "mean_mutual_information": summary["mean_mutual_information"],
        "error_detection_auroc_var": metrics_var["auroc"],
        "error_detection_auprc_var": metrics_var["auprc"]
    }
    with open(final_model_dir / "uncertainty.json", "w") as f:
        json.dump(final_unc_manifest, f, indent=2)
    print("       DONE\n", flush=True)
        
    print("=============================================================")
    print("UNCERTAINTY EXPERIMENT COMPLETED")
    print("=============================================================", flush=True)
    return summary

if __name__ == "__main__":
    run_uncertainty_experiment()
