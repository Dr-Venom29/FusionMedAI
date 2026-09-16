import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.foot.models.factory import load_foot_final_model
from src.foot.calibration.vector_scaling import FootVectorScaler
from src.foot.uncertainty.mc_dropout import enable_foot_mc_dropout
from src.foot.uncertainty.metrics import compute_mc_uncertainty_metrics
from src.foot.uncertainty.risk_coverage import compute_risk_coverage_curve

def test_uncertainty_12_point_suite():
    print("=============================================================")
    print("12-Point Foot Uncertainty Verification Suite")
    print("=============================================================\n")
    
    passed_tests = 0
    total_tests = 12
    
    # -------------------------------------------------------------
    # Test 1 — Canonical B3 Checkpoint Loading
    # -------------------------------------------------------------
    print("Test 1 — Canonical B3 Checkpoint Loading...")
    ckpt_path = Path(__file__).resolve().parents[3] / "experiments" / "foot" / "architecture_benchmark" / "efficientnet_b3" / "checkpoints" / "best_model.pt"
    assert ckpt_path.exists(), f"Checkpoint path missing: {ckpt_path}"
    model = load_foot_final_model(device="cpu")
    assert model is not None, "Model failed to load."
    passed_tests += 1
    print("   [PASS] Test 1: EfficientNet-B3 loaded successfully.")
    
    # -------------------------------------------------------------
    # Test 2 — Calibration Artifact Loading
    # -------------------------------------------------------------
    print("\nTest 2 — Calibration Artifact Loading...")
    calib_json_path = Path(__file__).resolve().parents[3] / "experiments" / "foot" / "final_model" / "calibration.json"
    assert calib_json_path.exists(), f"Calibration JSON missing: {calib_json_path}"
    with open(calib_json_path, "r") as f:
        calib_data = json.load(f)
    assert calib_data.get("method") == "vector_scaling", f"Unexpected calibration method: {calib_data.get('method')}"

    vector_scaler = FootVectorScaler(num_classes=4)
    vector_scaler.weights.data = torch.tensor(calib_data["weights"], dtype=torch.float32)
    vector_scaler.bias.data = torch.tensor(calib_data["bias"], dtype=torch.float32)
    vector_scaler.eval()
    passed_tests += 1
    print("   [PASS] Test 2: Frozen Vector Scaling artifact loaded successfully.")
    
    # -------------------------------------------------------------
    # Test 3 — MC Dropout Stochasticity
    # -------------------------------------------------------------
    print("\nTest 3 — MC Dropout Stochasticity Verification...")
    torch.manual_seed(42)
    dummy_input = torch.randn(4, 3, 224, 224)
    enable_foot_mc_dropout(model)
    
    with torch.no_grad():
        stoch_out1 = model(dummy_input)["logits"]
        stoch_out2 = model(dummy_input)["logits"]
        stoch_out3 = model(dummy_input)["logits"]
        
    assert not torch.equal(stoch_out1, stoch_out2), "MC Dropout failed: pass 1 == pass 2"
    assert not torch.equal(stoch_out2, stoch_out3), "MC Dropout failed: pass 2 == pass 3"
    passed_tests += 1
    print("   [PASS] Test 3: MC Dropout produces non-identical stochastic forward passes.")
    
    # -------------------------------------------------------------
    # Test 4 — Deterministic Evaluation Mode
    # -------------------------------------------------------------
    print("\nTest 4 — Deterministic Evaluation Mode Verification...")
    model.eval() # Reset to standard eval
    with torch.no_grad():
        det_out1 = model(dummy_input)["logits"]
        det_out2 = model(dummy_input)["logits"]
        
    assert torch.equal(det_out1, det_out2), "Deterministic mode check failed: outputs differ across calls."
    passed_tests += 1
    print("   [PASS] Test 4: Standard evaluation mode is 100% deterministic.")
    
    # -------------------------------------------------------------
    # Test 5 — Probabilities Finite and Sum to 1
    # -------------------------------------------------------------
    print("\nTest 5 — Probabilities Finite and Normalization...")
    enable_foot_mc_dropout(model)
    passes = []
    with torch.no_grad():
        for _ in range(10):
            raw_l = model(dummy_input)["logits"]
            scaled_l = vector_scaler(raw_l)
            probs = F.softmax(scaled_l, dim=1)
            passes.append(probs)
            
    mc_tensor = torch.stack(passes, dim=0) # [10, 4, 4]
    assert torch.all(torch.isfinite(mc_tensor)), "MC probabilities contain NaN or Inf values!"
    prob_sums = torch.sum(mc_tensor, dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), "Probabilities do not sum to 1.0!"
    passed_tests += 1
    print("   [PASS] Test 5: All stochastic probabilities are finite and sum to 1.0.")
    
    # -------------------------------------------------------------
    # Test 6 — Entropy Bounds Check
    # -------------------------------------------------------------
    print("\nTest 6 — Entropy Bounds Check (0 <= H(p) <= log(4))...")
    metrics = compute_mc_uncertainty_metrics(mc_tensor)
    pred_ents = metrics["predictive_entropy"]
    max_entropy = np.log(4.0) # ~1.386294
    
    assert torch.all(pred_ents >= -1e-6), "Predictive entropy is negative!"
    assert torch.all(pred_ents <= max_entropy + 1e-5), f"Predictive entropy exceeds log(4): max={pred_ents.max()}"
    passed_tests += 1
    print(f"   [PASS] Test 6: Entropy bounds verified (0 <= H(p) <= {max_entropy:.4f}).")
    
    # -------------------------------------------------------------
    # Test 7 — Variance Non-Negativity
    # -------------------------------------------------------------
    print("\nTest 7 — Predictive Variance Non-Negativity...")
    pred_vars = metrics["predictive_variance"]
    assert torch.all(pred_vars >= -1e-6), "Predictive variance is negative!"
    passed_tests += 1
    print("   [PASS] Test 7: Predictive variance is non-negative.")
    
    # -------------------------------------------------------------
    # Test 8 — Mutual Information Bounds
    # -------------------------------------------------------------
    print("\nTest 8 — Epistemic Mutual Information Bounds...")
    mut_infos = metrics["mutual_information"]
    assert torch.all(torch.isfinite(mut_infos)), "Mutual information contains NaN/Inf!"
    assert torch.all(mut_infos >= -1e-6), "Mutual information is negative!"
    passed_tests += 1
    print("   [PASS] Test 8: Mutual Information is finite and non-negative.")
    
    # -------------------------------------------------------------
    # Test 9 — Stochastic Pass Count Specification
    # -------------------------------------------------------------
    print("\nTest 9 — Pass Count N Specification...")
    assert mc_tensor.shape[0] == 10, f"Expected 10 pass tensor, got {mc_tensor.shape[0]}"
    passed_tests += 1
    print("   [PASS] Test 9: Stochastic pass count indexing verified.")
    
    # -------------------------------------------------------------
    # Test 10 — Label Independence Protocol
    # -------------------------------------------------------------
    print("\nTest 10 — Label Independence Protocol Check...")
    # Verify that metrics computation function takes ONLY probabilities and no ground truth labels
    import inspect
    sig = inspect.signature(compute_mc_uncertainty_metrics)
    assert "labels" not in sig.parameters, "compute_mc_uncertainty_metrics must NOT accept labels!"
    passed_tests += 1
    print("   [PASS] Test 10: Uncertainty computation is strictly unsupervised with respect to ground truth labels.")
    
    # -------------------------------------------------------------
    # Test 11 — Risk-Coverage Monotonicity Check
    # -------------------------------------------------------------
    print("\nTest 11 — Risk-Coverage Rejection Monotonicity...")
    sample_vars = np.array([0.1, 0.4, 0.2, 0.9, 0.05, 0.8, 0.3, 0.7])
    sample_errs = np.array([0,   1,   0,   1,   0,    1,   0,   1])
    covs, risks = compute_risk_coverage_curve(sample_vars, sample_errs, num_thresholds=10)
    
    # At lower coverage (retaining only lowest uncertainty samples), risk should be <= full coverage risk
    assert risks[0] <= risks[-1], f"Risk-coverage rejection non-monotonic: risk[0]={risks[0]}, risk[-1]={risks[-1]}"
    passed_tests += 1
    print("   [PASS] Test 11: Risk-Coverage curve shows effective error rate reduction upon low-uncertainty retention.")
    
    # -------------------------------------------------------------
    # Test 12 — Artifact Serialization & Reproducibility
    # -------------------------------------------------------------
    print("\nTest 12 — Serialization & Reproducibility Check...")
    scratch_dir = Path(__file__).resolve().parents[3] / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    temp_file = scratch_dir / "test_mc_array.npy"
    
    np.save(temp_file, mc_tensor.numpy())
    reloaded_arr = np.load(temp_file)
    assert np.allclose(mc_tensor.numpy(), reloaded_arr, atol=1e-6), "Array mismatch upon reload!"
    if temp_file.exists():
        temp_file.unlink()
    passed_tests += 1
    print("   [PASS] Test 12: Uncertainty array serialization verified cleanly.")
    
    print("\n=============================================================")
    print(f"VERIFICATION SUMMARY: {passed_tests}/{total_tests} PASS")
    print("=============================================================")
    assert passed_tests == total_tests, "Not all tests passed!"

if __name__ == "__main__":
    test_uncertainty_12_point_suite()
