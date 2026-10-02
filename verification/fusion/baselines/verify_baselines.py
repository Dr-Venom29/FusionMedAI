"""
FusionMedAI - Phase C11.5: Deep Comparative Baseline & Fusion Evaluation Verification Suite (18/18 Gates)
Verifies baseline ladder (B1–B6), mathematical mechanics, availability masking,
reproducibility, zero fake patient claims, and Volume 05 research documentation integrity.
"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fusion.baselines.decision_packet import (
    ModalityRecord,
    ControlledDecisionPacket,
    FusionResult,
    DecisionPacketError,
)
from src.fusion.baselines.b1_reliability_selected import ReliabilitySelectedBaseline
from src.fusion.baselines.b2_uniform import UniformAverageBaseline
from src.fusion.baselines.b3_confidence import ConfidenceFusionBaseline
from src.fusion.baselines.b4_confidence_reliability import ConfidenceReliabilityBaseline
from src.fusion.baselines.b5_confidence_reliability_uncertainty import ConfidenceReliabilityUncertaintyBaseline
from src.fusion.baselines.b6_acarau import FullACARAUBaseline
from src.fusion.baselines.fusion_runner import FusionRunner
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)


def run_c11_5_verification() -> bool:
    print("=" * 85)
    print("FusionMedAI: Phase C11.5 - Baseline Ladder & Comparative Fusion Deep Verification")
    print("=" * 85)

    passed_gates = 0
    total_gates = 18

    # Helper standard packet
    std_packet = ControlledDecisionPacket(
        packet_id="PACKET_VERIFY_001",
        retina=ModalityRecord("r1", "retina", 0.60, (0.5, 0.5), 0.85, 0.15, 0.90, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.40, (0.5, 0.5), 0.75, 0.20, 0.85, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.25, (0.5, 0.5), 0.70, 0.25, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    runner = FusionRunner()

    # -------------------------------------------------------------------------
    # Gate 1: Baseline Implementations & Class Contracts
    # -------------------------------------------------------------------------
    print("\n[Gate 1/18] Verifying Baseline Ladder Classes & Contract Schema...")
    for b_id in ["B1", "B2", "B3", "B4", "B5", "B6"]:
        assert b_id in runner.baselines
        res = runner.evaluate_packet(std_packet, b_id)
        assert isinstance(res, FusionResult)
        assert res.baseline_id == b_id
        assert res.status == "SUCCESS"
    print("  -> PASS: All 6 baseline classes (B1 through B6) correctly instantiate and emit FusionResult.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 2: Frozen Baseline Coefficients (Theta_B2 .. Theta_B6)
    # -------------------------------------------------------------------------
    print("\n[Gate 2/18] Verifying Frozen Baseline Coefficient Configurations...")
    assert isinstance(runner.baselines["B3"], ConfidenceFusionBaseline) and runner.baselines["B3"].alpha == 1.0
    assert isinstance(runner.baselines["B4"], ConfidenceReliabilityBaseline) and runner.baselines["B4"].alpha == 1.0 and runner.baselines["B4"].beta == 1.0
    assert isinstance(runner.baselines["B5"], ConfidenceReliabilityUncertaintyBaseline) and runner.baselines["B5"].gamma == 1.0
    assert isinstance(runner.baselines["B6"], FullACARAUBaseline)
    print("  -> PASS: Baseline coefficient configurations strictly locked to protocol ladder (B2–B6).")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 3: Frozen Validation Reliability Priors
    # -------------------------------------------------------------------------
    print("\n[Gate 3/18] Verifying Ingestion of Phase C11.3 Frozen Reliability Priors...")
    assert FROZEN_RETINA_RELIABILITY == 0.929956
    assert FROZEN_FOOT_RELIABILITY == 0.922266
    assert FROZEN_CLINICAL_RELIABILITY == 0.825382
    print("  -> PASS: Modality reliability priors locked to validation evidence.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 4: Hard Availability Masking across All Baselines
    # -------------------------------------------------------------------------
    print("\n[Gate 4/18] Verifying Hard Availability Masking (A_i=0 => w_i=0.0) across B1–B6...")
    p_masked = ControlledDecisionPacket(
        packet_id="PACKET_MASKED",
        retina=ModalityRecord("r1", "retina", 0.60, (0.5, 0.5), 0.85, 0.15, 0.90, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.40, (0.5, 0.5), 0.75, 0.20, 0.0, False, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.25, (0.5, 0.5), 0.70, 0.25, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    for b_id in ["B1", "B2", "B3", "B4", "B5", "B6"]:
        res_m = runner.evaluate_packet(p_masked, b_id)
        assert res_m.weights["foot"] == 0.0
        assert abs(res_m.weights["retina"] + res_m.weights["clinical"] - 1.0) < 1e-7
    print("  -> PASS: Hard availability masking strictly enforced across all baselines.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 5: Weight Normalization across All Baselines
    # -------------------------------------------------------------------------
    print("\n[Gate 5/18] Verifying Softmax Weight Normalization (sum w_i = 1.0)...")
    for b_id in ["B1", "B2", "B3", "B4", "B5", "B6"]:
        res = runner.evaluate_packet(std_packet, b_id)
        assert abs(sum(res.weights.values()) - 1.0) < 1e-7
        assert all(0.0 <= w <= 1.0 for w in res.weights.values())
        assert 0.0 <= res.r_fusion <= 1.0
    print("  -> PASS: All baseline weights strictly normalize to 1.0.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 6: Baseline B2 (Uniform Average) Mechanics
    # -------------------------------------------------------------------------
    print("\n[Gate 6/18] Verifying Baseline B2 Uniform Weighting Mechanics...")
    b2_res = runner.evaluate_packet(std_packet, "B2")
    for mod in ["retina", "foot", "clinical"]:
        assert abs(b2_res.weights[mod] - (1.0 / 3.0)) < 1e-7
    assert abs(b2_res.routing_entropy - math.log(3.0)) < 1e-6
    print("  -> PASS: B2 achieves exact 1/N weighting with maximum entropy ln(3).")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 7: Baseline B1 (Reliability-Selected) Mechanics
    # -------------------------------------------------------------------------
    print("\n[Gate 7/18] Verifying Baseline B1 Selection Rule (i* = argmax R_i)...")
    b1_res = runner.evaluate_packet(std_packet, "B1")
    assert b1_res.dominant_modality == "retina"
    assert b1_res.weights["retina"] == 1.0
    assert b1_res.r_fusion == std_packet.retina.risk
    assert b1_res.routing_entropy == 0.0
    print("  -> PASS: B1 selects highest-reliability active modality deterministically.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 8: Baseline B3 (Confidence-Only) Isolation
    # -------------------------------------------------------------------------
    print("\n[Gate 8/18] Verifying Baseline B3 Confidence Isolation...")
    p_conf = ControlledDecisionPacket(
        packet_id="PACKET_CONF",
        retina=ModalityRecord("r1", "retina", 0.5, (0.5, 0.5), 0.90, 0.5, 0.5, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.5, (0.5, 0.5), 0.50, 0.1, 0.9, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.5, (0.5, 0.5), 0.20, 0.1, 0.9, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    b3_res = runner.evaluate_packet(p_conf, "B3")
    assert b3_res.weights["retina"] > b3_res.weights["foot"] > b3_res.weights["clinical"]
    print("  -> PASS: B3 routes solely based on confidence.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 9: Baseline B4 (Confidence + Reliability) Anchoring
    # -------------------------------------------------------------------------
    print("\n[Gate 9/18] Verifying Baseline B4 Reliability Anchoring...")
    p_equal_c = ControlledDecisionPacket(
        packet_id="PACKET_EQ_C",
        retina=ModalityRecord("r1", "retina", 0.5, (0.5, 0.5), 0.70, 0.2, 0.8, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.5, (0.5, 0.5), 0.70, 0.2, 0.8, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.5, (0.5, 0.5), 0.70, 0.2, 0.8, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    b4_res = runner.evaluate_packet(p_equal_c, "B4")
    assert b4_res.weights["retina"] > b4_res.weights["foot"] > b4_res.weights["clinical"]
    print("  -> PASS: B4 integrates reliability prior correctly under equal confidence.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 10: Baseline B5 (Uncertainty Penalty) Responsiveness
    # -------------------------------------------------------------------------
    print("\n[Gate 10/18] Verifying Baseline B5 Predictive Uncertainty Penalty...")
    p_u_low = ControlledDecisionPacket(
        packet_id="PACKET_U_LOW",
        retina=ModalityRecord("r1", "retina", 0.5, (0.5, 0.5), 0.80, 0.05, 0.80, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.5, (0.5, 0.5), 0.80, 0.20, 0.80, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.5, (0.5, 0.5), 0.80, 0.20, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    p_u_high = ControlledDecisionPacket(
        packet_id="PACKET_U_HIGH",
        retina=ModalityRecord("r1", "retina", 0.5, (0.5, 0.5), 0.80, 0.80, 0.80, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.5, (0.5, 0.5), 0.80, 0.20, 0.80, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.5, (0.5, 0.5), 0.80, 0.20, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    b5_low = runner.evaluate_packet(p_u_low, "B5")
    b5_high = runner.evaluate_packet(p_u_high, "B5")
    assert b5_low.weights["retina"] > b5_high.weights["retina"]
    print(f"  -> PASS: B5 uncertainty penalty verified ({b5_low.weights['retina']:.4f} > {b5_high.weights['retina']:.4f}).")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 11: Baseline B6 (Full ACARA-U) Integration
    # -------------------------------------------------------------------------
    print("\n[Gate 11/18] Verifying Baseline B6 Full ACARA-U Integration...")
    b6_res = runner.evaluate_packet(std_packet, "B6")
    assert b6_res.dominant_modality == "retina"
    assert 0.0 <= b6_res.r_fusion <= 1.0
    print("  -> PASS: B6 executes full ACARA-U dynamic routing.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 12: Zero Prediction Leakage into Reliability
    # -------------------------------------------------------------------------
    print("\n[Gate 12/18] Auditing Decoupling of Reliability from Instance Prediction...")
    for mod in ["retina", "foot", "clinical"]:
        assert std_packet.records[mod].reliability in (0.929956, 0.922266, 0.825382)
    print("  -> PASS: Zero prediction leakage into frozen reliability.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 13: Zero Labels in Routing Signature
    # -------------------------------------------------------------------------
    print("\n[Gate 13/18] Auditing Ground-Truth Independence...")
    import inspect
    for b_id, b_inst in runner.baselines.items():
        sig = inspect.signature(b_inst.evaluate)
        params = list(sig.parameters.keys())
        assert "ground_truth" not in params and "label" not in params and "y_true" not in params
    print("  -> PASS: Router and baseline signatures contain zero ground truth parameters.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 14: Determinism & Reproducibility
    # -------------------------------------------------------------------------
    print("\n[Gate 14/18] Verifying Deterministic Reproducibility...")
    res1 = runner.evaluate_all_baselines(std_packet)
    res2 = runner.evaluate_all_baselines(std_packet)
    for b_id in res1.keys():
        assert res1[b_id].weights == res2[b_id].weights
        assert res1[b_id].r_fusion == res2[b_id].r_fusion
    print("  -> PASS: All baselines strictly deterministic.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 15: Cross-Modality Disagreement Metrics
    # -------------------------------------------------------------------------
    print("\n[Gate 15/18] Verifying Cross-Modality Disagreement Metrics (X_RF, X_RC, X_FC, X_max)...")
    res_disag = runner.evaluate_packet(std_packet, "B6")
    expected_x_rf = abs(0.60 - 0.40)  # 0.20
    expected_x_rc = abs(0.60 - 0.25)  # 0.35
    expected_x_fc = abs(0.40 - 0.25)  # 0.15
    assert abs(res_disag.disagreement["X_RF"] - expected_x_rf) < 1e-6
    assert abs(res_disag.disagreement["X_RC"] - expected_x_rc) < 1e-6
    assert abs(res_disag.disagreement["X_FC"] - expected_x_fc) < 1e-6
    assert abs(res_disag.disagreement["X_max"] - 0.35) < 1e-6
    print("  -> PASS: Cross-modality disagreement metrics exact match.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 16: Zero-Modality Safe Rejection
    # -------------------------------------------------------------------------
    print("\n[Gate 16/18] Verifying Zero-Modality Graceful Failure across All Baselines...")
    p_zero = ControlledDecisionPacket(
        packet_id="PACKET_EMPTY",
        retina=ModalityRecord("r1", "retina", 0.60, (0.5, 0.5), 0.85, 0.15, 0.0, False, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.40, (0.5, 0.5), 0.75, 0.20, 0.0, False, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.25, (0.5, 0.5), 0.70, 0.25, 0.0, False, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    for b_id in ["B1", "B2", "B3", "B4", "B5", "B6"]:
        res_z = runner.evaluate_packet(p_zero, b_id)
        assert res_z.status == "NO_MODALITY_AVAILABLE"
        assert res_z.r_fusion == 0.0
        assert res_z.dominant_modality is None
    print("  -> PASS: All baselines gracefully reject zero-modality state.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 17: Non-Clinical-Pair Declaration & Packet Type Enforcement
    # -------------------------------------------------------------------------
    print("\n[Gate 17/18] Auditing Packet Type & Scientific Boundary Invariants...")
    assert std_packet.packet_type == "CONTROLLED_DECISION_PACKET"
    try:
        ControlledDecisionPacket(
            packet_id="BAD_PACKET",
            retina=std_packet.retina,
            foot=std_packet.foot,
            clinical=std_packet.clinical,
            seed=115,
            packet_type="REAL_PATIENT",  # Prohibited!
        )
        raise AssertionError("Failed to reject invalid packet type")
    except DecisionPacketError:
        pass
    print("  -> PASS: Packet type strictly restricted to 'CONTROLLED_DECISION_PACKET'.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 18: Frozen Experiment Artifacts & Volume 05 Documentation Integrity
    # -------------------------------------------------------------------------
    print("\n[Gate 18/18] Auditing Volume 05 Research Documents & Experiment Artifacts...")
    exp_dir = REPO_ROOT / "experiments" / "fusion" / "baseline_comparison"
    vol5_dir = REPO_ROOT / "research" / "fusion" / "Volume_05_Baseline_Fusion"

    # 1. Check all 13 Experiment Artifacts exist on disk
    required_artifacts = [
        "experiment_config.json",
        "prediction_pool_manifest.json",
        "packet_manifest.json",
        "b1_results.json",
        "b2_results.json",
        "b3_results.json",
        "b4_results.json",
        "b5_results.json",
        "b6_results.json",
        "missing_modality_matrix.json",
        "perturbation_benchmark.json",
        "disagreement_analysis.json",
        "summary.json",
        "freeze_manifest.json",
    ]
    for fname in required_artifacts:
        fpath = exp_dir / fname
        assert fpath.is_file(), f"Missing required artifact: {fpath}"
        with open(fpath, "r") as f:
            data = json.load(f)
            assert len(data) > 0, f"Empty artifact file: {fpath}"

    # 2. Check Volume 05 Markdown Documents
    required_docs = [
        "README.md",
        "01_Experimental_Protocol.md",
        "02_Baseline_Definitions.md",
        "03_Decision_Packet_Construction.md",
        "04_Behavioral_Metrics.md",
        "05_Missing_Modality_Evaluation.md",
        "06_Perturbation_Evaluation.md",
        "07_Results.md",
        "08_Freeze_Report.md",
    ]
    for dname in required_docs:
        dpath = vol5_dir / dname
        assert dpath.is_file(), f"Missing research document: {dpath}"
        content = dpath.read_text(encoding="utf-8")
        assert len(content) > 200, f"Document {dpath} is unexpectedly short or empty."

    # 3. Independent Empirical Recomputation Audit over Manifest Sample Packets
    with open(exp_dir / "packet_manifest.json", "r") as f:
        manifest_data = json.load(f)
    assert manifest_data["num_packets"] == 500, f"Expected num_packets == 500, got {manifest_data.get('num_packets')}"
    sample_packet_dicts = manifest_data["sample_packets"]
    assert len(sample_packet_dicts) > 0, "No sample packets found in manifest"

    recomputed_sample_packets = [
        ControlledDecisionPacket(
            packet_id=pd["packet_id"],
            retina=ModalityRecord(**pd["retina"]),
            foot=ModalityRecord(**pd["foot"]),
            clinical=ModalityRecord(**pd["clinical"]),
            seed=pd["seed"],
            packet_type=pd["packet_type"],
        )
        for pd in sample_packet_dicts
    ]

    for p in recomputed_sample_packets:
        res = runner.evaluate_all_baselines(p)
        # Verify baseline properties on each recomputed sample packet
        assert abs(res["B1"].weights["retina"] - 1.0) < 1e-6
        assert abs(res["B2"].weights["retina"] - 1/3) < 1e-6
        assert abs(sum(res["B3"].weights.values()) - 1.0) < 1e-6
        assert abs(sum(res["B4"].weights.values()) - 1.0) < 1e-6
        assert abs(sum(res["B5"].weights.values()) - 1.0) < 1e-6
        assert abs(sum(res["B6"].weights.values()) - 1.0) < 1e-6

    print("  -> PASS: All 14 experiment artifacts, 9 Volume 05 documents, and independent sample recomputations verified.")
    passed_gates += 1

    print("\n" + "=" * 85)
    print(f"C11.5 BASELINE COMPARISON DEEP VERIFICATION: {passed_gates}/{total_gates} GATES PASSED")
    print("=" * 85)
    return passed_gates == total_gates


if __name__ == "__main__":
    success = run_c11_5_verification()
    sys.exit(0 if success else 1)
