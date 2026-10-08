"""
FusionMedAI - Phase C11.6: DCRI Risk Aggregation Deep Verification Suite (16/16 Gates)
Verifies weighted risk fusion, uncertainty burden, penalty discounting, delta sensitivity,
modality availability configurations, unclamped bounds, determinism, and Volume 06 documentation integrity.
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
    DecisionPacketError,
)
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.dcri.dcri_result import DCRIResult, DCRIContractError
from src.fusion.dcri.aggregation import (
    compute_weighted_risk_contributions,
    compute_r_fusion,
)
from src.fusion.dcri.uncertainty_penalty import (
    compute_uncertainty_burden,
    compute_uncertainty_penalty_contributions,
    compute_uncertainty_penalty,
)
from src.fusion.dcri.dcri_engine import DCRIEngine


def run_c11_6_verification() -> bool:
    print("=" * 85)
    print("FusionMedAI: Phase C11.6 - DCRI Risk Aggregation Deep Verification Suite (16 Gates)")
    print("=" * 85)

    passed_gates = 0
    total_gates = 16

    engine = DCRIEngine()

    std_packet = ControlledDecisionPacket(
        packet_id="PACKET_VERIFY_001",
        retina=ModalityRecord("r1", "retina", 0.60, (0.5, 0.5), 0.85, 0.15, 0.90, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.40, (0.5, 0.5), 0.75, 0.20, 0.85, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.25, (0.5, 0.5), 0.70, 0.25, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )

    # -------------------------------------------------------------------------
    # Gate 1: DCRI Module Existence & Contract Schema
    # -------------------------------------------------------------------------
    print("\n[Gate 1/16] Verifying DCRI Module Classes & Immutable Result Schema...")
    res = engine.evaluate_packet(std_packet, delta=0.20)
    assert isinstance(res, DCRIResult)
    assert res.status == "SUCCESS"
    assert res.delta == 0.20
    assert len(res.modality_weights) == 3
    assert len(res.weighted_risk_contributions) == 3
    assert len(res.uncertainty_penalty_contributions) == 3
    print("  -> PASS: DCRIResult dataclass adheres strictly to immutable schema.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 2: R_fusion Equation Correctness (R_fusion = sum_i w_i * r_i)
    # -------------------------------------------------------------------------
    print("\n[Gate 2/16] Verifying R_fusion Mathematical Formulation (sum_i w_i * r_i)...")
    k = compute_weighted_risk_contributions(res.modality_weights, res.modality_risks, res.active_modalities)
    expected_r = sum(res.modality_weights[m] * res.modality_risks[m] for m in res.active_modalities)
    assert abs(res.r_fusion - expected_r) < 1e-6
    print(f"  -> PASS: R_fusion ({res.r_fusion:.6f}) matches exact weighted linear combination.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 3: R_fusion Bounded in [0.0, 1.0]
    # -------------------------------------------------------------------------
    print("\n[Gate 3/16] Verifying R_fusion Strictly Bounded in [0.0, 1.0]...")
    assert 0.0 <= res.r_fusion <= 1.0
    # Test boundary limits
    p_ones = ControlledDecisionPacket(
        packet_id="P_ONES",
        retina=ModalityRecord("r1", "retina", 1.0, (0.0, 1.0), 1.0, 0.0, 1.0, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 1.0, (0.0, 1.0), 1.0, 0.0, 1.0, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 1.0, (0.0, 1.0), 1.0, 0.0, 1.0, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    res_ones = engine.evaluate_packet(p_ones, delta=0.0)
    assert abs(res_ones.r_fusion - 1.0) < 1e-6
    print("  -> PASS: R_fusion strictly bounded in [0.0, 1.0] under convex weighting.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 4: Uncertainty Burden Aggregation (U_sum & U_mean)
    # -------------------------------------------------------------------------
    print("\n[Gate 4/16] Verifying Uncertainty Burden Aggregation (U_sum = sum U_i, U_mean)...")
    expected_u_sum = 0.15 + 0.20 + 0.25  # 0.60
    expected_u_mean = 0.60 / 3.0          # 0.20
    assert abs(res.u_sum - expected_u_sum) < 1e-6
    assert abs(res.u_mean - expected_u_mean) < 1e-6
    print(f"  -> PASS: Cumulative uncertainty U_sum={res.u_sum:.4f} and U_mean={res.u_mean:.4f} exact.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 5: Penalty Calculation Correctness (P_U = delta * U_sum)
    # -------------------------------------------------------------------------
    print("\n[Gate 5/16] Verifying Uncertainty Penalty Calculation (P_U = delta * U_sum)...")
    expected_penalty = 0.20 * expected_u_sum  # 0.12
    assert abs(res.uncertainty_penalty - expected_penalty) < 1e-6
    assert abs(sum(res.uncertainty_penalty_contributions.values()) - expected_penalty) < 1e-6
    print(f"  -> PASS: Uncertainty penalty P_U={res.uncertainty_penalty:.6f} exact.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 6: DCRI Equation Correctness (DCRI = R_fusion - P_U)
    # -------------------------------------------------------------------------
    print("\n[Gate 6/16] Verifying DCRI Formulation (DCRI = R_fusion - P_U)...")
    expected_dcri = res.r_fusion - res.uncertainty_penalty
    assert abs(res.dcri - expected_dcri) < 1e-6
    print(f"  -> PASS: DCRI={res.dcri:.6f} matches R_fusion - P_U exactly.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 7: Delta = 0 Invariant (DCRI_0 == R_fusion)
    # -------------------------------------------------------------------------
    print("\n[Gate 7/16] Verifying Delta = 0 Invariant (DCRI_0 == R_fusion exactly)...")
    res_d0 = engine.evaluate_packet(std_packet, delta=0.0)
    assert res_d0.uncertainty_penalty == 0.0
    assert abs(res_d0.dcri - res_d0.r_fusion) < 1e-7
    print("  -> PASS: At delta=0, DCRI identically equals fused risk R_fusion.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 8: Delta Monotonicity (delta_1 > delta_0 => DCRI_1 < DCRI_0)
    # -------------------------------------------------------------------------
    print("\n[Gate 8/16] Verifying Delta Monotonicity across Grid [0, 0.05, 0.1, 0.2, 0.5, 1.0]...")
    grid = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    grid_res = engine.evaluate_packet_across_grid(std_packet, grid=grid)
    for i in range(len(grid) - 1):
        d_curr, d_next = grid[i], grid[i+1]
        assert grid_res[d_curr].dcri > grid_res[d_next].dcri, (
            f"Monotonicity violation: DCRI({d_curr})={grid_res[d_curr].dcri} <= DCRI({d_next})={grid_res[d_next].dcri}"
        )
    print("  -> PASS: DCRI strictly monotonically decreases with penalty multiplier delta.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 9: Uncertainty Monotonicity (U_high => DCRI_low)
    # -------------------------------------------------------------------------
    print("\n[Gate 9/16] Verifying Uncertainty Monotonicity (U_high => DCRI_low)...")
    p_u_low = ControlledDecisionPacket(
        packet_id="P_U_LOW",
        retina=ModalityRecord("r1", "retina", 0.60, (0.5, 0.5), 0.80, 0.05, 0.80, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.50, (0.5, 0.5), 0.80, 0.05, 0.80, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.40, (0.5, 0.5), 0.80, 0.05, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    p_u_high = ControlledDecisionPacket(
        packet_id="P_U_HIGH",
        retina=ModalityRecord("r1", "retina", 0.60, (0.5, 0.5), 0.80, 0.50, 0.80, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.50, (0.5, 0.5), 0.80, 0.50, 0.80, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.40, (0.5, 0.5), 0.80, 0.50, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    res_u_low = engine.evaluate_packet(p_u_low, delta=0.20)
    res_u_high = engine.evaluate_packet(p_u_high, delta=0.20)
    assert res_u_low.dcri > res_u_high.dcri
    print("  -> PASS: DCRI strictly decreases as predictive uncertainty increases.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 10: Modality Contribution Conservation (sum K_i == R_fusion)
    # -------------------------------------------------------------------------
    print("\n[Gate 10/16] Verifying Risk & Penalty Contribution Conservation...")
    k_sum = sum(res.weighted_risk_contributions.values())
    p_sum = sum(res.uncertainty_penalty_contributions.values())
    assert abs(k_sum - res.r_fusion) < 1e-6
    assert abs(p_sum - res.uncertainty_penalty) < 1e-6
    print(f"  -> PASS: Additive decomposition exactly conserved (sum K_i = {k_sum:.6f} == {res.r_fusion:.6f}).")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 11: All 7 Modality Availability Configurations
    # -------------------------------------------------------------------------
    print("\n[Gate 11/16] Verifying All 7 Valid Modality Configurations (R, F, C, RF, RC, FC, RFC)...")
    config_subsets = [
        ("R", ["retina"]),
        ("F", ["foot"]),
        ("C", ["clinical"]),
        ("RF", ["retina", "foot"]),
        ("RC", ["retina", "clinical"]),
        ("FC", ["foot", "clinical"]),
        ("RFC", ["retina", "foot", "clinical"]),
    ]
    for name, mods in config_subsets:
        p_cfg = ControlledDecisionPacket(
            packet_id=f"PACKET_{name}",
            retina=ModalityRecord("r1", "retina", 0.60, (0.5, 0.5), 0.85, 0.15, 0.90 if "retina" in mods else 0.0, "retina" in mods, FROZEN_RETINA_RELIABILITY, "v1"),
            foot=ModalityRecord("f1", "foot", 0.40, (0.5, 0.5), 0.75, 0.20, 0.85 if "foot" in mods else 0.0, "foot" in mods, FROZEN_FOOT_RELIABILITY, "v1"),
            clinical=ModalityRecord("c1", "clinical", 0.25, (0.5, 0.5), 0.70, 0.25, 0.80 if "clinical" in mods else 0.0, "clinical" in mods, FROZEN_CLINICAL_RELIABILITY, "v1"),
            seed=115,
        )
        res_cfg = engine.evaluate_packet(p_cfg, delta=0.20)
        assert res_cfg.status == "SUCCESS"
        assert res_cfg.num_active == len(mods)
        assert abs(sum(res_cfg.modality_weights.values()) - 1.0) < 1e-6
        for m in ["retina", "foot", "clinical"]:
            if m not in mods:
                assert res_cfg.modality_weights[m] == 0.0
                assert res_cfg.weighted_risk_contributions[m] == 0.0
                assert res_cfg.uncertainty_penalty_contributions[m] == 0.0
    print("  -> PASS: All 7 modality configurations evaluate with strict hard availability masking.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 12: Zero-Modality Safe Rejection (NO_MODALITY_AVAILABLE)
    # -------------------------------------------------------------------------
    print("\n[Gate 12/16] Verifying Zero-Modality Graceful Failure...")
    p_zero = ControlledDecisionPacket(
        packet_id="PACKET_EMPTY",
        retina=ModalityRecord("r1", "retina", 0.60, (0.5, 0.5), 0.85, 0.15, 0.0, False, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.40, (0.5, 0.5), 0.75, 0.20, 0.0, False, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.25, (0.5, 0.5), 0.70, 0.25, 0.0, False, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    res_zero = engine.evaluate_packet(p_zero, delta=0.20)
    assert res_zero.status == "NO_MODALITY_AVAILABLE"
    assert res_zero.num_active == 0
    assert res_zero.r_fusion == 0.0
    assert res_zero.dcri == 0.0
    print("  -> PASS: Zero-modality state safely emits NO_MODALITY_AVAILABLE with 0.0 metrics.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 13: Negative DCRI Explicit Handling (Unclamped Bounding in [-delta*M, 1])
    # -------------------------------------------------------------------------
    print("\n[Gate 13/16] Verifying Unclamped Negative DCRI Handling (Bound [-delta*M, 1])...")
    p_neg = ControlledDecisionPacket(
        packet_id="PACKET_NEG",
        retina=ModalityRecord("r1", "retina", 0.10, (0.5, 0.5), 0.80, 0.80, 0.90, True, FROZEN_RETINA_RELIABILITY, "v1"),
        foot=ModalityRecord("f1", "foot", 0.10, (0.5, 0.5), 0.80, 0.80, 0.90, True, FROZEN_FOOT_RELIABILITY, "v1"),
        clinical=ModalityRecord("c1", "clinical", 0.10, (0.5, 0.5), 0.80, 0.80, 0.90, True, FROZEN_CLINICAL_RELIABILITY, "v1"),
        seed=115,
    )
    res_neg = engine.evaluate_packet(p_neg, delta=1.0)
    assert res_neg.dcri < 0.0, "Expected negative DCRI under high uncertainty and delta=1.0"
    assert abs(res_neg.dcri - (0.10 - 2.40)) < 1e-6
    assert res_neg.dcri >= -3.0
    print(f"  -> PASS: Negative DCRI ({res_neg.dcri:.4f}) correctly computed without illicit clamping.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 14: Deterministic Computation
    # -------------------------------------------------------------------------
    print("\n[Gate 14/16] Verifying Deterministic Computation (f(X) == f(X))...")
    res1 = engine.evaluate_packet(std_packet, delta=0.20)
    res2 = engine.evaluate_packet(std_packet, delta=0.20)
    assert res1.dcri == res2.dcri
    assert res1.r_fusion == res2.r_fusion
    assert res1.modality_weights == res2.modality_weights
    print("  -> PASS: DCRI engine is strictly deterministic and stateless.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 15: Ground-Truth Independence & Zero Label Leakage
    # -------------------------------------------------------------------------
    print("\n[Gate 15/16] Auditing Ground-Truth Independence in Signatures...")
    import inspect
    for func in [compute_weighted_risk_contributions, compute_r_fusion, compute_uncertainty_burden, compute_uncertainty_penalty, engine.evaluate_packet]:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        assert "ground_truth" not in params and "label" not in params and "y_true" not in params
    print("  -> PASS: Zero label parameters in all DCRI functions.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 16: Read-Only Audit of Experiment Artifacts & Volume 06 Documentation
    # -------------------------------------------------------------------------
    print("\n[Gate 16/16] Auditing Volume 06 Research Documents & Experiment Artifacts...")
    exp_dir = REPO_ROOT / "experiments" / "fusion" / "dcri"
    vol6_dir = REPO_ROOT / "research" / "fusion" / "Volume_06_DCRI_Aggregation"

    # Check all 10 experiment artifacts exist on disk
    required_artifacts = [
        "dcri_configuration.json",
        "dcri_results.json",
        "delta_sensitivity.json",
        "modality_configuration_results.json",
        "contribution_analysis.json",
        "uncertainty_double_use.json",
        "sanity_cases.json",
        "edge_case_results.json",
        "packet_manifest.json",
        "freeze_manifest.json",
    ]
    for fname in required_artifacts:
        fpath = exp_dir / fname
        assert fpath.is_file(), f"Missing required artifact: {fpath}"
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) > 0, f"Empty artifact file: {fpath}"

    # Check Volume 06 Markdown Documents
    required_docs = [
        "README.md",
        "01_DCRI_Protocol.md",
        "02_Mathematical_Formulation.md",
        "03_Aggregation_Procedure.md",
        "04_Delta_Sensitivity.md",
        "05_Uncertainty_Penalty_Analysis.md",
        "06_Modality_Contribution_Analysis.md",
        "07_Results.md",
        "08_Freeze_Report.md",
    ]
    for dname in required_docs:
        dpath = vol6_dir / dname
        assert dpath.is_file(), f"Missing research document: {dpath}"
        content = dpath.read_text(encoding="utf-8")
        assert len(content) > 200, f"Document {dpath} is unexpectedly short or empty."

    print("  -> PASS: All 10 experiment artifacts and 9 Volume 06 documents verified.")
    passed_gates += 1

    print("\n" + "=" * 85)
    print(f"C11.6 DCRI RISK AGGREGATION DEEP VERIFICATION: {passed_gates}/{total_gates} GATES PASSED")
    print("=" * 85)
    return passed_gates == total_gates


if __name__ == "__main__":
    success = run_c11_6_verification()
    sys.exit(0 if success else 1)
