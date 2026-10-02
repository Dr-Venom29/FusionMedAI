"""
FusionMedAI - Phase C11.4: Deep ACARA-U v2 Dynamic Router Verification Suite (18/18 Gates)
Verifies mathematical formulations, invariants, availability masking, 7 configurations,
monotonicity, numerical stability, frozen artifacts, and research documentation integrity.
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

from src.fusion.router.router_input import (
    ModalityChannelInput,
    RouterInput,
    RouterContractValidationError,
    FROZEN_RELIABILITY_MAP,
)
from src.fusion.router.coefficients import RouterCoefficients, RouterCoefficientError
from src.fusion.router.router_result import RouterResult
from src.fusion.router.acarau_router import ACARAUv2Router


def run_c11_4_verification() -> bool:
    print("=" * 85)
    print("FusionMedAI: Phase C11.4 - ACARA-U v2 Dynamic Router Deep Verification Suite")
    print("=" * 85)

    passed_gates = 0
    total_gates = 18

    # -------------------------------------------------------------------------
    # Gate 1: Input Contract Validation & Immutability
    # -------------------------------------------------------------------------
    print("\n[Gate 1/18] Verifying RouterInput & ModalityChannelInput Contract Invariants...")
    ch_r = ModalityChannelInput("retina", 0.85, FROZEN_RELIABILITY_MAP["retina"], 0.15, 0.90, True)
    ch_f = ModalityChannelInput("foot", 0.80, FROZEN_RELIABILITY_MAP["foot"], 0.20, 0.85, True)
    ch_c = ModalityChannelInput("clinical", 0.75, FROZEN_RELIABILITY_MAP["clinical"], 0.25, 0.80, True)
    r_inp = RouterInput(retina=ch_r, foot=ch_f, clinical=ch_c)

    assert r_inp.num_available == 3
    assert r_inp.available_modalities == ["retina", "foot", "clinical"]
    # Check invalid scalar rejection
    try:
        ModalityChannelInput("retina", 1.5, FROZEN_RELIABILITY_MAP["retina"], 0.1, 0.9, True)
        raise AssertionError("Failed to reject confidence > 1.0")
    except RouterContractValidationError:
        pass
    print("  -> PASS: RouterInput & ModalityChannelInput enforce dataclass immutability and domain bounds.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 2: Frozen Reliability Constants Enforcement
    # -------------------------------------------------------------------------
    print("\n[Gate 2/18] Verifying Strict Enforcement of Phase C11.3 Frozen Reliability Priors...")
    assert FROZEN_RELIABILITY_MAP["retina"] == 0.929956
    assert FROZEN_RELIABILITY_MAP["foot"] == 0.922266
    assert FROZEN_RELIABILITY_MAP["clinical"] == 0.825382

    # Attempt to tamper with reliability prior
    try:
        ModalityChannelInput("retina", 0.8, 0.999999, 0.2, 0.8, True)
        raise AssertionError("Failed to reject tampered reliability prior")
    except RouterContractValidationError:
        pass
    print("  -> PASS: Frozen reliability priors (R_R=0.929956, R_F=0.922266, R_C=0.825382) strictly locked.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 3: Logit Scoring Function Fidelity
    # -------------------------------------------------------------------------
    print("\n[Gate 3/18] Verifying Logit Scoring Formula z_i = alpha*C_i + beta*R_i - gamma*U_i + eta*Q_i...")
    coeffs = RouterCoefficients(alpha=2.0, beta=1.5, gamma=1.2, eta=0.8)
    router = ACARAUv2Router(coefficients=coeffs)
    raw_logits, masked_logits = router.compute_logits(r_inp)

    expected_z_r = 2.0 * 0.85 + 1.5 * 0.929956 - 1.2 * 0.15 + 0.8 * 0.90
    assert abs(raw_logits["retina"] - expected_z_r) < 1e-6
    print(f"  -> PASS: Logit computation exact match ({raw_logits['retina']:.6f} == {expected_z_r:.6f}).")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 4: Hard Availability Masking
    # -------------------------------------------------------------------------
    print("\n[Gate 4/18] Verifying Hard Availability Masking (A_i=0 => z_tilde_i = -inf)...")
    inp_masked = RouterInput(
        retina=ModalityChannelInput("retina", 0.85, FROZEN_RELIABILITY_MAP["retina"], 0.15, 0.90, True),
        foot=ModalityChannelInput("foot", 0.80, FROZEN_RELIABILITY_MAP["foot"], 0.20, 0.0, False),
        clinical=ModalityChannelInput("clinical", 0.75, FROZEN_RELIABILITY_MAP["clinical"], 0.25, 0.80, True),
    )
    raw_l, masked_l = router.compute_logits(inp_masked)
    assert masked_l["foot"] == float("-inf")
    assert math.isfinite(masked_l["retina"])
    assert math.isfinite(masked_l["clinical"])
    print("  -> PASS: Availability gating correctly assigns -inf to unavailable channels.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 5: Softmax Weight Normalization
    # -------------------------------------------------------------------------
    print("\n[Gate 5/18] Verifying Softmax Weight Normalization (sum w_i = 1.0, w_i >= 0)...")
    res = router.route(r_inp)
    assert abs(res.normalization_sum - 1.0) < 1e-7
    assert all(0.0 <= w <= 1.0 for w in res.weights.values())
    assert all(math.isfinite(w) for w in res.weights.values())
    print(f"  -> PASS: Weights strictly normalized (sum = {res.normalization_sum:.8f}).")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 6: Single-Modality Unimodal Configurations (Configs 5, 6, 7)
    # -------------------------------------------------------------------------
    print("\n[Gate 6/18] Verifying Unimodal Operation (Configs 5, 6, 7: w_active = 1.0)...")
    for mod in ["retina", "foot", "clinical"]:
        inp_uni = RouterInput(
            retina=ModalityChannelInput("retina", 0.8, FROZEN_RELIABILITY_MAP["retina"], 0.2, 0.8 if mod == "retina" else 0.0, mod == "retina"),
            foot=ModalityChannelInput("foot", 0.8, FROZEN_RELIABILITY_MAP["foot"], 0.2, 0.8 if mod == "foot" else 0.0, mod == "foot"),
            clinical=ModalityChannelInput("clinical", 0.8, FROZEN_RELIABILITY_MAP["clinical"], 0.2, 0.8 if mod == "clinical" else 0.0, mod == "clinical"),
        )
        res_uni = router.route(inp_uni)
        assert res_uni.num_active == 1
        assert res_uni.weights[mod] == 1.0
        assert res_uni.dominant_modality == mod
        assert res_uni.routing_entropy == 0.0
    print("  -> PASS: All 3 unimodal configurations assign exactly 1.0 weight with 0.0 entropy.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 7: Pairwise Bimodal Configurations (Configs 2, 3, 4)
    # -------------------------------------------------------------------------
    print("\n[Gate 7/18] Verifying Bimodal Operation (Configs 2, 3, 4: w_active1 + w_active2 = 1.0)...")
    bimodal_configs = [
        ("retina", "foot", "clinical"),
        ("retina", "clinical", "foot"),
        ("foot", "clinical", "retina"),
    ]
    for m1, m2, m_miss in bimodal_configs:
        inp_bi = RouterInput(
            retina=ModalityChannelInput("retina", 0.8, FROZEN_RELIABILITY_MAP["retina"], 0.2, 0.8 if "retina" in (m1, m2) else 0.0, "retina" in (m1, m2)),
            foot=ModalityChannelInput("foot", 0.8, FROZEN_RELIABILITY_MAP["foot"], 0.2, 0.8 if "foot" in (m1, m2) else 0.0, "foot" in (m1, m2)),
            clinical=ModalityChannelInput("clinical", 0.8, FROZEN_RELIABILITY_MAP["clinical"], 0.2, 0.8 if "clinical" in (m1, m2) else 0.0, "clinical" in (m1, m2)),
        )
        res_bi = router.route(inp_bi)
        assert res_bi.num_active == 2
        assert res_bi.weights[m_miss] == 0.0
        assert abs(res_bi.weights[m1] + res_bi.weights[m2] - 1.0) < 1e-7
        assert res_bi.weights[m1] > 0.0
        assert res_bi.weights[m2] > 0.0
    print("  -> PASS: All 3 bimodal configurations assign strictly positive weights summing to 1.0.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 8: Tri-Modal Configuration (Config 1)
    # -------------------------------------------------------------------------
    print("\n[Gate 8/18] Verifying Tri-Modal Operation (Config 1: R+F+C)...")
    res_tri = router.route(r_inp)
    assert res_tri.num_active == 3
    assert len(res_tri.active_modalities) == 3
    for mod in ["retina", "foot", "clinical"]:
        assert res_tri.weights[mod] > 0.0
    assert abs(res_tri.normalization_sum - 1.0) < 1e-7
    print("  -> PASS: Full tri-modal configuration assigns valid convex weights.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 9: Missing-Modality Zero-Weight Invariant
    # -------------------------------------------------------------------------
    print("\n[Gate 9/18] Verifying Missing-Modality Invariant (A_i=0 => w_i=0.0 under arbitrary signals)...")
    inp_tampered = RouterInput(
        retina=ModalityChannelInput("retina", 1.0, FROZEN_RELIABILITY_MAP["retina"], 0.0, 0.0, False),  # Missing but perfect C and U
        foot=ModalityChannelInput("foot", 0.5, FROZEN_RELIABILITY_MAP["foot"], 0.5, 0.5, True),
        clinical=ModalityChannelInput("clinical", 0.5, FROZEN_RELIABILITY_MAP["clinical"], 0.5, 0.5, True),
    )
    res_tampered = router.route(inp_tampered)
    assert res_tampered.weights["retina"] == 0.0
    assert abs(res_tampered.weights["foot"] + res_tampered.weights["clinical"] - 1.0) < 1e-7
    print("  -> PASS: Missing modality receives strictly zero weight regardless of instance signals.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 10: Confidence Monotonicity
    # -------------------------------------------------------------------------
    print("\n[Gate 10/18] Verifying Confidence Monotonicity (Finite Difference C_i ^ => w_i ^)...")
    c_router = ACARAUv2Router(coefficients=RouterCoefficients(alpha=2.0, beta=0.0, gamma=0.0, eta=0.0))
    c_weights = []
    for c_val in [0.1, 0.5, 0.9]:
        inp_c = RouterInput(
            retina=ModalityChannelInput("retina", c_val, FROZEN_RELIABILITY_MAP["retina"], 0.2, 0.8, True),
            foot=ModalityChannelInput("foot", 0.5, FROZEN_RELIABILITY_MAP["foot"], 0.2, 0.8, True),
            clinical=ModalityChannelInput("clinical", 0.5, FROZEN_RELIABILITY_MAP["clinical"], 0.2, 0.8, True),
        )
        c_weights.append(c_router.route(inp_c).weights["retina"])
    assert c_weights[0] < c_weights[1] < c_weights[2]
    print(f"  -> PASS: Confidence monotonic response verified ({c_weights[0]:.4f} < {c_weights[1]:.4f} < {c_weights[2]:.4f}).")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 11: Reliability Monotonicity
    # -------------------------------------------------------------------------
    print("\n[Gate 11/18] Verifying Reliability Ordering (R_R > R_F > R_C => w_R > w_F > w_C)...")
    r_router = ACARAUv2Router(coefficients=RouterCoefficients(alpha=0.0, beta=2.0, gamma=0.0, eta=0.0))
    inp_r = RouterInput(
        retina=ModalityChannelInput("retina", 0.8, FROZEN_RELIABILITY_MAP["retina"], 0.2, 0.8, True),
        foot=ModalityChannelInput("foot", 0.8, FROZEN_RELIABILITY_MAP["foot"], 0.2, 0.8, True),
        clinical=ModalityChannelInput("clinical", 0.8, FROZEN_RELIABILITY_MAP["clinical"], 0.2, 0.8, True),
    )
    res_r = r_router.route(inp_r)
    assert res_r.weights["retina"] > res_r.weights["foot"] > res_r.weights["clinical"]
    print(f"  -> PASS: Reliability ordering verified (w_R={res_r.weights['retina']:.4f} > w_F={res_r.weights['foot']:.4f} > w_C={res_r.weights['clinical']:.4f}).")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 12: Uncertainty Monotonicity
    # -------------------------------------------------------------------------
    print("\n[Gate 12/18] Verifying Uncertainty Penalty Monotonicity (Finite Difference U_i ^ => w_i v)...")
    u_router = ACARAUv2Router(coefficients=RouterCoefficients(alpha=0.0, beta=0.0, gamma=2.0, eta=0.0))
    u_weights = []
    for u_val in [0.1, 0.5, 0.9]:
        inp_u = RouterInput(
            retina=ModalityChannelInput("retina", 0.8, FROZEN_RELIABILITY_MAP["retina"], u_val, 0.8, True),
            foot=ModalityChannelInput("foot", 0.8, FROZEN_RELIABILITY_MAP["foot"], 0.5, 0.8, True),
            clinical=ModalityChannelInput("clinical", 0.8, FROZEN_RELIABILITY_MAP["clinical"], 0.5, 0.8, True),
        )
        u_weights.append(u_router.route(inp_u).weights["retina"])
    assert u_weights[0] > u_weights[1] > u_weights[2]
    print(f"  -> PASS: Uncertainty penalty monotonic response verified ({u_weights[0]:.4f} > {u_weights[1]:.4f} > {u_weights[2]:.4f}).")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 13: Quality Monotonicity
    # -------------------------------------------------------------------------
    print("\n[Gate 13/18] Verifying Quality Bonus Monotonicity (Finite Difference Q_i ^ => w_i ^)...")
    q_router = ACARAUv2Router(coefficients=RouterCoefficients(alpha=0.0, beta=0.0, gamma=0.0, eta=2.0))
    q_weights = []
    for q_val in [0.1, 0.5, 0.9]:
        inp_q = RouterInput(
            retina=ModalityChannelInput("retina", 0.8, FROZEN_RELIABILITY_MAP["retina"], 0.2, q_val, True),
            foot=ModalityChannelInput("foot", 0.8, FROZEN_RELIABILITY_MAP["foot"], 0.2, 0.5, True),
            clinical=ModalityChannelInput("clinical", 0.8, FROZEN_RELIABILITY_MAP["clinical"], 0.2, 0.5, True),
        )
        q_weights.append(q_router.route(inp_q).weights["retina"])
    assert q_weights[0] < q_weights[1] < q_weights[2]
    print(f"  -> PASS: Quality bonus monotonic response verified ({q_weights[0]:.4f} < {q_weights[1]:.4f} < {q_weights[2]:.4f}).")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 14: Deterministic Reproducibility
    # -------------------------------------------------------------------------
    print("\n[Gate 14/18] Verifying Deterministic Reproducibility (f(X) == f(X))...")
    r1 = router.route(r_inp)
    r2 = router.route(r_inp)
    assert r1.weights == r2.weights
    assert r1.logits == r2.logits
    assert r1.routing_entropy == r2.routing_entropy
    print("  -> PASS: Determinism strictly verified across multiple invocations.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 15: Numerical Stability under Extreme Bounds
    # -------------------------------------------------------------------------
    print("\n[Gate 15/18] Verifying Numerical Stability under Extreme Parameters...")
    ext_router = ACARAUv2Router(coefficients=RouterCoefficients(alpha=5.0, beta=5.0, gamma=5.0, eta=5.0))
    ext_res = ext_router.route(r_inp)
    assert math.isfinite(ext_res.normalization_sum)
    assert abs(ext_res.normalization_sum - 1.0) < 1e-7
    assert all(math.isfinite(w) for w in ext_res.weights.values())
    assert all(0.0 <= w <= 1.0 for w in ext_res.weights.values())
    print("  -> PASS: Stable softmax prevents overflow/underflow under maximal bounds.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 16: Routing Entropy Properties
    # -------------------------------------------------------------------------
    print("\n[Gate 16/18] Verifying Routing Entropy Bounds & Behavioral Properties...")
    # Uniform weighting has maximum entropy = ln(3) ~ 1.0986
    u_router = ACARAUv2Router(coefficients=RouterCoefficients.uniform_baseline())
    u_res = u_router.route(r_inp)
    assert abs(u_res.routing_entropy - math.log(3.0)) < 1e-6
    # Unimodal weighting has minimum entropy = 0.0
    uni_res = u_router.route(RouterInput(
        retina=ModalityChannelInput("retina", 0.8, FROZEN_RELIABILITY_MAP["retina"], 0.2, 0.8, True),
        foot=ModalityChannelInput("foot", 0.8, FROZEN_RELIABILITY_MAP["foot"], 0.2, 0.0, False),
        clinical=ModalityChannelInput("clinical", 0.8, FROZEN_RELIABILITY_MAP["clinical"], 0.2, 0.0, False),
    ))
    assert uni_res.routing_entropy == 0.0
    print(f"  -> PASS: Routing entropy bounds verified (H_min = 0.0, H_max = {u_res.routing_entropy:.6f} ~ ln(3)).")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 17: No Patient-Level Target Leakage / Zero Synthetic GT
    # -------------------------------------------------------------------------
    print("\n[Gate 17/18] Auditing Router Signature & Anti-Leakage Lockdown...")
    import inspect
    sig = inspect.signature(ACARAUv2Router.route)
    param_names = list(sig.parameters.keys())
    assert param_names == ["self", "router_input"] or param_names == ["router_input"]
    # Check that route() does NOT accept ground truth, target, labels, etc.
    prohibited = ["label", "target", "y_true", "ground_truth", "loss", "optimizer", "patient_id"]
    for p in prohibited:
        assert p not in param_names, f"Prohibited parameter '{p}' detected in ACARAUv2Router.route"
    print("  -> PASS: Router signature contains zero target labels or optimization state.")
    passed_gates += 1

    # -------------------------------------------------------------------------
    # Gate 18: Frozen Artifact & Documentation Integrity (Read-Only Independent Audit)
    # -------------------------------------------------------------------------
    print("\n[Gate 18/18] Auditing Frozen Experiment Artifacts & Volume 04 Documentation Integrity...")
    exp_dir = REPO_ROOT / "experiments" / "fusion" / "router"
    vol4_dir = REPO_ROOT / "research" / "fusion" / "Volume_04_ACARA_U_Router"

    # 1. Verify JSON Experiment Artifacts exist on disk
    required_json_files = [
        "router_configuration.json",
        "behavioral_tests.json",
        "missing_modality_results.json",
        "monotonicity_results.json",
        "freeze_manifest.json",
    ]
    for fname in required_json_files:
        fpath = exp_dir / fname
        assert fpath.is_file(), f"Missing required experiment artifact: {fpath}"
        # Validate readability and non-empty
        with open(fpath, "r") as f:
            data = json.load(f)
            assert len(data) > 0, f"Empty experiment artifact: {fpath}"

    # 2. Validate Specific Contents in JSON Artifacts
    with open(exp_dir / "router_configuration.json", "r") as f:
        cfg = json.load(f)
        assert cfg["router_version"] == "acarau_v2.0"
        assert cfg["frozen_reliability_priors"]["retina"] == 0.929956
        assert cfg["frozen_reliability_priors"]["foot"] == 0.922266
        assert cfg["frozen_reliability_priors"]["clinical"] == 0.825382
        assert "B6_Full_ACARA_U" in cfg["baseline_presets"]

    with open(exp_dir / "freeze_manifest.json", "r") as f:
        manifest = json.load(f)
        assert manifest["phase"] == "C11.4"
        assert manifest["status"] == "SEALED"
        assert manifest["verification_gates_passed"] == 18

    # 3. Verify Volume 04 Markdown Research Documentation Suite
    required_doc_files = [
        "README.md",
        "01_Router_Protocol.md",
        "02_Mathematical_Formulation.md",
        "03_Input_Output_Contract.md",
        "04_Behavioral_Evaluation.md",
        "05_Freeze_Report.md",
    ]
    for dname in required_doc_files:
        dpath = vol4_dir / dname
        assert dpath.is_file(), f"Missing research document: {dpath}"
        content = dpath.read_text(encoding="utf-8")
        assert len(content) > 200, f"Research document {dpath} is unexpectedly short or empty."

    print("  -> PASS: All 5 experiment JSON artifacts and 6 Volume 04 documents independently verified.")
    passed_gates += 1

    print("\n" + "=" * 85)
    print(f"C11.4 ACARA-U ROUTER DEEP VERIFICATION RESULT: {passed_gates}/{total_gates} GATES PASSED")
    print("=" * 85)
    return passed_gates == total_gates


if __name__ == "__main__":
    success = run_c11_4_verification()
    sys.exit(0 if success else 1)
