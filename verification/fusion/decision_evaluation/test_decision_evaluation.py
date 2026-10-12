"""
FusionMedAI - Phase C11.17 Decision-Level Evaluation: Comprehensive Test Suite.
Includes unit tests for metrics, engine validation, integration gate checks,
and negative mutation tests proving the verifier rejects corrupted artifacts.
"""

import pytest
import math
import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = REPO_ROOT / "experiments" / "fusion" / "decision_evaluation"
RESEARCH_DIR = REPO_ROOT / "research" / "fusion" / "Volume_17_Decision_Level_Evaluation"

from src.fusion.decision_evaluation.evaluation_config import (
    validate_evaluation_config,
    TAU_1_DEFAULT,
    TAU_2_DEFAULT,
    TIER_0_ROUTINE,
    TIER_1_ASSESSMENT,
    TIER_2_ESCALATION,
)
from src.fusion.decision_evaluation.evaluation_metrics import (
    calc_stats,
    compute_conflict_metrics,
    compute_dcri,
    assign_decision_tier,
    compute_paired_differences,
    compute_paired_bootstrap_cis,
    compute_decision_tier_transitions,
)
from src.fusion.decision_evaluation.evaluation_engine import validate_record
from verification.fusion.decision_evaluation.verify_decision_evaluation_artifacts import (
    DecisionEvaluationVerifier,
    GateFailure,
)


# =============================================================================
# Unit Tests: Metrics Primitives
# =============================================================================

def test_calc_stats_valid():
    data = [0.10, 0.20, 0.30, 0.40, 0.50]
    res = calc_stats(data)
    assert res["count"] == 5
    assert math.isclose(res["mean"], 0.30, abs_tol=1e-6)
    assert math.isclose(res["median"], 0.30, abs_tol=1e-6)
    assert math.isclose(res["min"], 0.10, abs_tol=1e-6)
    assert math.isclose(res["max"], 0.50, abs_tol=1e-6)


def test_calc_stats_empty_raises():
    with pytest.raises(ValueError, match="calc_stats encountered empty"):
        calc_stats([])

    with pytest.raises(ValueError, match="calc_stats encountered empty"):
        calc_stats([None, float("nan")])


def test_calc_stats_allow_empty():
    res = calc_stats([], allow_empty=True)
    assert res["count"] == 0
    assert res["mean"] is None


def test_compute_conflict_metrics():
    # Tri-modal
    risks = {"retina": 0.20, "foot": 0.50, "clinical": 0.10}
    weights = {"retina": 0.40, "foot": 0.40, "clinical": 0.20}
    r_fusion = 0.40 * 0.20 + 0.40 * 0.50 + 0.20 * 0.10  # 0.08 + 0.20 + 0.02 = 0.30
    conf = compute_conflict_metrics(["retina", "foot", "clinical"], risks, weights, r_fusion)
    assert math.isclose(conf["delta_max"], 0.40, abs_tol=1e-6)
    assert conf["css"] == conf["delta_max"]
    assert conf["conflict_category"] == "HIGH"

    # Singleton
    conf_sing = compute_conflict_metrics(["retina"], risks, weights, 0.20)
    assert conf_sing["css"] == 0.0
    assert conf_sing["conflict_category"] == "NOT_APPLICABLE"

    # Empty
    conf_empty = compute_conflict_metrics([], {}, {}, 0.0)
    assert conf_empty["css"] == 0.0
    assert conf_empty["conflict_category"] == "NO_MODALITY_AVAILABLE"


def test_compute_dcri_formula():
    dcri, p_u = compute_dcri(
        r_fusion=0.30,
        uncertainties={"retina": 0.20, "foot": 0.40, "clinical": 0.10},
        active_modalities=["retina", "foot", "clinical"],
        delta=0.10,
    )
    # p_u = 0.10 * (0.20 + 0.40 + 0.10) = 0.07
    # dcri = 0.30 - 0.07 = 0.23
    assert math.isclose(p_u, 0.07, abs_tol=1e-6)
    assert math.isclose(dcri, 0.23, abs_tol=1e-6)


def test_assign_decision_tier():
    assert assign_decision_tier(0.15) == TIER_0_ROUTINE
    assert assign_decision_tier(0.20) == TIER_1_ASSESSMENT
    assert assign_decision_tier(0.35) == TIER_1_ASSESSMENT
    assert assign_decision_tier(0.40) == TIER_2_ESCALATION
    assert assign_decision_tier(0.85) == TIER_2_ESCALATION
    assert assign_decision_tier(-0.05) == TIER_0_ROUTINE  # Unclamped negative


def test_compute_decision_tier_transitions():
    ref = [0, 0, 1, 1, 2]
    cand = [0, 1, 1, 0, 2]
    trans = compute_decision_tier_transitions(ref, cand)
    assert trans["total_packets"] == 5
    assert trans["total_unchanged"] == 3  # (0->0, 1->1, 2->2)
    assert trans["total_reclassified"] == 2  # (0->1, 1->0)
    assert trans["total_escalations"] == 1  # (0->1)
    assert trans["total_downgrades"] == 1  # (1->0)


def test_compute_aligned_tier_transitions():
    from src.fusion.decision_evaluation.evaluation_metrics import compute_aligned_tier_transitions
    ref_records = [
        {"packet_id": "P_001", "decision_tier": 0},
        {"packet_id": "P_002", "decision_tier": 1},
        {"packet_id": "P_003", "decision_tier": 2},
    ]
    # Shuffled candidate order
    cand_records = [
        {"packet_id": "P_003", "decision_tier": 2},
        {"packet_id": "P_001", "decision_tier": 1},  # Reclassified 0->1
        {"packet_id": "P_002", "decision_tier": 1},
    ]
    trans = compute_aligned_tier_transitions(ref_records, cand_records)
    assert trans["total_packets"] == 3
    assert trans["total_unchanged"] == 2
    assert trans["total_reclassified"] == 1
    assert trans["total_escalations"] == 1

    # Mismatched ID sets raise ValueError
    mismatched_cand = [
        {"packet_id": "P_999", "decision_tier": 0},
        {"packet_id": "P_002", "decision_tier": 1},
        {"packet_id": "P_003", "decision_tier": 2},
    ]
    with pytest.raises(ValueError, match="Packet ID set mismatch"):
        compute_aligned_tier_transitions(ref_records, mismatched_cand)


def test_compute_conflict_metrics_rejects_missing_or_nonfinite():
    # Missing active modality risk raises ValueError
    with pytest.raises(ValueError, match="missing from risks"):
        compute_conflict_metrics(["retina", "foot"], {"retina": 0.20}, {"retina": 0.5, "foot": 0.5}, 0.20)

    # Missing active modality weight raises ValueError
    with pytest.raises(ValueError, match="missing from weights"):
        compute_conflict_metrics(["retina", "foot"], {"retina": 0.20, "foot": 0.30}, {"retina": 1.0}, 0.25)

    # Non-finite r_fusion raises ValueError
    with pytest.raises(ValueError, match="r_fusion is non-finite"):
        compute_conflict_metrics(
            ["retina", "foot"],
            {"retina": 0.20, "foot": 0.30},
            {"retina": 0.5, "foot": 0.5},
            float("nan"),
        )


# =============================================================================
# Unit Tests: Configuration & Record Invariant Validation
# =============================================================================

def test_validate_evaluation_config():
    validate_evaluation_config()  # Should pass without error


def test_validate_record_valid():
    rec = {
        "packet_id": "P_001",
        "status": "SUCCESS",
        "is_finite": True,
        "active_modalities": ["retina", "foot", "clinical"],
        "num_active": 3,
        "weights": {"retina": 0.4, "foot": 0.4, "clinical": 0.2},
        "input_risks": {"retina": 0.25, "foot": 0.35, "clinical": 0.30},
        "input_uncertainties": {"retina": 0.20, "foot": 0.30, "clinical": 0.20},
        "input_qualities": {"retina": 0.80, "foot": 0.85, "clinical": 0.90},
        "input_reliabilities": {"retina": 0.82, "foot": 0.84, "clinical": 0.78},
        "input_confidences": {"retina": 0.85, "foot": 0.80, "clinical": 0.75},
        "r_fusion": 0.30,
        "u_sum": 0.70,
        "uncertainty_penalty": 0.07,
        "dcri": 0.23,
        "routing_entropy": 1.05,
        "css": 0.40,
        "delta_max": 0.40,
        "sigma_w": 0.15,
        "conflict_category": "HIGH",
        "decision_tier": 1,
    }
    validate_record(rec, delta=0.10)  # Should pass


def test_validate_record_catches_invalid_simplex():
    rec = {
        "packet_id": "P_001",
        "status": "SUCCESS",
        "is_finite": True,
        "active_modalities": ["retina", "foot", "clinical"],
        "num_active": 3,
        "weights": {"retina": 0.5, "foot": 0.5, "clinical": 0.5},  # sum = 1.5
        "input_risks": {"retina": 0.25, "foot": 0.35, "clinical": 0.30},
        "input_uncertainties": {"retina": 0.20, "foot": 0.30, "clinical": 0.20},
        "input_qualities": {"retina": 0.80, "foot": 0.85, "clinical": 0.90},
        "input_reliabilities": {"retina": 0.82, "foot": 0.84, "clinical": 0.78},
        "input_confidences": {"retina": 0.85, "foot": 0.80, "clinical": 0.75},
        "r_fusion": 0.30,
        "u_sum": 0.70,
        "uncertainty_penalty": 0.07,
        "dcri": 0.23,
        "routing_entropy": 1.05,
        "css": 0.40,
        "delta_max": 0.40,
        "sigma_w": 0.15,
        "conflict_category": "HIGH",
        "decision_tier": 1,
    }
    with pytest.raises(ValueError, match="weight sum"):
        validate_record(rec, delta=0.10)


def test_validate_record_catches_dcri_mismatch():
    rec = {
        "packet_id": "P_001",
        "status": "SUCCESS",
        "is_finite": True,
        "active_modalities": ["retina", "foot", "clinical"],
        "num_active": 3,
        "weights": {"retina": 0.4, "foot": 0.4, "clinical": 0.2},
        "input_risks": {"retina": 0.25, "foot": 0.35, "clinical": 0.30},
        "input_uncertainties": {"retina": 0.20, "foot": 0.30, "clinical": 0.20},
        "input_qualities": {"retina": 0.80, "foot": 0.85, "clinical": 0.90},
        "input_reliabilities": {"retina": 0.82, "foot": 0.84, "clinical": 0.78},
        "input_confidences": {"retina": 0.85, "foot": 0.80, "clinical": 0.75},
        "r_fusion": 0.30,
        "u_sum": 0.70,
        "uncertainty_penalty": 0.07,
        "dcri": 0.99,  # Mismatch: 0.30 - 0.07 != 0.99
        "routing_entropy": 1.05,
        "css": 0.40,
        "delta_max": 0.40,
        "sigma_w": 0.15,
        "conflict_category": "HIGH",
        "decision_tier": 1,
    }
    with pytest.raises(ValueError, match=r"DCRI .* != R_fusion"):
        validate_record(rec, delta=0.10)


# =============================================================================
# Acceptance Gate Integration Tests (All 16 Gates)
# =============================================================================

@pytest.fixture(scope="module")
def verifier():
    return DecisionEvaluationVerifier(repo_root=REPO_ROOT)


def test_gate_01_manifest(verifier):
    verifier.verify_c17_01_artifact_integrity()
    assert verifier.gate_results["Gate C17-01: Artifact Manifest & Checksum Integrity"]["passed"]


def test_gate_02_reproduction(verifier):
    verifier.verify_c17_02_reference_reproduction()
    assert verifier.gate_results["Gate C17-02: Reference B6 Reproduction Invariant"]["passed"]


def test_gate_03_simplex(verifier):
    verifier.verify_c17_03_simplex_preservation()
    assert verifier.gate_results["Gate C17-03: Simplex Preservation Across All Evaluated Policies"]["passed"]


def test_gate_04_mask(verifier):
    verifier.verify_c17_04_mask_enforcement()
    assert verifier.gate_results["Gate C17-04: Availability Mask Enforcement"]["passed"]


def test_gate_05_empty_regime(verifier):
    verifier.verify_c17_05_fail_closed_empty()
    assert verifier.gate_results["Gate C17-05: Fail-Closed EMPTY Availability Handling"]["passed"]


def test_gate_06_unimodal(verifier):
    verifier.verify_c17_06_unimodal_degeneracy()
    assert verifier.gate_results["Gate C17-06: Unimodal Singleton Degeneracy Invariance"]["passed"]


def test_gate_07_dcri_formula(verifier):
    verifier.verify_c17_07_unweighted_dcri_penalty()
    assert verifier.gate_results["Gate C17-07: Unweighted DCRI Penalty Orthogonality & Exact Form"]["passed"]


def test_gate_08_negative_dcri(verifier):
    verifier.verify_c17_08_negative_dcri_preservation()
    assert verifier.gate_results["Gate C17-08: Negative DCRI Preservation"]["passed"]


def test_gate_09_tier_conservation(verifier):
    verifier.verify_c17_09_decision_tier_conservation()
    assert verifier.gate_results["Gate C17-09: Decision Action Tier Conservation"]["passed"]


def test_gate_10_transitions(verifier):
    verifier.verify_c17_10_transition_conservation()
    assert verifier.gate_results["Gate C17-10: Baseline Transition Conservation"]["passed"]


def test_gate_11_quality_bonus(verifier):
    verifier.verify_c17_11_quality_bonus_directionality()
    assert verifier.gate_results["Gate C17-11: Quality Bonus Directionality & Specificity"]["passed"]


def test_gate_12_degradation(verifier):
    verifier.verify_c17_12_degradation_monotonicity()
    assert verifier.gate_results["Gate C17-12: Input Degradation Monotonicity / Response"]["passed"]


def test_gate_13_calibration(verifier):
    verifier.verify_c17_13_calibration_condition()
    assert verifier.gate_results["Gate C17-13: Calibration Condition Integrity"]["passed"]


def test_gate_14_conflict(verifier):
    verifier.verify_c17_14_conflict_stratification()
    assert verifier.gate_results["Gate C17-014: Continuous Discordance & Conflict Stratification"]["passed"]


def test_gate_15_bootstrap(verifier):
    verifier.verify_c17_15_bootstrap_reproducibility()
    assert verifier.gate_results["Gate C17-15: Paired Bootstrap Reproducibility"]["passed"]


def test_gate_16_documentation(verifier):
    verifier.verify_c17_16_documentation_presence()
    assert verifier.gate_results["Gate C17-16: Research Volume Structural Presence & Population Audit"]["passed"]


# =============================================================================
# Negative Mutation Tests: Demonstrating Verifier Catches Defects
# =============================================================================

def test_negative_manifest_missing_artifact(monkeypatch, tmp_path):
    """Manifest omitting a declared required artifact causes Gate C17-01 to fail."""
    import json
    test_verifier = DecisionEvaluationVerifier(repo_root=REPO_ROOT)
    # Tamper with manifest in a temp copy or by monkeypatching
    manifest_path = EXP_DIR / "09_artifact_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    mutated = copy.deepcopy(manifest)
    # Remove one required file
    del mutated["artifacts"]["04_summary_metrics.json"]
    
    # Write to tmp and point verifier
    temp_manifest = tmp_path / "09_artifact_manifest.json"
    with open(temp_manifest, "w", encoding="utf-8") as f:
        json.dump(mutated, f)

    orig_exp_dir = test_verifier.exp_dir
    monkeypatch.setattr(test_verifier, "exp_dir", tmp_path)
    # Copy other required files so only manifest entry is missing
    for f in orig_exp_dir.glob("*"):
        if f.is_file() and f.name != "09_artifact_manifest.json":
            (tmp_path / f.name).write_bytes(f.read_bytes())

    with pytest.raises(GateFailure):
        test_verifier.verify_c17_01_artifact_integrity()


def test_negative_manifest_corrupted_checksum(monkeypatch, tmp_path):
    """Manifest with mismatched SHA-256 causes Gate C17-01 to fail."""
    import json
    test_verifier = DecisionEvaluationVerifier(repo_root=REPO_ROOT)
    manifest_path = EXP_DIR / "09_artifact_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    mutated = copy.deepcopy(manifest)
    mutated["artifacts"]["04_summary_metrics.json"]["sha256"] = "0" * 64

    temp_manifest = tmp_path / "09_artifact_manifest.json"
    with open(temp_manifest, "w", encoding="utf-8") as f:
        json.dump(mutated, f)

    orig_exp_dir = test_verifier.exp_dir
    monkeypatch.setattr(test_verifier, "exp_dir", tmp_path)
    for f in orig_exp_dir.glob("*"):
        if f.is_file() and f.name != "09_artifact_manifest.json":
            (tmp_path / f.name).write_bytes(f.read_bytes())

    with pytest.raises(GateFailure):
        test_verifier.verify_c17_01_artifact_integrity()


def test_negative_reproduction_mutation(verifier):
    """Mutating canonical B6 risk causes Gate C17-02 to fail."""
    test_verifier = DecisionEvaluationVerifier(repo_root=REPO_ROOT)
    data = test_verifier.load_canonical_records()
    mutated = copy.deepcopy(data)
    mutated["baselines"]["B6"][0]["r_fusion"] = 0.999999
    test_verifier.canonical_data = mutated

    with pytest.raises(GateFailure):
        test_verifier.verify_c17_02_reference_reproduction()


def test_negative_simplex_mutation(verifier):
    """Mutating weights to violate simplex causes Gate C17-03 to fail."""
    test_verifier = DecisionEvaluationVerifier(repo_root=REPO_ROOT)
    data = test_verifier.load_canonical_records()
    mutated = copy.deepcopy(data)
    mutated["baselines"]["B6"][0]["weights"]["retina"] = 0.99
    test_verifier.canonical_data = mutated

    with pytest.raises(GateFailure):
        test_verifier.verify_c17_03_simplex_preservation()


def test_negative_mask_leak_mutation(verifier):
    """Allocating non-zero weight to an inactive modality causes Gate C17-04 to fail."""
    test_verifier = DecisionEvaluationVerifier(repo_root=REPO_ROOT)
    data = test_verifier.load_canonical_records()
    mutated = copy.deepcopy(data)
    # RF regime should have clinical = 0.0
    mutated["regimes_b6"]["RF"][0]["weights"]["clinical"] = 0.05
    test_verifier.canonical_data = mutated

    with pytest.raises(GateFailure):
        test_verifier.verify_c17_04_mask_enforcement()


def test_negative_duplicate_packet_id_mutation(verifier):
    """Duplicate packet ID within a regime causes Gate C17-04 to fail."""
    test_verifier = DecisionEvaluationVerifier(repo_root=REPO_ROOT)
    data = test_verifier.load_canonical_records()
    mutated = copy.deepcopy(data)
    mutated["regimes_b6"]["RFC"][1]["packet_id"] = mutated["regimes_b6"]["RFC"][0]["packet_id"]
    test_verifier.canonical_data = mutated

    with pytest.raises(GateFailure):
        test_verifier.verify_c17_04_mask_enforcement()


def test_negative_empty_regime_leak_mutation(verifier):
    """Non-zero risk in EMPTY regime causes Gate C17-05 to fail."""
    test_verifier = DecisionEvaluationVerifier(repo_root=REPO_ROOT)
    data = test_verifier.load_canonical_records()
    mutated = copy.deepcopy(data)
    mutated["regimes_b6"]["EMPTY"][0]["r_fusion"] = 0.25
    test_verifier.canonical_data = mutated

    with pytest.raises(GateFailure):
        test_verifier.verify_c17_05_fail_closed_empty()


def test_negative_empty_regime_nonzero_penalty(verifier):
    """Non-zero uncertainty penalty in EMPTY regime causes Gate C17-05 to fail."""
    test_verifier = DecisionEvaluationVerifier(repo_root=REPO_ROOT)
    data = test_verifier.load_canonical_records()
    mutated = copy.deepcopy(data)
    mutated["regimes_b6"]["EMPTY"][0]["uncertainty_penalty"] = 0.05
    test_verifier.canonical_data = mutated

    with pytest.raises(GateFailure):
        test_verifier.verify_c17_05_fail_closed_empty()


def test_negative_dcri_formula_mutation(verifier):
    """Violating DCRI = r_fusion - P_U causes Gate C17-07 to fail."""
    test_verifier = DecisionEvaluationVerifier(repo_root=REPO_ROOT)
    data = test_verifier.load_canonical_records()
    mutated = copy.deepcopy(data)
    mutated["baselines"]["B6"][0]["dcri"] = float(mutated["baselines"]["B6"][0]["r_fusion"]) + 0.10
    test_verifier.canonical_data = mutated

    with pytest.raises(GateFailure):
        test_verifier.verify_c17_07_unweighted_dcri_penalty()


def test_negative_decision_tier_mismatch_mutation(verifier):
    """Mismatched decision action tier causes Gate C17-09 to fail."""
    test_verifier = DecisionEvaluationVerifier(repo_root=REPO_ROOT)
    data = test_verifier.load_canonical_records()
    mutated = copy.deepcopy(data)
    # Change tier from whatever it is to something else
    orig_tier = mutated["baselines"]["B6"][0]["decision_tier"]
    mutated["baselines"]["B6"][0]["decision_tier"] = 2 if orig_tier != 2 else 0
    test_verifier.canonical_data = mutated

    with pytest.raises(GateFailure):
        test_verifier.verify_c17_09_decision_tier_conservation()


def test_negative_degradation_trend_contradiction(verifier):
    """Inverting the degradation penalty monotonicity causes Gate C17-12 to fail."""
    test_verifier = DecisionEvaluationVerifier(repo_root=REPO_ROOT)
    data = test_verifier.load_canonical_records()
    mutated = copy.deepcopy(data)
    # Swap D0 and D3 uncertainty penalties
    for r0, r3 in zip(mutated["degradations_b6"]["D0"], mutated["degradations_b6"]["D3"]):
        r0["uncertainty_penalty"], r3["uncertainty_penalty"] = r3["uncertainty_penalty"], r0["uncertainty_penalty"]
    test_verifier.canonical_data = mutated

    with pytest.raises(GateFailure):
        test_verifier.verify_c17_12_degradation_monotonicity()


def test_negative_calibration_transition_mutation(verifier):
    """Altering calibration transition classifications causes Gate C17-13 to fail."""
    test_verifier = DecisionEvaluationVerifier(repo_root=REPO_ROOT)
    data = test_verifier.load_canonical_records()
    mutated = copy.deepcopy(data)
    # Change an uncalibrated record's tier (original is 2, change to 0)
    mutated["calibration"]["uncalibrated"][0]["decision_tier"] = 0
    test_verifier.canonical_data = mutated

    with pytest.raises(GateFailure):
        test_verifier.verify_c17_13_calibration_condition()


def test_negative_validate_record_nonfinite():
    """validate_record rejects NaN, infinite, or None values."""
    bad_rec = {
        "packet_id": "TEST_001",
        "policy_id": "B6",
        "regime": "RFC",
        "active_modalities": ["retina", "foot", "clinical"],
        "num_active": 3,
        "weights": {"retina": float("nan"), "foot": 0.5, "clinical": 0.5},
        "input_risks": {"retina": 0.2, "foot": 0.3, "clinical": 0.4},
        "input_uncertainties": {"retina": 0.1, "foot": 0.1, "clinical": 0.1},
        "input_qualities": {"retina": 0.9, "foot": 0.9, "clinical": 0.9},
        "input_reliabilities": {"retina": 0.8, "foot": 0.8, "clinical": 0.8},
        "input_confidences": {"retina": 0.85, "foot": 0.85, "clinical": 0.85},
        "r_fusion": 0.3,
        "u_sum": 0.3,
        "uncertainty_penalty": 0.015,
        "dcri": 0.285,
        "routing_entropy": 1.0,
        "css": 0.2,
        "delta_max": 0.2,
        "sigma_w": 0.05,
        "conflict_category": "MODERATE",
        "decision_tier": 1,
        "status": "SUCCESS",
        "is_finite": True,
    }
    with pytest.raises(ValueError, match="weight retina is not finite"):
        validate_record(bad_rec, delta=0.05, tau_1=0.20, tau_2=0.40)


def test_negative_verifier_rejects_nan_in_canonical_record(verifier):
    """Independent verifier Gate C17-07 explicitly rejects a canonical record with NaN."""
    test_verifier = DecisionEvaluationVerifier(repo_root=REPO_ROOT)
    data = test_verifier.load_canonical_records()
    mutated = copy.deepcopy(data)
    # Inject NaN into a numerical field of a canonical record
    mutated["baselines"]["B6"][0]["dcri"] = float("nan")
    test_verifier.canonical_data = mutated

    with pytest.raises(GateFailure):
        test_verifier.verify_c17_07_unweighted_dcri_penalty()


def test_negative_regime_cohort_mismatch_mutation(verifier):
    """Gate C17-04 rejects a regime with 500 unique IDs that differs from canonical reference cohort."""
    test_verifier = DecisionEvaluationVerifier(repo_root=REPO_ROOT)
    data = test_verifier.load_canonical_records()
    mutated = copy.deepcopy(data)
    # Substitute one valid packet_id with an unknown unique packet_id
    mutated["regimes_b6"]["RFC"][0]["packet_id"] = "PACKET_SUBSTITUTED_FOREIGN_9999"
    test_verifier.canonical_data = mutated

    with pytest.raises(GateFailure):
        test_verifier.verify_c17_04_mask_enforcement()


def test_negative_empty_regime_nan_in_validate_record():
    """validate_record rejects NaN weights or metrics under NO_MODALITY_AVAILABLE."""
    empty_rec_nan_weight = {
        "packet_id": "TEST_EMPTY_001",
        "policy_id": "B6",
        "regime": "EMPTY",
        "active_modalities": [],
        "num_active": 0,
        "weights": {"retina": float("nan"), "foot": 0.0, "clinical": 0.0},
        "r_fusion": 0.0,
        "dcri": 0.0,
        "uncertainty_penalty": 0.0,
        "decision_tier": 0,
        "status": "NO_MODALITY_AVAILABLE",
        "is_finite": True,
    }
    with pytest.raises(ValueError, match="non-zero or non-finite weight retina"):
        validate_record(empty_rec_nan_weight)

    empty_rec_nan_metric = {
        "packet_id": "TEST_EMPTY_002",
        "policy_id": "B6",
        "regime": "EMPTY",
        "active_modalities": [],
        "num_active": 0,
        "weights": {"retina": 0.0, "foot": 0.0, "clinical": 0.0},
        "r_fusion": float("nan"),
        "dcri": 0.0,
        "uncertainty_penalty": 0.0,
        "decision_tier": 0,
        "status": "NO_MODALITY_AVAILABLE",
        "is_finite": True,
    }
    with pytest.raises(ValueError, match="non-zero or non-finite r_fusion"):
        validate_record(empty_rec_nan_metric)


def test_negative_empty_regime_nan_in_verifier(verifier):
    """Independent verifier Gate C17-05 rejects an EMPTY record with NaN weight or metric."""
    test_verifier = DecisionEvaluationVerifier(repo_root=REPO_ROOT)
    data = test_verifier.load_canonical_records()

    # Case 1: NaN weight
    mutated_weight = copy.deepcopy(data)
    mutated_weight["regimes_b6"]["EMPTY"][0]["weights"]["retina"] = float("nan")
    test_verifier.canonical_data = mutated_weight
    with pytest.raises(GateFailure):
        test_verifier.verify_c17_05_fail_closed_empty()

    # Case 2: NaN r_fusion
    mutated_metric = copy.deepcopy(data)
    mutated_metric["regimes_b6"]["EMPTY"][0]["r_fusion"] = float("nan")
    test_verifier.canonical_data = mutated_metric
    with pytest.raises(GateFailure):
        test_verifier.verify_c17_05_fail_closed_empty()


