"""
Pytest test suite for Phase C11.16 ACARA-U Component Ablation Study.
Includes:
- Unit tests for compute_paired_differences() ensuring variant - reference logic
- Unit tests for A5 discount orthogonality
- Regression tests for raw delta calculations
- Faulted-artifact rejection tests
- All 12 formal acceptance gates (C16-01 through C16-12)
"""

import pytest
import math
import json
import tempfile
from pathlib import Path
import sys
import numpy as np

# Portable repository root resolution
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fusion.component_ablation.ablation_metrics import compute_paired_differences
from verification.fusion.component_ablation.verify_ablation_artifacts import (
    check_c16_01_manifest_and_checksums,
    check_c16_02_reference_reproduction,
    check_c16_03_simplex_safety,
    check_c16_04_negative_control_rejection,
    check_c16_05_unimodal_regime_invariance,
    check_c16_06_dcri_discount_orthogonality,
    check_c16_07_uncertainty_penalty_directionality,
    check_c16_08_component_attribution_hierarchy,
    check_c16_09_decision_tier_transitions,
    check_c16_10_paired_bootstrap_determinism,
    check_c16_11_empty_regime_invariance,
    check_c16_12_documentation_completeness,
)

EXP_DIR = REPO_ROOT / "experiments" / "fusion" / "component_ablation"


# =============================================================================
# Direct Unit Tests for Statistical Metrics & Calculations
# =============================================================================

def test_compute_paired_differences_synthetic_unit():
    """Unit test for compute_paired_differences ensuring (v_rec - r_rec) is strictly computed."""
    ref_records = [
        {
            "packet_id": "P1",
            "weights": {"retina": 0.50, "foot": 0.30, "clinical": 0.20},
            "r_fusion": 0.35,
            "dcri": 0.25,
            "routing_entropy": 1.05,
        },
        {
            "packet_id": "P2",
            "weights": {"retina": 0.40, "foot": 0.40, "clinical": 0.20},
            "r_fusion": 0.45,
            "dcri": 0.35,
            "routing_entropy": 1.08,
        },
    ]

    var_records = [
        {
            "packet_id": "P1",
            "weights": {"retina": 0.60, "foot": 0.20, "clinical": 0.20},
            "r_fusion": 0.40,  # Delta: +0.05
            "dcri": 0.32,      # Delta: +0.07
            "routing_entropy": 1.02,
        },
        {
            "packet_id": "P2",
            "weights": {"retina": 0.30, "foot": 0.45, "clinical": 0.25},
            "r_fusion": 0.42,  # Delta: -0.03
            "dcri": 0.30,      # Delta: -0.05
            "routing_entropy": 1.10,
        },
    ]

    res = compute_paired_differences(var_records, ref_records)

    # Verify deltas are computed as var - ref
    raw_rf = res["raw_deltas"]["delta_rfusion"]
    raw_dcri = res["raw_deltas"]["delta_dcri"]
    assert pytest.approx(raw_rf[0], abs=1e-6) == 0.05
    assert pytest.approx(raw_rf[1], abs=1e-6) == -0.03
    assert pytest.approx(raw_dcri[0], abs=1e-6) == 0.07
    assert pytest.approx(raw_dcri[1], abs=1e-6) == -0.05

    # Explicit regression guards: assert deltas are strictly non-zero
    assert raw_rf[0] != 0.0, "delta_rfusion[0] must not be zero"
    assert raw_rf[1] != 0.0, "delta_rfusion[1] must not be zero"
    assert raw_dcri[0] != 0.0, "delta_dcri[0] must not be zero"
    assert raw_dcri[1] != 0.0, "delta_dcri[1] must not be zero"
    assert not all(d == 0.0 for d in raw_rf), "delta_rfusion cannot be identically zero"
    assert not all(d == 0.0 for d in raw_dcri), "delta_dcri cannot be identically zero"


def test_compute_paired_differences_rejects_self_subtraction_bug():
    """Direct test ensuring delta_rfusion and delta_dcri calculate variant - reference and never collapse to zero."""
    ref_records = [{"packet_id": "P1", "weights": {"retina": 0.5, "foot": 0.3, "clinical": 0.2}, "r_fusion": 0.30, "dcri": 0.20, "routing_entropy": 1.0}]
    var_records = [{"packet_id": "P1", "weights": {"retina": 0.6, "foot": 0.2, "clinical": 0.2}, "r_fusion": 0.35, "dcri": 0.28, "routing_entropy": 1.0}]

    diff = compute_paired_differences(var_records, ref_records)
    rf_diff = diff["raw_deltas"]["delta_rfusion"][0]
    dcri_diff = diff["raw_deltas"]["delta_dcri"][0]

    # If the bug (r_rec - r_rec or v_rec - v_rec) were present, rf_diff or dcri_diff would be 0.0
    assert rf_diff != 0.0, "Bug detected: delta_rfusion is zero for differing records"
    assert dcri_diff != 0.0, "Bug detected: delta_dcri is zero for differing records"
    assert pytest.approx(rf_diff, abs=1e-6) == 0.05
    assert pytest.approx(dcri_diff, abs=1e-6) == 0.08


def test_a5_dcri_discount_orthogonality_direct():
    """Direct test that A5 produces zero weight and risk deltas, but strictly positive DCRI delta across all individual packets."""
    with open(EXP_DIR / "results" / "ablation_results.json", "r", encoding="utf-8") as f:
        abl = json.load(f)["variants"]
    a0_records = abl["A0"]["packet_records"]
    a5_records = abl["A5"]["packet_records"]

    for i in range(500):
        assert a5_records[i]["packet_id"] == a0_records[i]["packet_id"]
        for m in ["retina", "foot", "clinical"]:
            assert abs(a5_records[i]["weights"][m] - a0_records[i]["weights"][m]) < 1e-9
        assert abs(a5_records[i]["r_fusion"] - a0_records[i]["r_fusion"]) < 1e-9
        dcri_shift = a5_records[i]["dcri"] - a0_records[i]["dcri"]
        assert dcri_shift > 0.0
        assert abs(dcri_shift - a0_records[i]["uncertainty_penalty"]) < 1e-9

    with open(EXP_DIR / "results" / "paired_comparisons.json", "r", encoding="utf-8") as f:
        paired = json.load(f)

    a5 = paired["A5"]["paired_differences"]
    for m in ["retina", "foot", "clinical"]:
        assert pytest.approx(a5["weight_deltas"][m]["stats"]["mean"], abs=1e-9) == 0.0

    assert pytest.approx(a5["r_fusion_delta"]["stats"]["mean"], abs=1e-9) == 0.0
    mean_dcri_delta = a5["dcri_delta"]["stats"]["mean"]
    assert 0.06 < mean_dcri_delta < 0.07
    assert a5["dcri_delta"]["positive_count"] == 500  # Every single packet has positive delta


def test_regression_raw_delta_matches_record_difference():
    """Regression test proving variant-minus-reference difference matches saved raw deltas for all packets."""
    with open(EXP_DIR / "results" / "reference_b6.json", "r", encoding="utf-8") as f:
        ref_records = json.load(f)["packet_records"]

    with open(EXP_DIR / "results" / "ablation_results.json", "r", encoding="utf-8") as f:
        abl_records = json.load(f)["variants"]["A3"]["packet_records"]

    with open(EXP_DIR / "results" / "paired_comparisons.json", "r", encoding="utf-8") as f:
        a3_paired = json.load(f)["A3"]["paired_differences"]["raw_deltas"]

    for i in range(len(ref_records)):
        expected_rf_delta = abl_records[i]["r_fusion"] - ref_records[i]["r_fusion"]
        expected_dcri_delta = abl_records[i]["dcri"] - ref_records[i]["dcri"]
        assert pytest.approx(a3_paired["delta_rfusion"][i], abs=1e-9) == expected_rf_delta
        assert pytest.approx(a3_paired["delta_dcri"][i], abs=1e-9) == expected_dcri_delta


def test_faulted_manifest_checksum_rejection():
    """Verify that verifier rejects an artifact if its SHA-256 hash does not match the manifest."""
    manifest_path = EXP_DIR / "freeze_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Tamper with a hash
    tampered_manifest = dict(manifest)
    tampered_manifest["files"] = dict(manifest["files"])
    tampered_manifest["files"]["protocol.json"] = {
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "size_bytes": 100,
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(tampered_manifest, tf)
        tf_path = Path(tf.name)

    try:
        passed, msg = check_c16_01_manifest_and_checksums(manifest_path=tf_path)
        assert passed is False, "Verifier must reject tampered manifest"
        assert "Hash mismatch for protocol.json" in msg, f"Expected hash mismatch in error message, got: {msg}"
    finally:
        if tf_path.exists():
            tf_path.unlink()


def test_upstream_manifest_size_mandatory_rejection(tmp_path):
    """Verify that Gate C16-01 rejects protocol missing or invalid source_manifest_size_bytes."""
    for req in [
        "protocol.json", "variants.json", "results/reference_b6.json",
        "results/ablation_results.json", "results/paired_comparisons.json",
        "results/regime_analysis.json", "results/failure_analysis.json"
    ]:
        p = tmp_path / req
        p.parent.mkdir(parents=True, exist_ok=True)
        orig = EXP_DIR / req
        p.write_bytes(orig.read_bytes())

    # 1. Test missing source_manifest_size_bytes
    proto_p = tmp_path / "protocol.json"
    with open(proto_p, "r", encoding="utf-8") as f:
        proto_data = json.load(f)
    del proto_data["cohort"]["source_manifest_size_bytes"]
    with open(proto_p, "w", encoding="utf-8") as f:
        json.dump(proto_data, f)

    from verification.fusion.component_ablation.verify_ablation_artifacts import sha256_file
    mock_manifest = {
        "files": {
            req: {"sha256": sha256_file(tmp_path / req), "size_bytes": (tmp_path / req).stat().st_size}
            for req in [
                "protocol.json", "variants.json", "results/reference_b6.json",
                "results/ablation_results.json", "results/paired_comparisons.json",
                "results/regime_analysis.json", "results/failure_analysis.json"
            ]
        }
    }
    manifest_p = tmp_path / "freeze_manifest.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(mock_manifest, f)

    passed, msg = check_c16_01_manifest_and_checksums(manifest_path=manifest_p, exp_dir=tmp_path)
    assert passed is False
    assert "Missing or invalid source_manifest_size_bytes" in msg


def test_a0_early_failure_persistence_on_exception(tmp_path, monkeypatch):
    """Verify that Stage 1 of ComponentAblationRunner persists failure_analysis.json and raises RuntimeError without TypeError when an A0 packet fails."""
    from src.fusion.component_ablation.ablation_runner import ComponentAblationRunner
    from src.fusion.component_ablation.ablation_engine import AblationEngine

    runner = ComponentAblationRunner(repo_root=REPO_ROOT, exp_dir=tmp_path)

    orig_eval_packet = AblationEngine.evaluate_packet
    call_count = [0]

    def mock_evaluate_packet(self, pkt, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("Simulated transient failure on packet P1")
        return orig_eval_packet(self, pkt, **kwargs)

    monkeypatch.setattr(AblationEngine, "evaluate_packet", mock_evaluate_packet)

    with pytest.raises(RuntimeError) as exc_info:
        runner.run_all()

    assert "A0 Reference reproduction failed" in str(exc_info.value)
    failure_report = tmp_path / "results" / "failure_analysis.json"
    assert failure_report.is_file(), "failure_analysis.json must be persisted even when A0 fails"

    with open(failure_report, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["status"] == "EARLY_STAGE_FAILURE"
    assert rep["a0_reproduction"]["packet_exceptions"] == 1


def test_variant_failure_persistence_on_exception(tmp_path, monkeypatch):
    """Verify that Stage 2/3 of ComponentAblationRunner persists failure_analysis.json and aborts before aggregation when a later variant packet fails."""
    from src.fusion.component_ablation.ablation_runner import ComponentAblationRunner
    from src.fusion.component_ablation.ablation_engine import AblationEngine

    runner = ComponentAblationRunner(repo_root=REPO_ROOT, exp_dir=tmp_path)

    orig_eval_packet = AblationEngine.evaluate_packet

    def mock_evaluate_packet(self, pkt, **kwargs):
        # Trigger failure specifically on variant A1 on first packet
        if self.variant.variant_id == "A1" and getattr(pkt, "packet_id", "").endswith("001"):
            raise RuntimeError("Simulated failure in variant A1")
        return orig_eval_packet(self, pkt, **kwargs)

    monkeypatch.setattr(AblationEngine, "evaluate_packet", mock_evaluate_packet)

    with pytest.raises(RuntimeError) as exc_info:
        runner.run_all()

    assert "Variant A1 evaluation failed" in str(exc_info.value)
    failure_report = tmp_path / "results" / "failure_analysis.json"
    assert failure_report.is_file(), "failure_analysis.json must be persisted when variant A1 fails"

    with open(failure_report, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["status"] == "EARLY_STAGE_FAILURE"
    assert rep["variant_failure"]["variant_id"] == "A1"


def test_regime_failure_persistence_on_exception(tmp_path, monkeypatch):
    """Verify that Stage 4 of ComponentAblationRunner persists failure_analysis.json and aborts before aggregation when a regime evaluation fails."""
    from src.fusion.component_ablation.ablation_runner import ComponentAblationRunner
    from src.fusion.component_ablation.ablation_engine import AblationEngine

    runner = ComponentAblationRunner(repo_root=REPO_ROOT, exp_dir=tmp_path)

    orig_eval_packet = AblationEngine.evaluate_packet

    def mock_evaluate_packet(self, pkt, regime_override=None, **kwargs):
        # Trigger failure specifically in Stage 4 during EMPTY regime on first packet
        if regime_override == "EMPTY" and getattr(pkt, "packet_id", "").endswith("001"):
            raise RuntimeError("Simulated failure in EMPTY regime")
        return orig_eval_packet(self, pkt, regime_override=regime_override, **kwargs)

    monkeypatch.setattr(AblationEngine, "evaluate_packet", mock_evaluate_packet)

    with pytest.raises(RuntimeError) as exc_info:
        runner.run_all()

    assert "regime EMPTY failed" in str(exc_info.value)
    failure_report = tmp_path / "results" / "failure_analysis.json"
    assert failure_report.is_file(), "failure_analysis.json must be persisted when regime EMPTY fails"

    with open(failure_report, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["status"] == "EARLY_STAGE_FAILURE"
    assert rep["regime_failure"]["regime"] == "EMPTY"


def test_calc_stats_filters_none_and_nan():
    """Verify that calc_stats() defensively ignores None, NaN, and Inf without raising TypeError."""
    from src.fusion.component_ablation.ablation_metrics import calc_stats

    raw = [0.1, None, 0.2, float("nan"), 0.3, float("inf"), 0.4]
    stats = calc_stats(raw)
    assert pytest.approx(stats["mean"], abs=1e-6) == 0.25
    assert stats["min"] == 0.1
    assert stats["max"] == 0.4

    empty_stats = calc_stats([None, float("nan")])
    assert empty_stats["mean"] == 0.0


def test_compute_paired_differences_rejects_null_inputs():
    """Verify that compute_paired_differences() raises ValueError when numeric fields contain None."""
    ref_records = [{"packet_id": "P1", "weights": {"retina": 0.5, "foot": 0.3, "clinical": 0.2}, "r_fusion": 0.30, "dcri": 0.20, "routing_entropy": 1.0}]
    var_records = [{"packet_id": "P1", "weights": {"retina": 0.6, "foot": 0.2, "clinical": 0.2}, "r_fusion": None, "dcri": 0.28, "routing_entropy": 1.0}]

    with pytest.raises(ValueError) as exc_info:
        compute_paired_differences(var_records, ref_records)
    assert "Null r_fusion encountered" in str(exc_info.value)


# =============================================================================
# Formal Acceptance Gates (C16-01 through C16-12)
# =============================================================================

def test_gate_c16_01():
    passed, msg = check_c16_01_manifest_and_checksums()
    assert passed, f"Gate C16-01 failed: {msg}"


def test_gate_c16_02():
    passed, msg = check_c16_02_reference_reproduction()
    assert passed, f"Gate C16-02 failed: {msg}"


def test_gate_c16_03():
    passed, msg = check_c16_03_simplex_safety()
    assert passed, f"Gate C16-03 failed: {msg}"


def test_gate_c16_04():
    passed, msg = check_c16_04_negative_control_rejection()
    assert passed, f"Gate C16-04 failed: {msg}"


def test_gate_c16_05():
    passed, msg = check_c16_05_unimodal_regime_invariance()
    assert passed, f"Gate C16-05 failed: {msg}"


def test_gate_c16_06():
    passed, msg = check_c16_06_dcri_discount_orthogonality()
    assert passed, f"Gate C16-06 failed: {msg}"


def test_gate_c16_07():
    passed, msg = check_c16_07_uncertainty_penalty_directionality()
    assert passed, f"Gate C16-07 failed: {msg}"


def test_gate_c16_08():
    passed, msg = check_c16_08_component_attribution_hierarchy()
    assert passed, f"Gate C16-08 failed: {msg}"


def test_gate_c16_09():
    passed, msg = check_c16_09_decision_tier_transitions()
    assert passed, f"Gate C16-09 failed: {msg}"


def test_gate_c16_10():
    passed, msg = check_c16_10_paired_bootstrap_determinism()
    assert passed, f"Gate C16-10 failed: {msg}"


def test_gate_c16_11():
    passed, msg = check_c16_11_empty_regime_invariance()
    assert passed, f"Gate C16-11 failed: {msg}"


def test_gate_c16_12():
    passed, msg = check_c16_12_documentation_completeness()
    assert passed, f"Gate C16-12 failed: {msg}"
