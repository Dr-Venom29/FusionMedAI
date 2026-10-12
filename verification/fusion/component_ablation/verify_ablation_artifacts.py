"""
Phase C11.16 Independent Verification Suite: ACARA-U Component Ablation Study.

Verifies acceptance gates C16-01 through C16-12:
- C16-01: Artifact Manifest & Checksum Integrity (7 payload artifacts + upstream packet manifest)
- C16-02: Reference B6 Reproduction Invariant (recomputed directly from individual packet records)
- C16-03: Simplex & Safety Invariant Preservation (recomputed directly from 3,500 packet records)
- C16-04: Negative Control Fault Rejection (A6 independently inspected on fault records, verdict derived)
- C16-05: Unimodal Regime Invariance (all 500 packets locked to active singleton unity w_i=1.0, inactive w_j=0.0)
- C16-06: DCRI Discount Orthogonality (A5 delta_w=0, delta_R=0, delta_DCRI=+0.063394)
- C16-07: Uncertainty Penalty Directionality (A3 surges foot weight and increases fused risk)
- C16-08: Component Attribution Hierarchy (Uncertainty & Confidence > Reliability > Quality)
- C16-09: Decision Tier Transition Integrity (3x3 transition conservation, zero 2-tier jumps)
- C16-10: Paired Bootstrap Determinism & Comprehensive Recomputation (fused risk, DCRI, entropy, weights)
- C16-11: Fail-Closed EMPTY Regime Invariance (R=0, DCRI=0, w=0 fallback)
- C16-12: Research Documentation Volume Structural Presence & Population Audit
"""

import sys
import os
import json
import math
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# Portably resolve repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "fusion" / "component_ablation"
DOCS_DIR = REPO_ROOT / "research" / "fusion" / "Volume_16_Component_Ablation_Study"

REQUIRED_PAYLOAD_ARTIFACTS = [
    "protocol.json",
    "variants.json",
    "results/reference_b6.json",
    "results/ablation_results.json",
    "results/paired_comparisons.json",
    "results/regime_analysis.json",
    "results/failure_analysis.json",
]

REQUIRED_DOCS = [
    "README.md",
    "01_Ablation_Protocol.md",
    "02_Variant_Specifications.md",
    "03_Reference_Reproduction.md",
    "04_Routing_Weight_Attribution.md",
    "05_Fused_Risk_Propagation.md",
    "06_DCRI_Propagation_and_Discount_Ablation.md",
    "07_Decision_Tier_Transitions.md",
    "08_Availability_Regime_Breakdown.md",
    "09_Negative_Control_Fault_Injection.md",
    "10_Component_Sensitivity_and_Failure_Modes.md",
    "11_Statistical_Synthesis.md",
    "12_Results.md",
    "13_Freeze_Report.md",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_c16_01_manifest_and_checksums(
    manifest_path: Optional[Path] = None,
    exp_dir: Optional[Path] = None,
) -> Tuple[bool, str]:
    """C16-01: Artifact Manifest & Checksum Integrity."""
    target_manifest = manifest_path if manifest_path else (EXP_DIR / "freeze_manifest.json")
    target_exp_dir = exp_dir if exp_dir else EXP_DIR

    if not target_manifest.exists():
        return False, f"Missing manifest: {target_manifest}"

    with open(target_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    file_entries = manifest.get("files", {})
    if len(file_entries) != len(REQUIRED_PAYLOAD_ARTIFACTS):
        return False, f"Expected {len(REQUIRED_PAYLOAD_ARTIFACTS)} manifest entries, found {len(file_entries)}"

    for rel_path in REQUIRED_PAYLOAD_ARTIFACTS:
        full_path = target_exp_dir / rel_path
        if not full_path.exists():
            return False, f"Missing required payload artifact: {rel_path}"
        actual_hash = sha256_file(full_path)
        actual_size = full_path.stat().st_size
        expected_entry = file_entries.get(rel_path)
        if not expected_entry:
            return False, f"Artifact not listed in manifest: {rel_path}"
        expected_hash = expected_entry.get("sha256")
        expected_size = expected_entry.get("size_bytes")
        if actual_hash != expected_hash:
            return False, f"Hash mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}"
        if expected_size is None:
            return False, f"Missing required size_bytes entry in manifest for {rel_path}"
        if actual_size != expected_size:
            return False, f"Size mismatch for {rel_path}: expected {expected_size} bytes, got {actual_size} bytes"

    # Verify input packet manifest matches upstream source
    with open(target_exp_dir / "protocol.json", "r", encoding="utf-8") as f:
        protocol = json.load(f)
    cohort_info = protocol.get("cohort", {})
    input_manifest = REPO_ROOT / cohort_info["source_manifest_path"]
    if not input_manifest.exists():
        return False, f"Input packet manifest does not exist: {input_manifest}"
    input_hash = sha256_file(input_manifest)
    if input_hash != cohort_info["source_manifest_sha256"]:
        return False, f"Input packet manifest hash mismatch: {input_hash} vs {cohort_info['source_manifest_sha256']}"
    expected_input_size = cohort_info.get("source_manifest_size_bytes")
    if expected_input_size is None or not isinstance(expected_input_size, int) or expected_input_size < 0:
        return False, f"Missing or invalid source_manifest_size_bytes in protocol cohort: {expected_input_size}"
    input_size = input_manifest.stat().st_size
    if input_size != expected_input_size:
        return False, f"Input packet manifest size mismatch: {input_size} vs {expected_input_size}"

    return True, f"7 payload artifacts verified against freeze_manifest.json; upstream packet manifest verified."


def check_c16_02_reference_reproduction() -> Tuple[bool, str]:
    """C16-02: Reference B6 Reproduction Invariant (recomputed directly from individual packet records)."""
    ref_path = EXP_DIR / "results" / "reference_b6.json"
    if not ref_path.exists():
        return False, f"Missing reference file: {ref_path}"

    with open(ref_path, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    records = ref_data.get("packet_records", [])
    if len(records) != 500:
        return False, f"Expected 500 packet records, got {len(records)}"

    # Recompute aggregate metrics directly from saved per-packet records
    recomputed_rf = float(np.mean([r["r_fusion"] for r in records]))
    recomputed_dcri = float(np.mean([r["dcri"] for r in records]))
    recomputed_pu = float(np.mean([r["uncertainty_penalty"] for r in records]))
    recomputed_neg = int(sum(1 for r in records if r["dcri"] < 0))

    # Reference B6 frozen values from Phase C11.13 and C11.15
    expected_rf = 0.28989984657769124
    expected_dcri = 0.22650622777769125
    expected_pu = 0.0633936188
    expected_neg = 37

    if abs(recomputed_rf - expected_rf) > 1e-6:
        return False, f"Recomputed mean fused risk mismatch: {recomputed_rf} vs {expected_rf}"
    if abs(recomputed_dcri - expected_dcri) > 1e-6:
        return False, f"Recomputed mean DCRI mismatch: {recomputed_dcri} vs {expected_dcri}"
    if abs(recomputed_pu - expected_pu) > 1e-6:
        return False, f"Recomputed mean uncertainty penalty mismatch: {recomputed_pu} vs {expected_pu}"
    if recomputed_neg != expected_neg:
        return False, f"Recomputed negative DCRI count mismatch: {recomputed_neg} vs {expected_neg}"

    return True, f"A0 reproduced from packet records: R={recomputed_rf:.6f}, DCRI={recomputed_dcri:.6f}, P_U={recomputed_pu:.6f}, neg={recomputed_neg}."


def check_c16_03_simplex_safety() -> Tuple[bool, str]:
    """C16-03: Simplex & Safety Invariant Preservation (recomputed directly from actual saved packet records)."""
    abl_path = EXP_DIR / "results" / "ablation_results.json"
    with open(abl_path, "r", encoding="utf-8") as f:
        abl_data = json.load(f)

    valid_variants = ["A0", "A1", "A2", "A3", "A4", "A5", "A_UNIFORM"]
    variants_dict = abl_data.get("variants", {})

    for v_id in valid_variants:
        v_entry = variants_dict.get(v_id)
        if not v_entry:
            return False, f"Missing results for variant {v_id}"
        records = v_entry.get("packet_records", [])
        if len(records) != 500:
            return False, f"Expected 500 records for {v_id}, got {len(records)}"

        for r in records:
            weights = r["weights"]
            active = r.get("evaluated_modalities", r.get("active_modalities", []))

            # Explicit finite-value check on every individual weight
            for m in ["retina", "foot", "clinical"]:
                w_val = weights.get(m, 0.0)
                if not math.isfinite(w_val):
                    return False, f"Non-finite weight {m}={w_val} in {v_id} for packet {r['packet_id']}"
                if w_val < -1e-9:
                    return False, f"Negative weight {m}={w_val} in {v_id} for packet {r['packet_id']}"

            active_sum = sum(weights.get(m, 0.0) for m in active)
            if abs(active_sum - 1.0) > 1e-6:
                return False, f"Simplex violation in {v_id} for packet {r['packet_id']}: sum={active_sum}"

            inactive = set(["retina", "foot", "clinical"]) - set(active)
            for m in inactive:
                if weights.get(m, 0.0) > 1e-6:
                    return False, f"Availability mask violation in {v_id} for packet {r['packet_id']}: {m}={weights.get(m)}"

            if not (math.isfinite(r["r_fusion"]) and math.isfinite(r["dcri"]) and math.isfinite(r["routing_entropy"])):
                return False, f"Non-finite output in {v_id} for packet {r['packet_id']}"

    return True, f"All 7 valid variants strictly recomputed: 0 simplex, 0 masking, 0 non-finite violations across 3,500 packet records."


def check_c16_04_negative_control_rejection() -> Tuple[bool, str]:
    """C16-04: Negative Control Fault Rejection (A6 independently inspected on fault records, verdict derived)."""
    fail_path = EXP_DIR / "results" / "failure_analysis.json"
    with open(fail_path, "r", encoding="utf-8") as f:
        fail_data = json.load(f)

    a6_entry = fail_data.get("negative_control_A6", {})
    fault_records = a6_entry.get("fault_records", [])
    if not fault_records or len(fault_records) != 500:
        return False, f"Expected 500 fault records in negative control A6, got {len(fault_records)}"

    # Independently inspect every packet record to detect deliberate mask leaks
    detected_violations = 0
    for r in fault_records:
        active = r.get("evaluated_modalities", r.get("active_modalities", []))
        weights = r.get("weights", {})
        # Under regime RF, clinical is inactive
        if "clinical" not in active and weights.get("clinical", 0.0) > 1e-6:
            detected_violations += 1

    if detected_violations != 500:
        return False, f"Verifier expected 500/500 mask breaches in A6, observed {detected_violations}"

    # Independently derive rejection verdict directly from observed violations
    derived_verdict = "REJECTED_AS_INVALID_CANDIDATE" if detected_violations > 0 else "VALID_CANDIDATE"
    if derived_verdict != "REJECTED_AS_INVALID_CANDIDATE":
        return False, f"Negative control A6 failed to trigger rejection verdict: derived={derived_verdict}"

    return True, f"A6 fault records independently audited: {detected_violations}/500 deliberate mask breaches detected; verifier independently derives {derived_verdict}."


def check_c16_05_unimodal_regime_invariance() -> Tuple[bool, str]:
    """C16-05: Unimodal Regime Invariance (all 500 packets locked to active singleton unity w_i=1.0, inactive w_j=0.0 across all variants)."""
    reg_path = EXP_DIR / "results" / "regime_analysis.json"
    with open(reg_path, "r", encoding="utf-8") as f:
        reg_data = json.load(f)

    all_variants = ["A0", "A1", "A2", "A3", "A4", "A5", "A_UNIFORM"]
    unimodal_regimes = ["R", "F", "C"]
    modalities = ["retina", "foot", "clinical"]
    active_map = {"R": "retina", "F": "foot", "C": "clinical"}

    for r in unimodal_regimes:
        active_mod = active_map[r]
        inactive_mods = [m for m in modalities if m != active_mod]

        a0_entry = reg_data.get("A0", {}).get(r, {})
        a0_records = a0_entry.get("packet_records", [])
        if len(a0_records) != 500:
            return False, f"Regime {r} A0 missing 500 packet records (got {len(a0_records)})"

        for v in all_variants:
            entry = reg_data.get(v, {}).get(r, {})
            records = entry.get("packet_records", [])
            if len(records) != 500:
                return False, f"Regime {r}, variant {v} missing 500 packet records (got {len(records)})"

            for pkt_idx in range(500):
                rec = records[pkt_idx]
                a0_rec = a0_records[pkt_idx]

                # Explicitly verify aligned packet IDs
                if rec["packet_id"] != a0_rec["packet_id"]:
                    return False, f"Regime {r}, variant {v} packet ID mismatch at {pkt_idx}: {rec['packet_id']} vs {a0_rec['packet_id']}"

                w = rec["weights"]
                if not (math.isfinite(w[active_mod]) and all(math.isfinite(w[in_m]) for in_m in inactive_mods)):
                    return False, f"Regime {r}, variant {v}, packet {rec['packet_id']} has non-finite weights"

                if abs(w[active_mod] - 1.0) > 1e-9:
                    return False, f"Regime {r}, variant {v}, packet {rec['packet_id']}: active weight {w[active_mod]} != 1.0"

                for in_m in inactive_mods:
                    if abs(w[in_m] - 0.0) > 1e-9:
                        return False, f"Regime {r}, variant {v}, packet {rec['packet_id']}: inactive weight {w[in_m]} != 0.0"

                # Verify fused risk invariance relative to reference A0
                if abs(rec["r_fusion"] - a0_rec["r_fusion"]) > 1e-9:
                    return False, f"Regime {r}, variant {v}, packet {rec['packet_id']}: delta R != 0.0 ({rec['r_fusion']} vs {a0_rec['r_fusion']})"

    return True, "Unimodal regimes (R, F, C) strictly verified across all 7 variants and 500 packet records: active w_i=1.0, inactive w_j=0.0, delta_R=0.0."


def check_c16_06_dcri_discount_orthogonality() -> Tuple[bool, str]:
    """C16-06: DCRI Discount Orthogonality (A5 recomputed directly from raw packet records)."""
    abl_path = EXP_DIR / "results" / "ablation_results.json"
    with open(abl_path, "r", encoding="utf-8") as f:
        abl_data = json.load(f)

    a0_records = abl_data["variants"]["A0"]["packet_records"]
    a5_records = abl_data["variants"]["A5"]["packet_records"]

    if len(a0_records) != 500 or len(a5_records) != 500:
        return False, f"Expected 500 records for A0 and A5, got {len(a0_records)} and {len(a5_records)}"

    delta_dcri_list = []
    for idx in range(500):
        a0_r = a0_records[idx]
        a5_r = a5_records[idx]
        if a0_r["packet_id"] != a5_r["packet_id"]:
            return False, f"Packet ID mismatch at index {idx}: {a5_r['packet_id']} vs {a0_r['packet_id']}"

        # Per-packet weight orthogonality
        for mod in ["retina", "foot", "clinical"]:
            w_diff = abs(a5_r["weights"][mod] - a0_r["weights"][mod])
            if w_diff > 1e-9:
                return False, f"A5 packet {a5_r['packet_id']} has non-zero routing weight shift for {mod}: {w_diff}"

        # Per-packet fused risk orthogonality
        rf_diff = abs(a5_r["r_fusion"] - a0_r["r_fusion"])
        if rf_diff > 1e-9:
            return False, f"A5 packet {a5_r['packet_id']} has non-zero fused risk shift: {rf_diff}"

        # Per-packet DCRI difference exact match against expected discount delta (uncertainty penalty)
        dcri_diff = a5_r["dcri"] - a0_r["dcri"]
        expected_discount = a0_r["uncertainty_penalty"]
        if abs(dcri_diff - expected_discount) > 1e-9:
            return False, f"A5 packet {a5_r['packet_id']} DCRI shift mismatch: {dcri_diff} vs expected {expected_discount}"
        if dcri_diff <= 0.0:
            return False, f"A5 packet {a5_r['packet_id']} non-positive DCRI shift: {dcri_diff}"

        delta_dcri_list.append(dcri_diff)

    mean_delta_dcri = float(np.mean(delta_dcri_list))
    if mean_delta_dcri <= 0.05 or mean_delta_dcri >= 0.08:
        return False, f"A5 delta DCRI {mean_delta_dcri} outside expected range [0.05, 0.08]"

    # Verify 0 negative DCRI packets directly from a5_records
    neg_count = sum(1 for r in a5_records if r["dcri"] < 0)
    if neg_count != 0:
        return False, f"A5 should have 0 negative DCRI values, observed {neg_count} in raw packet records"

    # Also confirm consistency with paired_comparisons.json
    paired_path = EXP_DIR / "results" / "paired_comparisons.json"
    with open(paired_path, "r", encoding="utf-8") as f:
        paired = json.load(f)
    stored_delta_dcri = paired.get("A5", {}).get("paired_differences", {}).get("dcri_delta", {}).get("stats", {}).get("mean", 0.0)
    if abs(stored_delta_dcri - mean_delta_dcri) > 1e-6:
        return False, f"Discrepancy between recomputed delta DCRI ({mean_delta_dcri}) and paired_comparisons.json ({stored_delta_dcri})"

    return True, f"A5 independently recomputed from packet records: delta_w=0 (all 500 pkts), delta_R=0 (all 500 pkts), delta_DCRI=+{mean_delta_dcri:.6f} (matches penalty exactly), negative_DCRI=0."


def check_c16_07_uncertainty_penalty_directionality() -> Tuple[bool, str]:
    """C16-07: Uncertainty Penalty Directionality (A3 vs A0 recomputed directly from raw packet records)."""
    abl_path = EXP_DIR / "results" / "ablation_results.json"
    with open(abl_path, "r", encoding="utf-8") as f:
        abl_data = json.load(f)

    a0_records = abl_data["variants"]["A0"]["packet_records"]
    a3_records = abl_data["variants"]["A3"]["packet_records"]

    if len(a0_records) != 500 or len(a3_records) != 500:
        return False, f"Expected 500 records for A0 and A3, got {len(a0_records)} and {len(a3_records)}"

    delta_foot_arr = []
    delta_rf_arr = []
    foot_pos, foot_neg, foot_zero = 0, 0, 0
    rf_pos, rf_neg, rf_zero = 0, 0, 0
    tol = 1e-6

    for idx in range(500):
        a0_r = a0_records[idx]
        a3_r = a3_records[idx]
        if a0_r["packet_id"] != a3_r["packet_id"]:
            return False, f"Packet ID mismatch at index {idx}: {a3_r['packet_id']} vs {a0_r['packet_id']}"

        dw_foot = a3_r["weights"]["foot"] - a0_r["weights"]["foot"]
        delta_foot_arr.append(dw_foot)
        if dw_foot > tol:
            foot_pos += 1
        elif dw_foot < -tol:
            foot_neg += 1
        else:
            foot_zero += 1

        drf = a3_r["r_fusion"] - a0_r["r_fusion"]
        delta_rf_arr.append(drf)
        if drf > tol:
            rf_pos += 1
        elif drf < -tol:
            rf_neg += 1
        else:
            rf_zero += 1

    mean_delta_foot = float(np.mean(delta_foot_arr))
    mean_delta_rf = float(np.mean(delta_rf_arr))

    # Cohort-level acceptance criteria
    if mean_delta_foot <= 0.05:
        return False, f"A3 expected foot ulcer weight increase > 0.05, got {mean_delta_foot:.6f}"
    if mean_delta_rf <= 0.01:
        return False, f"A3 expected fused risk increase > 0.01, got {mean_delta_rf:.6f}"

    # Per-packet directional counts verification: exact distribution enforcement
    if foot_pos != 500 or foot_neg != 0 or foot_zero != 0:
        return False, f"A3 foot weight directionality mismatch: expected 500 pos, 0 neg, 0 zero; got pos={foot_pos}, neg={foot_neg}, zero={foot_zero}"
    if rf_pos != 406 or rf_neg != 94 or rf_zero != 0:
        return False, f"A3 fused risk directionality mismatch: expected 406 pos, 94 neg, 0 zero; got pos={rf_pos}, neg={rf_neg}, zero={rf_zero}"

    # Also confirm consistency with paired_comparisons.json
    paired_path = EXP_DIR / "results" / "paired_comparisons.json"
    with open(paired_path, "r", encoding="utf-8") as f:
        paired = json.load(f)
    stored_delta_foot = paired.get("A3", {}).get("paired_differences", {}).get("weight_deltas", {}).get("foot", {}).get("stats", {}).get("mean", 0.0)
    if abs(stored_delta_foot - mean_delta_foot) > 1e-6:
        return False, f"Discrepancy in delta_foot: recomputed {mean_delta_foot} vs stored {stored_delta_foot}"

    return True, (
        f"A3 independently recomputed from packet records: foot weight delta=+{mean_delta_foot:.6f} "
        f"(100% positive: {foot_pos}/500, 0 neg), fused risk delta=+{mean_delta_rf:.6f} "
        f"(majority positive: {rf_pos}/500 pos, {rf_neg}/500 neg)."
    )


def check_c16_08_component_attribution_hierarchy() -> Tuple[bool, str]:
    """C16-08: Component Attribution Hierarchy (recomputed directly from raw packet records)."""
    abl_path = EXP_DIR / "results" / "ablation_results.json"
    with open(abl_path, "r", encoding="utf-8") as f:
        abl_data = json.load(f)

    a0_records = abl_data["variants"]["A0"]["packet_records"]

    def get_raw_retina_shifts(vid: str) -> Tuple[float, float]:
        v_records = abl_data["variants"][vid]["packet_records"]
        shifts = []
        abs_shifts = []
        for idx in range(500):
            if v_records[idx]["packet_id"] != a0_records[idx]["packet_id"]:
                raise ValueError(f"Packet ID mismatch in {vid} at {idx}")
            s = v_records[idx]["weights"]["retina"] - a0_records[idx]["weights"]["retina"]
            shifts.append(s)
            abs_shifts.append(abs(s))
        mean_abs_shift = float(np.mean(abs_shifts))
        abs_mean_signed_shift = abs(float(np.mean(shifts)))
        return mean_abs_shift, abs_mean_signed_shift

    mean_abs = {v: get_raw_retina_shifts(v)[0] for v in ["A1", "A2", "A3", "A4"]}
    abs_mean = {v: get_raw_retina_shifts(v)[1] for v in ["A1", "A2", "A3", "A4"]}

    # Hierarchy verification under mean absolute per-packet shift:
    # Confidence (A1) & Uncertainty (A3) > Reliability (A2) > Quality (A4)
    if not (mean_abs["A1"] > mean_abs["A2"] and mean_abs["A3"] > mean_abs["A2"] and mean_abs["A2"] > mean_abs["A4"]):
        return False, f"Mean absolute shift hierarchy violated: {mean_abs}"

    # Hierarchy verification under absolute value of cohort mean signed shift:
    # Uncertainty (A3) & Confidence (A1) > Reliability (A2) > Quality (A4)
    if not (abs_mean["A3"] > abs_mean["A2"] and abs_mean["A1"] > abs_mean["A2"] and abs_mean["A2"] > abs_mean["A4"]):
        return False, f"Absolute mean signed shift hierarchy violated: {abs_mean}"

    return True, (
        f"Attribution hierarchy verified under both metrics: "
        f"Mean absolute per-packet shift: A1={mean_abs['A1']:.4f}, A3={mean_abs['A3']:.4f} > A2={mean_abs['A2']:.4f} > A4={mean_abs['A4']:.4f}; "
        f"Absolute mean signed shift: A3={abs_mean['A3']:.4f}, A1={abs_mean['A1']:.4f} > A2={abs_mean['A2']:.4f} > A4={abs_mean['A4']:.4f}."
    )


def check_c16_09_decision_tier_transitions() -> Tuple[bool, str]:
    """C16-09: Decision Tier Transition Integrity (reconstructed directly from raw packet records)."""
    abl_path = EXP_DIR / "results" / "ablation_results.json"
    with open(abl_path, "r", encoding="utf-8") as f:
        abl_data = json.load(f)

    paired_path = EXP_DIR / "results" / "paired_comparisons.json"
    with open(paired_path, "r", encoding="utf-8") as f:
        paired = json.load(f)

    a0_records = abl_data["variants"]["A0"]["packet_records"]

    for vid in ["A1", "A2", "A3", "A4", "A5", "A_UNIFORM"]:
        v_records = abl_data["variants"][vid]["packet_records"]
        if len(v_records) != 500:
            return False, f"Variant {vid} missing 500 records (got {len(v_records)})"

        # Reconstruct 3x3 transition matrix from packet records
        reconstructed = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for idx in range(500):
            if v_records[idx]["packet_id"] != a0_records[idx]["packet_id"]:
                return False, f"Packet ID mismatch in {vid} at {idx}"
            t_ref = a0_records[idx]["decision_tier"]
            t_var = v_records[idx]["decision_tier"]
            reconstructed[t_ref][t_var] += 1

        # Check total sum
        total = sum(sum(row) for row in reconstructed)
        if total != 500:
            return False, f"Reconstructed transition matrix for {vid} sum is {total}, expected 500"

        # No 2-tier jumps (Routine <-> Urgent, i.e. reconstructed[0][2] or reconstructed[2][0])
        jump_02 = reconstructed[0][2]
        jump_20 = reconstructed[2][0]
        if jump_02 > 0 or jump_20 > 0:
            return False, f"Variant {vid} caused paradoxical 2-tier jump (0->2: {jump_02}, 2->0: {jump_20})"

        # Verify cell-by-cell identity against saved transition matrix in paired_comparisons.json
        saved_matrix = paired[vid]["tier_transitions"].get("transition_matrix", paired[vid]["tier_transitions"].get("matrix"))
        if reconstructed != saved_matrix:
            return False, f"Reconstructed matrix for {vid} differs from saved matrix: {reconstructed} vs {saved_matrix}"

    return True, "All transition matrices independently reconstructed from packet records: sum=500, match saved matrices cell-by-cell, 0 paradoxical 2-tier shifts."


def check_c16_10_paired_bootstrap_determinism() -> Tuple[bool, str]:
    """C16-10: Paired Bootstrap Determinism & Comprehensive Recomputation across all metrics from raw packet records."""
    abl_path = EXP_DIR / "results" / "ablation_results.json"
    with open(abl_path, "r", encoding="utf-8") as f:
        abl_data = json.load(f)

    paired_path = EXP_DIR / "results" / "paired_comparisons.json"
    with open(paired_path, "r", encoding="utf-8") as f:
        paired = json.load(f)

    a0_records = abl_data["variants"]["A0"]["packet_records"]

    rng = np.random.RandomState(7116)
    indices = rng.randint(0, 500, size=(1000, 500))

    for vid in ["A1", "A2", "A3", "A4", "A5", "A_UNIFORM"]:
        v_records = abl_data["variants"][vid]["packet_records"]
        cis = paired[vid].get("bootstrap_cis", {})

        # Reconstruct all raw paired differences directly from packet records
        delta_rf_list = []
        delta_dcri_list = []
        delta_entropy_list = []
        delta_weights_dict = {m: [] for m in ["retina", "foot", "clinical"]}

        for idx in range(500):
            if v_records[idx]["packet_id"] != a0_records[idx]["packet_id"]:
                return False, f"Packet ID mismatch in {vid} at {idx}"
            delta_rf_list.append(v_records[idx]["r_fusion"] - a0_records[idx]["r_fusion"])
            delta_dcri_list.append(v_records[idx]["dcri"] - a0_records[idx]["dcri"])
            delta_entropy_list.append(v_records[idx]["routing_entropy"] - a0_records[idx]["routing_entropy"])
            for m in ["retina", "foot", "clinical"]:
                delta_weights_dict[m].append(v_records[idx]["weights"][m] - a0_records[idx]["weights"][m])

        reconstructed_deltas = {
            "r_fusion": np.array(delta_rf_list, dtype=np.float64),
            "dcri": np.array(delta_dcri_list, dtype=np.float64),
            "routing_entropy": np.array(delta_entropy_list, dtype=np.float64),
        }

        # 1. Recompute scalar metrics: r_fusion, dcri, routing_entropy
        for m_name, arr in reconstructed_deltas.items():
            means = np.mean(arr[indices], axis=1)
            recomp_low = float(np.percentile(means, 2.5))
            recomp_high = float(np.percentile(means, 97.5))
            recomp_point = float(np.mean(arr))
            recomp_se = float(np.std(means, ddof=1))
            recomp_cz = bool(recomp_low <= 0.0 <= recomp_high)

            stored = cis.get(m_name, {})
            st_low = stored.get("ci_lower")
            st_high = stored.get("ci_upper")
            st_point = stored.get("point_estimate")
            st_se = stored.get("std_error")
            st_cz = stored.get("crosses_zero")

            if abs(recomp_point - st_point) > 1e-6:
                return False, f"Point estimate mismatch in {vid} for {m_name}: stored {st_point} vs reconstructed {recomp_point}"
            if abs(recomp_low - st_low) > 1e-5 or abs(recomp_high - st_high) > 1e-5:
                return False, f"Bootstrap recomputation mismatch for {vid} on {m_name}: stored [{st_low}, {st_high}] vs recomputed [{recomp_low}, {recomp_high}]"
            if abs(recomp_se - st_se) > 1e-5:
                return False, f"Standard error mismatch in {vid} for {m_name}: stored {st_se} vs recomputed {recomp_se}"
            if st_cz != recomp_cz:
                return False, f"crosses_zero flag mismatch in {vid} for {m_name}: stored {st_cz} vs recomputed {recomp_cz}"
            if not (recomp_low <= st_point <= recomp_high):
                return False, f"Point estimate {st_point} outside recomputed 95% CI [{recomp_low}, {recomp_high}] in {vid} for {m_name}"

        # 2. Recompute modality weight metrics: retina, foot, clinical
        for mod in ["retina", "foot", "clinical"]:
            arr = np.array(delta_weights_dict[mod], dtype=np.float64)
            means = np.mean(arr[indices], axis=1)
            recomp_low = float(np.percentile(means, 2.5))
            recomp_high = float(np.percentile(means, 97.5))
            recomp_point = float(np.mean(arr))
            recomp_se = float(np.std(means, ddof=1))
            recomp_cz = bool(recomp_low <= 0.0 <= recomp_high)

            stored = cis.get("weights", {}).get(mod, {})
            st_low = stored.get("ci_lower")
            st_high = stored.get("ci_upper")
            st_point = stored.get("point_estimate")
            st_se = stored.get("std_error")
            st_cz = stored.get("crosses_zero")

            if abs(recomp_point - st_point) > 1e-6:
                return False, f"Weight point estimate mismatch in {vid} for {mod}: stored {st_point} vs reconstructed {recomp_point}"
            if abs(recomp_low - st_low) > 1e-5 or abs(recomp_high - st_high) > 1e-5:
                return False, f"Bootstrap recomputation mismatch for {vid} on weight_{mod}: stored [{st_low}, {st_high}] vs recomputed [{recomp_low}, {recomp_high}]"
            if abs(recomp_se - st_se) > 1e-5:
                return False, f"Weight standard error mismatch in {vid} for {mod}: stored {st_se} vs recomputed {recomp_se}"
            if st_cz != recomp_cz:
                return False, f"Weight crosses_zero flag mismatch in {vid} for {mod}: stored {st_cz} vs recomputed {recomp_cz}"
            if not (recomp_low <= st_point <= recomp_high):
                return False, f"Point estimate {st_point} outside recomputed 95% CI in {vid} for weight_{mod}"

    return True, "Paired bootstrap intervals (B=1000, seed 7116) reconstructed directly from packet records and verified across all metrics and fields (point est, CI bounds, SE, crosses_zero)."


def check_c16_11_empty_regime_invariance() -> Tuple[bool, str]:
    """C16-11: Fail-Closed EMPTY Regime Invariance (recomputed directly from 500 individual packet records)."""
    reg_path = EXP_DIR / "results" / "regime_analysis.json"
    with open(reg_path, "r", encoding="utf-8") as f:
        reg_data = json.load(f)

    for vid in ["A0", "A1", "A2", "A3", "A4", "A5", "A_UNIFORM"]:
        if vid not in reg_data or "EMPTY" not in reg_data[vid]:
            return False, f"EMPTY regime missing for variant {vid}"
        entry = reg_data[vid]["EMPTY"]
        records = entry.get("packet_records", [])
        if len(records) != 500:
            return False, f"EMPTY regime {vid} missing exactly 500 packet records (got {len(records)})"

        for r in records:
            if not (math.isfinite(r["r_fusion"]) and math.isfinite(r["dcri"])):
                return False, f"EMPTY regime {vid} packet {r['packet_id']} has non-finite values"
            if abs(r["r_fusion"]) > 1e-9 or abs(r["dcri"]) > 1e-9:
                return False, f"EMPTY regime {vid} packet {r['packet_id']} non-zero: R={r['r_fusion']}, DCRI={r['dcri']}"
            for m in ["retina", "foot", "clinical"]:
                w_val = r["weights"].get(m, 0.0)
                if not math.isfinite(w_val) or abs(w_val) > 1e-9:
                    return False, f"EMPTY regime {vid} packet {r['packet_id']} weight {m}={w_val}"

    return True, "EMPTY regime fail-closed invariant strictly verified across all 7 variants and 3,500 individual packet records (R=0, DCRI=0, w=0, all finite)."


def check_c16_12_documentation_completeness() -> Tuple[bool, str]:
    """C16-12: Research Documentation Volume Structural Presence & Population Audit."""
    if not DOCS_DIR.exists():
        return False, f"Volume 16 directory does not exist: {DOCS_DIR}"

    missing = []
    for doc in REQUIRED_DOCS:
        doc_path = DOCS_DIR / doc
        if not doc_path.exists():
            missing.append(f"{doc} (missing)")
            continue

        size = doc_path.stat().st_size
        min_size = 500 if doc == "README.md" else 1500
        if size < min_size:
            missing.append(f"{doc} (too small: {size} bytes < {min_size})")
            continue

        content = doc_path.read_text(encoding="utf-8")
        if "#" not in content or ("##" not in content and doc != "README.md"):
            missing.append(f"{doc} (lacks structured markdown sections)")
        if len(content.splitlines()) < 20:
            missing.append(f"{doc} (insufficient lines: {len(content.splitlines())})")

        # Substantive content audit: verify scientific elements and lack of placeholders
        placeholders = ["TODO", "TBD", "PLACEHOLDER", "FIXME"]
        content_upper = content.upper()
        for p in placeholders:
            if p in content_upper:
                missing.append(f"{doc} (contains unresolved placeholder '{p}')")
        if not ("|" in content or "$" in content or "ACARA-U" in content):
            missing.append(f"{doc} (lacks substantive scientific tables, equations, or protocol terms)")

    if missing:
        return False, f"Documentation completeness audit failed: {', '.join(missing)}"

    return True, f"All {len(REQUIRED_DOCS)} research volume documents structurally complete, substantively populated, and audit-verified."


GATES = [
    ("C16-01", "Artifact Manifest & Checksum Integrity", check_c16_01_manifest_and_checksums),
    ("C16-02", "Reference B6 Reproduction Invariant", check_c16_02_reference_reproduction),
    ("C16-03", "Simplex & Safety Invariant Preservation", check_c16_03_simplex_safety),
    ("C16-04", "Negative Control Fault Rejection (A6)", check_c16_04_negative_control_rejection),
    ("C16-05", "Unimodal Regime Invariance", check_c16_05_unimodal_regime_invariance),
    ("C16-06", "DCRI Discount Orthogonality", check_c16_06_dcri_discount_orthogonality),
    ("C16-07", "Uncertainty Penalty Directionality", check_c16_07_uncertainty_penalty_directionality),
    ("C16-08", "Component Attribution Hierarchy", check_c16_08_component_attribution_hierarchy),
    ("C16-09", "Decision Tier Transition Integrity", check_c16_09_decision_tier_transitions),
    ("C16-10", "Paired Bootstrap Determinism & Comprehensive Recomputation", check_c16_10_paired_bootstrap_determinism),
    ("C16-11", "Fail-Closed EMPTY Regime Invariance", check_c16_11_empty_regime_invariance),
    ("C16-12", "Research Documentation Volume Structural Presence & Content-Marker Audit", check_c16_12_documentation_completeness),
]


def run_all_checks() -> bool:
    print("=" * 80)
    print(" PHASE C11.16 INDEPENDENT VERIFICATION AUDIT: ACARA-U COMPONENT ABLATION ")
    print("=" * 80)

    all_passed = True
    passed_count = 0

    for gate_id, title, func in GATES:
        try:
            passed, msg = func()
        except Exception as e:
            passed = False
            msg = f"Exception during check: {str(e)}"

        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"{status_str} {gate_id}: {title}")
        print(f"       Details: {msg}")

        if passed:
            passed_count += 1
        else:
            all_passed = False

    print("=" * 80)
    print(f"AUDIT SUMMARY: {passed_count}/{len(GATES)} GATES CERTIFIED")
    if all_passed:
        print("VERDICT: PHASE C11.16 CERTIFICATION APPROVED (ALL GATES PASSED)")
    else:
        print("VERDICT: PHASE C11.16 NOT APPROVED (GATES REMAIN UNMET)")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
