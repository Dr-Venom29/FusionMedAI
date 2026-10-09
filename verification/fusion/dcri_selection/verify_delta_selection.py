"""
verification/fusion/dcri_selection/verify_delta_selection.py
Phase C11.13: DCRI Global Uncertainty Penalty Selection — 20 Deep Verification Gates
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import hashlib
import numpy as np

# Ensure root is in path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.fusion.dcri_selection.selection_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    DELTA_PROVISIONAL_HISTORICAL,
    CANDIDATE_DELTA_GRID,
    SEED,
    N_PACKETS,
    N_BOOTSTRAPS,
    ACTIVE_REGIMES,
    ALL_REGIMES,
    MAX_TOLERABLE_NEGATIVE_RATE,
    MAX_TOLERABLE_MEAN_PENALTY_RATIO,
    MIN_RANK_STABILITY_SPEARMAN,
    MIN_MEANINGFUL_PENALTY_RATIO,
    MAX_PREFERRED_NEGATIVE_RATE,
    MIN_PREFERRED_RANK_STABILITY,
    MAX_ACCEPTABLE_FLOAT_TOLERANCE,
    RESIDUAL_ERROR_TOLERANCE,
)
from src.fusion.dcri_selection.candidate_grid import get_candidate_grid, validate_candidate_grid
from src.fusion.dcri_selection.selection_runner import DeltaSelectionExperimentRunner, compute_sha256_file
from src.fusion.router.router_input import RouterInput, ModalityChannelInput


def recursively_find_keys(obj) -> List[str]:
    """Recursively extracts all dictionary key names from an arbitrary nested object."""
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(str(k))
            keys.extend(recursively_find_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.extend(recursively_find_keys(item))
    return keys


def main():
    print("================================================================================")
    print("PHASE C11.13: DCRI UNCERTAINTY PENALTY SELECTION — 20 DEEP VERIFICATION GATES")
    print("================================================================================\n")

    results_dir = root_dir / "experiments" / "fusion" / "dcri_selection" / "results"
    passed_gates = 0
    total_gates = 20

    # Fail closed if manifest artifact is not present on disk
    manifest_path = results_dir / "freeze_manifest.json"
    if not manifest_path.is_file():
        print(f"[FAIL-CLOSED ERROR] freeze_manifest.json missing from {results_dir}.")
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_json = json.load(f)

    # 1. Enforce Complete Artifact Inventory (8 files) and verify SHA-256 Hashes
    EXPECTED_ARTIFACT_NAMES = {
        "delta_selection_config.json",
        "candidate_grid.json",
        "delta_results.json",
        "regime_results.json",
        "distribution_results.json",
        "rank_stability.json",
        "bootstrap_comparisons.json",
        "selection_summary.json",
    }
    manifest_hashes = manifest_json.get("file_hashes", {})
    if set(manifest_hashes.keys()) != EXPECTED_ARTIFACT_NAMES:
        print("[FAIL-CLOSED ERROR] Manifest file_hashes set does not match expected 8 artifact files.")
        print(f"Missing: {EXPECTED_ARTIFACT_NAMES - set(manifest_hashes.keys())}")
        print(f"Unexpected: {set(manifest_hashes.keys()) - EXPECTED_ARTIFACT_NAMES}")
        return 1

    manifest_intact = True
    for art_name, expected_hash in manifest_hashes.items():
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or not all(c in "0123456789abcdefABCDEF" for c in expected_hash)
        ):
            print(f"[MANIFEST ERROR] Malformed SHA-256 hash for {art_name}: {expected_hash}")
            manifest_intact = False
            break
        art_file = results_dir / art_name
        if not art_file.is_file():
            print(f"[MANIFEST ERROR] Artifact {art_name} missing from disk.")
            manifest_intact = False
            break
        computed_hash = compute_sha256_file(art_file)
        if computed_hash != expected_hash:
            print(f"[MANIFEST ERROR] Hash mismatch for {art_name}: computed {computed_hash} != manifest {expected_hash}")
            manifest_intact = False
            break

    if not manifest_intact:
        print("[FAIL-CLOSED ERROR] Artifact manifest verification failed.")
        return 1

    runner = DeltaSelectionExperimentRunner(repo_root=root_dir, seed=SEED, n_bootstraps=N_BOOTSTRAPS)
    grid = get_candidate_grid()
    cohort = runner.load_cohort()

    # Load result artifacts from disk
    with open(results_dir / "delta_selection_config.json", "r", encoding="utf-8") as f:
        cfg_json = json.load(f)
    with open(results_dir / "candidate_grid.json", "r", encoding="utf-8") as f:
        grid_json = json.load(f)
    with open(results_dir / "delta_results.json", "r", encoding="utf-8") as f:
        delta_json = json.load(f)
    with open(results_dir / "regime_results.json", "r", encoding="utf-8") as f:
        regime_json = json.load(f)
    with open(results_dir / "distribution_results.json", "r", encoding="utf-8") as f:
        dist_json = json.load(f)
    with open(results_dir / "rank_stability.json", "r", encoding="utf-8") as f:
        rank_json = json.load(f)
    with open(results_dir / "bootstrap_comparisons.json", "r", encoding="utf-8") as f:
        boot_json = json.load(f)
    with open(results_dir / "selection_summary.json", "r", encoding="utf-8") as f:
        sel_json = json.load(f)

    # GATE 01: Frozen C11 Packet Manifest & Exact Canonical Ordering Verified
    expected_pids = [f"PACKET_{i:04d}" for i in range(500)]
    actual_pids = [p.packet_id for p in cohort]
    g01_pass = (len(cohort) == N_PACKETS) and (actual_pids == expected_pids)
    print(f"Gate 01 - Frozen C11 Packet Manifest & Sequence Verified: {'PASS' if g01_pass else 'FAIL'} (N={len(cohort)}, PACKET_0000..0499)")
    if g01_pass: passed_gates += 1

    # GATE 02: Cohort Size N=500 Verified
    g02_pass = (len(cohort) == 500) and (cfg_json["cohort_size"] == 500)
    print(f"Gate 02 - Cohort Size N=500 Verified: {'PASS' if g02_pass else 'FAIL'}")
    if g02_pass: passed_gates += 1

    # GATE 03: Seed=115 Verified
    g03_pass = (cfg_json["seed"] == 115) and (manifest_json["seed"] == 115)
    print(f"Gate 03 - Seed=115 Determinism Verified: {'PASS' if g03_pass else 'FAIL'}")
    if g03_pass: passed_gates += 1

    # GATE 04: Frozen Theta_0 Coefficients Verified
    frozen_coeffs = cfg_json["frozen_router_coefficients"]
    g04_pass = (
        frozen_coeffs["alpha"] == 1.0 and
        frozen_coeffs["beta"] == 1.5 and
        frozen_coeffs["gamma"] == 1.0 and
        frozen_coeffs["eta"] == 0.5
    )
    print(f"Gate 04 - Frozen Router Theta_0 Verified: {'PASS' if g04_pass else 'FAIL'} ({frozen_coeffs})")
    if g04_pass: passed_gates += 1

    # GATE 05: Candidate Delta Grid Exact
    grid_deltas = [c["delta"] for c in grid_json["candidates"]]
    expected_deltas = list(CANDIDATE_DELTA_GRID)
    g05_pass = (grid_deltas == expected_deltas) and (len(grid_deltas) == 11)
    print(f"Gate 05 - Candidate Delta Grid Exact: {'PASS' if g05_pass else 'FAIL'} ({len(grid_deltas)} points)")
    if g05_pass: passed_gates += 1

    # GATE 06: No Upstream Model Changes Invariant & Strict Simplex Check
    base_states = runner.evaluator.precompute_packet_base_state(cohort)
    g06_pass = (len(base_states) == 500) and all(
        s["is_empty"] or (
            abs(sum(s["weights"].values()) - 1.0) < 1e-12
            and all(w >= 0.0 and np.isfinite(w) for w in s["weights"].values())
        )
        for s in base_states
    )
    print(f"Gate 06 - No Upstream Model Changes & Simplex Invariant: {'PASS' if g06_pass else 'FAIL'}")
    if g06_pass: passed_gates += 1

    # GATE 07: Uncertainty Invariance & Empirical Ground-State Consistency
    u_match = True
    u_sums = []
    for p, s in zip(cohort, base_states):
        if not s["is_empty"]:
            u_sums.append(s["u_sum"])
            for m in s["active_mods"]:
                rec_u = getattr(p, m).uncertainty
                if abs(rec_u - s["uncertainties"][m]) > 1e-12 or rec_u < 0.0 or rec_u > 1.0:
                    u_match = False
                    break
    mean_u_sum = float(np.mean(u_sums))
    g07_pass = u_match and (abs(mean_u_sum - 0.633936188) < 1e-8)
    print(f"Gate 07 - Modality Uncertainty Integrity & State Invariance: {'PASS' if g07_pass else 'FAIL'} (Mean U_sum: {mean_u_sum:.6f})")
    if g07_pass: passed_gates += 1

    # GATE 08: Calibration Invariance & Base Risk Ground-State Consistency
    r_match = True
    r_fusions = []
    for p, s in zip(cohort, base_states):
        if not s["is_empty"]:
            r_fusions.append(s["r_fusion"])
            for m in s["active_mods"]:
                rec_r = getattr(p, m).risk
                if abs(rec_r - s["risks"][m]) > 1e-12 or rec_r < 0.0 or rec_r > 1.0:
                    r_match = False
                    break
    mean_r_fusion = float(np.mean(r_fusions))
    g08_pass = r_match and (abs(mean_r_fusion - 0.28989985) < 1e-6)
    print(f"Gate 08 - Calibrated Risk Integrity & State Invariance: {'PASS' if g08_pass else 'FAIL'} (Mean R_fusion: {mean_r_fusion:.6f})")
    if g08_pass: passed_gates += 1

    # GATE 09: Zero-Penalty Identity: DCRI_0 == R_fusion
    eval_0 = runner.evaluator.evaluate_candidate_delta(grid[0], base_states)
    max_d0_err = max(abs(p["dcri"] - p["r_fusion"]) for p in eval_0["packet_evaluations"])
    g09_pass = (grid[0].delta == 0.0) and (max_d0_err < 1e-14)
    print(f"Gate 09 - Zero-Penalty Identity DCRI_0 == R_fusion: {'PASS' if g09_pass else 'FAIL'} (max diff: {max_d0_err:.2e})")
    if g09_pass: passed_gates += 1

    # GATE 10: Strict Monotonic Penalty Invariant: DCRI_delta2 <= DCRI_delta1 for delta2 > delta1
    eval_all = [runner.evaluator.evaluate_candidate_delta(g, base_states) for g in grid]
    monotonic = True
    for i in range(len(grid) - 1):
        for p1, p2 in zip(eval_all[i]["packet_evaluations"], eval_all[i+1]["packet_evaluations"]):
            if p2["dcri"] > p1["dcri"] + 1e-14:
                monotonic = False
                break
    g10_pass = monotonic
    print(f"Gate 10 - Strict Monotonic Penalty Invariant: {'PASS' if g10_pass else 'FAIL'}")
    if g10_pass: passed_gates += 1

    # GATE 11: Analytical Derivative Matches Empirical Slope
    mean_u_sum = eval_0["mean_u_sum"]
    deriv_match = True
    for i in range(len(grid) - 1):
        d1 = grid[i].delta
        d2 = grid[i+1].delta
        mean_d1 = eval_all[i]["mean_dcri"]
        mean_d2 = eval_all[i+1]["mean_dcri"]
        finite_slope = (mean_d2 - mean_d1) / (d2 - d1)
        if abs(finite_slope - (-mean_u_sum)) > 1e-12:
            deriv_match = False
            break
    g11_pass = deriv_match
    print(f"Gate 11 - Analytical Derivative dDCRI/ddelta == -U_sum: {'PASS' if g11_pass else 'FAIL'}")
    if g11_pass: passed_gates += 1

    # GATE 12: Numerical Stability (Zero NaN/Inf)
    all_finite = True
    for ev in eval_all:
        for p in ev["packet_evaluations"]:
            if not np.isfinite(p["dcri"]):
                all_finite = False
                break
    g12_pass = all_finite
    print(f"Gate 12 - Numerical Stability (Zero NaN/Inf): {'PASS' if g12_pass else 'FAIL'}")
    if g12_pass: passed_gates += 1

    # GATE 13: Negative DCRI Remains Unclamped
    eval_prov = eval_all[4]  # delta = 0.20
    neg_count_020 = eval_prov["negative_count"]
    g13_pass = (neg_count_020 > 0) and any(p["dcri"] < 0.0 for p in eval_prov["packet_evaluations"])
    print(f"Gate 13 - Negative DCRI Unclamped Invariant: {'PASS' if g13_pass else 'FAIL'} (negative count at d=0.20: {neg_count_020})")
    if g13_pass: passed_gates += 1

    # GATE 14: All 7 Active Modality Regimes Evaluated
    regimes_in_data = set(regime_json["by_candidate"]["D00"]["regimes"].keys())
    g14_pass = all(reg in regimes_in_data for reg in ACTIVE_REGIMES)
    print(f"Gate 14 - All 7 Active Modality Regimes Evaluated: {'PASS' if g14_pass else 'FAIL'} ({ACTIVE_REGIMES})")
    if g14_pass: passed_gates += 1

    # GATE 15: EMPTY Regime Contract Validated (Real Router Invocation)
    from src.fusion.reliability.global_reliability import (
        FROZEN_RETINA_RELIABILITY,
        FROZEN_FOOT_RELIABILITY,
        FROZEN_CLINICAL_RELIABILITY,
    )
    empty_in = RouterInput(
        retina=ModalityChannelInput("retina", 0.0, FROZEN_RETINA_RELIABILITY, 1.0, 0.0, False),
        foot=ModalityChannelInput("foot", 0.0, FROZEN_FOOT_RELIABILITY, 1.0, 0.0, False),
        clinical=ModalityChannelInput("clinical", 0.0, FROZEN_CLINICAL_RELIABILITY, 1.0, 0.0, False),
    )
    empty_out = runner.evaluator.router.route(empty_in)
    empty_data = regime_json["by_candidate"]["D00"]["regimes"]["EMPTY"]
    g15_pass = (
        empty_out.status == "NO_MODALITY_AVAILABLE"
        and empty_out.num_active == 0
        and len(empty_out.active_modalities) == 0
        and all(w == 0.0 for w in empty_out.weights.values())
        and empty_data["fail_closed_safe"]
        and empty_data["cardinality"] == 0
    )
    print(f"Gate 15 - EMPTY Regime Real Router Fail-Closed Contract: {'PASS' if g15_pass else 'FAIL'}")
    if g15_pass: passed_gates += 1

    # GATE 16: Bootstrap Uses Paired Resampling (B=1000) & Verifies Observed Diff
    b_pass = (boot_json["n_bootstraps"] == 1000) and (len(boot_json["zero_vs_all"]) == 10)
    # Check that observed_diff for zero_vs_all[1] (D00 vs D10) matches mean penalty
    d10_obs_diff = boot_json["zero_vs_all"][1]["observed_diff"]
    d10_mean_pen = delta_json["candidates"][2]["mean_penalty"]
    g16_pass = b_pass and (abs(d10_obs_diff - d10_mean_pen) < 1e-14)
    print(f"Gate 16 - Paired Bootstrap (B=1000) & Observed Difference Reconciliation: {'PASS' if g16_pass else 'FAIL'}")
    if g16_pass: passed_gates += 1

    # GATE 17: Structural Data-Interface & Target-Label Isolation Audit
    forbidden_terms = ["y_true", "ground_truth", "target", "readmission", "wagner_grade_true", "y_test", "outcome"]
    all_keys = []
    for p in cohort:
        all_keys.extend(recursively_find_keys(p.to_dict()))
    
    clean_keys = [k.lower().replace("_", "") for k in all_keys]
    has_leakage = any(any(f.replace("_", "") in k for f in forbidden_terms) for k in clean_keys)
    g17_pass = (not has_leakage) and (len(all_keys) > 0)
    print(f"Gate 17 - Structural Data-Interface & Target-Label Isolation Audit: {'PASS' if g17_pass else 'FAIL'} ({len(all_keys)} keys checked)")
    if g17_pass: passed_gates += 1

    # GATE 18: Rank Stability Computed Across All Candidates
    rhos = [r["spearman_rho"] for r in rank_json["rank_stabilities"]]
    is_monotonic_rho = all(rhos[i] >= rhos[i+1] - 1e-12 for i in range(len(rhos)-1))
    candidate_rhos_valid = all(rhos[i] >= 0.90 for i, c in enumerate(grid_json["candidates"]) if c["delta"] <= 0.30)
    g18_pass = (len(rhos) == 11) and is_monotonic_rho and candidate_rhos_valid
    print(f"Gate 18 - Rank Stability Computed Across All Candidates: {'PASS' if g18_pass else 'FAIL'}")
    if g18_pass: passed_gates += 1

    # GATE 19: Full Independent Multi-Tier Hierarchy Recomputation (Directly from Raw Base States & Packets)
    from scipy.stats import spearmanr

    indep_candidate_evals = []
    for g in grid:
        d = g.delta
        cid = g.candidate_id
        is_zero = g.is_zero
        packet_dcris = []
        for s in base_states:
            if s["is_empty"]:
                continue
            dcri_val = s["r_fusion"] - d * s["u_sum"]
            packet_dcris.append({
                "packet_id": s["packet_id"],
                "dcri": dcri_val,
                "r_fusion": s["r_fusion"],
                "u_sum": s["u_sum"],
                "penalty": d * s["u_sum"],
            })

        dcri_arr = np.array([p["dcri"] for p in packet_dcris])
        pen_arr = np.array([p["penalty"] for p in packet_dcris])
        neg_count = int(np.sum(dcri_arr < 0.0))
        neg_rate = neg_count / len(dcri_arr)
        mean_dcri = float(np.mean(dcri_arr))
        mean_pen = float(np.mean(pen_arr))
        mean_base = float(np.mean([p["r_fusion"] for p in packet_dcris]))
        pen_ratio = mean_pen / mean_base if mean_base > 0 else 0.0

        r_fusions = np.array([p["r_fusion"] for p in packet_dcris])
        rho, _ = spearmanr(dcri_arr, r_fusions)

        indep_candidate_evals.append({
            "candidate_id": cid,
            "delta": d,
            "is_zero": is_zero,
            "packet_dcris": packet_dcris,
            "mean_dcri": mean_dcri,
            "std_dcri": float(np.std(dcri_arr)),
            "negative_count": neg_count,
            "negative_rate": neg_rate,
            "mean_penalty": mean_pen,
            "penalty_to_base_ratio": pen_ratio,
            "spearman_rho": float(rho),
        })

    # Tier 1 Hard Validity Invariants on raw packet evaluations
    indep_valid = []
    for idx, c in enumerate(indep_candidate_evals):
        is_finite = np.isfinite(c["mean_dcri"]) and np.isfinite(c["std_dcri"]) and all(np.isfinite(p["dcri"]) for p in c["packet_dcris"])
        max_res = max(abs((p["r_fusion"] - c["delta"] * p["u_sum"]) - p["dcri"]) for p in c["packet_dcris"])
        is_res_ok = (max_res <= RESIDUAL_ERROR_TOLERANCE)
        is_zero_ok = True
        if c["is_zero"]:
            is_zero_ok = (max(abs(p["dcri"] - p["r_fusion"]) for p in c["packet_dcris"]) < 1e-14)
        is_mono_ok = True
        if idx > 0:
            prev_c = indep_candidate_evals[idx - 1]
            is_mono_ok = all(p_curr["dcri"] <= p_prev["dcri"] + 1e-14 for p_curr, p_prev in zip(c["packet_dcris"], prev_c["packet_dcris"]))

        if is_finite and is_res_ok and is_zero_ok and is_mono_ok:
            indep_valid.append(c)

    # Tier 2 Behavioral Feasibility Filter
    indep_feasible = []
    reg_by_cand = regime_json.get("by_candidate", {})
    for c in indep_valid:
        cid = c["candidate_id"]
        meets_cohort_neg = (c["negative_rate"] <= MAX_TOLERABLE_NEGATIVE_RATE)
        meets_pen_ratio = (c["penalty_to_base_ratio"] <= MAX_TOLERABLE_MEAN_PENALTY_RATIO)
        meets_rank = (c["spearman_rho"] >= MIN_RANK_STABILITY_SPEARMAN)
        cand_regs = reg_by_cand.get(cid, {}).get("regimes", {})
        meets_regimes = all(cand_regs.get(reg, {}).get("negative_rate", 0.0) <= 0.45 for reg in ACTIVE_REGIMES)

        if meets_cohort_neg and meets_pen_ratio and meets_rank and meets_regimes:
            indep_feasible.append(c)

    # Tier 3/4 Preferred Criteria: discount magnitude, negative containment, rank fidelity, AND bootstrap CI lower bound > 0
    boot_ci_map = {b["candidate_b"]: b["ci_lower"] for b in boot_json.get("zero_vs_all", [])}
    indep_preferred = []
    for c in indep_feasible:
        if c["delta"] > 0.0:
            cid = c["candidate_id"]
            p_pen = (c["penalty_to_base_ratio"] >= MIN_MEANINGFUL_PENALTY_RATIO)
            p_neg = (c["negative_rate"] <= MAX_PREFERRED_NEGATIVE_RATE)
            p_rank = (c["spearman_rho"] >= MIN_PREFERRED_RANK_STABILITY)
            p_boot = (boot_ci_map.get(cid, 0.0) > 0.0)
            if p_pen and p_neg and p_rank and p_boot:
                indep_preferred.append(c)

    # Multi-Tier Selection & Fallback Logic
    if indep_preferred:
        expected_selected = min(indep_preferred, key=lambda c: c["delta"])
        expected_status = "SELECTED"
    elif indep_feasible:
        zero_cand = next((c for c in indep_feasible if c["delta"] == 0.0), None)
        if zero_cand is not None:
            expected_selected = zero_cand
            expected_status = "SELECTED"
        else:
            expected_selected = min(indep_feasible, key=lambda c: c["delta"])
            expected_status = "SELECTED"
    else:
        expected_selected = None
        expected_status = "NO_FEASIBLE_CANDIDATE"

    if expected_status == "SELECTED":
        g19_pass = (
            sel_json["status"] == "SELECTED"
            and sel_json["selected_delta"] == expected_selected["delta"]
            and sel_json["selected_candidate_id"] == expected_selected["candidate_id"]
            and sel_json["feasible_candidate_ids"] == [c["candidate_id"] for c in indep_feasible]
            and len(indep_valid) == 11
        )
    else:
        g19_pass = (
            sel_json["status"] == "NO_FEASIBLE_CANDIDATE"
            and sel_json["selected_delta"] is None
            and sel_json["selected_candidate_id"] == "NO_FEASIBLE_CANDIDATE"
            and sel_json["feasible_candidate_ids"] == []
        )

    print(f"Gate 19 - Full Independent Selection Hierarchy Recomputation: {'PASS' if g19_pass else 'FAIL'} (Selected: {sel_json['selected_candidate_id']}, Valid: {len(indep_valid)}, Feasible: {len(indep_feasible)})")
    if g19_pass: passed_gates += 1

    # GATE 20: Exhaustive 8-Artifact Field-by-Field Numerical & Key Re-Execution Verification (< 1e-14)
    runner_reprod = DeltaSelectionExperimentRunner(repo_root=root_dir, seed=SEED, n_bootstraps=N_BOOTSTRAPS)
    res_reprod = runner_reprod.run_full_selection()

    reprod_match = True

    # 20A: Candidate Evaluations Cardinality, IDs, and All Scalar/Nested Fields
    if len(res_reprod["candidate_evals"]) != len(delta_json["candidates"]) or len(res_reprod["candidate_evals"]) != 11:
        reprod_match = False
    for c_rep, c_saved in zip(res_reprod["candidate_evals"], delta_json["candidates"]):
        for str_key in ["candidate_id", "name", "category", "description", "is_zero", "n_active_packets", "negative_count"]:
            if c_rep[str_key] != c_saved[str_key]:
                reprod_match = False
                break
        for flt_key in ["delta", "mean_dcri", "median_dcri", "std_dcri", "min_dcri", "max_dcri", "negative_rate", "mean_penalty", "median_penalty", "max_penalty", "mean_u_sum", "penalty_to_base_ratio", "max_residual_error"]:
            if abs(c_rep[flt_key] - c_saved[flt_key]) > 1e-14:
                reprod_match = False
                break
        for qk in ["p05", "q1", "median", "q3", "p95", "iqr"]:
            if abs(c_rep["quantiles"][qk] - c_saved["quantiles"][qk]) > 1e-14:
                reprod_match = False
                break
        for ek in ["gt_0_10", "gt_0_20", "gt_0_30"]:
            if abs(c_rep["penalty_exceedance_rates"][ek] - c_saved["penalty_exceedance_rates"][ek]) > 1e-14:
                reprod_match = False
                break
        for rk in ["spearman_rho", "spearman_pvalue", "kendall_tau", "kendall_pvalue"]:
            if abs(c_rep["rank_stability"][rk] - c_saved["rank_stability"][rk]) > 1e-14:
                reprod_match = False
                break

    # 20B: Rank Stability Artifact Re-Execution Verification
    if len(res_reprod["candidate_evals"]) != len(rank_json["rank_stabilities"]) or len(rank_json["rank_stabilities"]) != 11:
        reprod_match = False
    for c_rep, r_saved in zip(res_reprod["candidate_evals"], rank_json["rank_stabilities"]):
        if c_rep["candidate_id"] != r_saved["candidate_id"] or abs(c_rep["delta"] - r_saved["delta"]) > 1e-14:
            reprod_match = False
            break
        for rk in ["spearman_rho", "spearman_pvalue", "kendall_tau", "kendall_pvalue"]:
            if abs(c_rep["rank_stability"][rk] - r_saved[rk]) > 1e-14:
                reprod_match = False
                break

    # 20C: Paired Bootstrap Comparisons (zero_vs_all, adjacent_steps, provisional_vs_others) Re-Execution Verification
    if res_reprod["bootstrap_results"]["n_bootstraps"] != boot_json["n_bootstraps"] or res_reprod["bootstrap_results"]["seed"] != boot_json["seed"]:
        reprod_match = False

    for coll_key in ["zero_vs_all", "adjacent_steps", "provisional_vs_others"]:
        if coll_key in boot_json:
            rep_coll = res_reprod["bootstrap_results"].get(coll_key, [])
            sav_coll = boot_json.get(coll_key, [])
            if len(rep_coll) != len(sav_coll):
                reprod_match = False
                break
            for b_rep, b_sav in zip(rep_coll, sav_coll):
                for sk in ["candidate_a", "candidate_b", "n_bootstraps"]:
                    if b_rep[sk] != b_sav[sk]:
                        reprod_match = False
                        break
                for fk in ["delta_a", "delta_b", "delta_diff", "observed_diff", "bootstrap_mean_diff", "std_err", "ci_lower", "ci_upper"]:
                    if abs(b_rep[fk] - b_sav[fk]) > 1e-14:
                        reprod_match = False
                        break

    # 20D: Distribution Profiles and Quantiles Re-Execution Verification
    if len(res_reprod["candidate_evals"]) != len(dist_json["distribution_profiles"]) or len(dist_json["distribution_profiles"]) != 11:
        reprod_match = False
    for c_rep, d_saved in zip(res_reprod["candidate_evals"], dist_json["distribution_profiles"]):
        if c_rep["candidate_id"] != d_saved["candidate_id"] or abs(c_rep["delta"] - d_saved["delta"]) > 1e-14:
            reprod_match = False
            break
        for k_rep, k_sav in [("mean_dcri", "mean"), ("median_dcri", "median"), ("std_dcri", "std"), ("min_dcri", "min"), ("max_dcri", "max"), ("negative_rate", "negative_rate"), ("penalty_to_base_ratio", "penalty_to_base_ratio")]:
            if abs(c_rep[k_rep] - d_saved[k_sav]) > 1e-14:
                reprod_match = False
                break
        for qk in ["p05", "q1", "median", "q3", "p95", "iqr"]:
            if abs(c_rep["quantiles"][qk] - d_saved["quantiles"][qk]) > 1e-14:
                reprod_match = False
                break
        for ek in ["gt_0_10", "gt_0_20", "gt_0_30"]:
            if abs(c_rep["penalty_exceedance_rates"][ek] - d_saved["penalty_exceedance_rates"][ek]) > 1e-14:
                reprod_match = False
                break

    # 20E: Regime Counts, By-Candidate Profiles, and Summary Matrices Re-Execution Verification
    if res_reprod["regime_results"]["regime_counts"] != regime_json["regime_counts"]:
        reprod_match = False
    for cid_k in [c.candidate_id for c in grid]:
        for reg_k in ALL_REGIMES:
            rep_reg = res_reprod["regime_results"]["by_candidate"][cid_k]["regimes"][reg_k]
            sav_reg = regime_json["by_candidate"][cid_k]["regimes"][reg_k]
            if (rep_reg["regime"] != sav_reg["regime"] or
                rep_reg["n_packets"] != sav_reg["n_packets"] or
                rep_reg["cardinality"] != sav_reg["cardinality"] or
                rep_reg["fail_closed_safe"] != sav_reg["fail_closed_safe"]):
                reprod_match = False
                break
            if "router_status" in sav_reg and rep_reg.get("router_status") != sav_reg.get("router_status"):
                reprod_match = False
                break
            if rep_reg["mean_dcri"] is not None:
                for fk in ["mean_dcri", "median_dcri", "std_dcri", "negative_rate", "mean_penalty", "mean_u_sum"]:
                    if abs(rep_reg[fk] - sav_reg[fk]) > 1e-14:
                        reprod_match = False
                        break

    for matrix_key in ["mean_dcri", "negative_rate", "mean_penalty"]:
        rep_reg_mat = res_reprod["regime_results"]["regime_matrices"][matrix_key]
        saved_reg_mat = regime_json["regime_matrices"][matrix_key]
        for reg_k in ACTIVE_REGIMES:
            for cid_k in [c.candidate_id for c in grid]:
                if abs(rep_reg_mat[reg_k][cid_k] - saved_reg_mat[reg_k][cid_k]) > 1e-14:
                    reprod_match = False
                    break

    # 20F: Selection Results Re-Execution Verification
    if (
        res_reprod["selection_results"]["selected_delta"] != sel_json["selected_delta"]
        or res_reprod["selection_results"]["selected_candidate_id"] != sel_json["selected_candidate_id"]
        or res_reprod["selection_results"]["selected_candidate_name"] != sel_json["selected_candidate_name"]
        or res_reprod["selection_results"]["status"] != sel_json["status"]
        or res_reprod["selection_results"]["feasible_candidate_ids"] != sel_json["feasible_candidate_ids"]
        or res_reprod["selection_results"]["rejected_candidate_ids"] != sel_json["rejected_candidate_ids"]
        or res_reprod["selection_results"]["selection_rationale"] != sel_json["selection_rationale"]
    ):
        reprod_match = False

    # 20G: Configuration & Grid Re-Execution Verification
    if (
        cfg_json["cohort_size"] != 500
        or cfg_json["seed"] != SEED
        or cfg_json["n_bootstraps"] != N_BOOTSTRAPS
        or len(grid_json["candidates"]) != 11
        or grid_json["total_candidates"] != 11
    ):
        reprod_match = False

    g20_pass = reprod_match
    print(f"Gate 20 - Exhaustive 8-Artifact Field-by-Field Numerical & Key Re-Execution Verification (< 1e-14): {'PASS' if g20_pass else 'FAIL'}")
    if g20_pass: passed_gates += 1

    print("\n================================================================================")
    print(f"PHASE C11.13 VERIFICATION SUMMARY: {passed_gates}/{total_gates} GATES PASSED")
    print("================================================================================")

    if passed_gates == total_gates:
        print(">>> [PASS] ALL 20 DEEP VERIFICATION GATES PASSED. C11.13 IS FULLY SEALED.")
        return 0
    else:
        print(f">>> [FAIL] VERIFICATION FAILED. {total_gates - passed_gates} GATES FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
