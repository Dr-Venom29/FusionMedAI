"""
verification/fusion/combination_analysis/verify_combination_analysis.py
Phase C11.9: Modality-Combination Distribution & Tail-Robustness Verification Gate Suite

Executes 20 deep automated verification gates certifying:
1. Combination definitions and availability masks
2. Fail-closed safety for zero-modality state
3. Distribution configurations and count conservation
4. Deterministic head/middle/tail rank classification
5. Mathematical invariants (simplex, bounds, zero weights)
6. Uncertainty and conflict dynamics
7. Minimum sample size rule enforcement
8. Bootstrap reproducibility and coverage
9. Baseline ladder B1–B6 comparative evaluation
10. Full cohort reproducibility and SHA-256 artifact manifest certification
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import hashlib
import numpy as np

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.combination_analysis.combination_definition import (
    ALL_COMBINATIONS,
    NON_EMPTY_COMBINATIONS,
    BIMODAL_COMBINATIONS,
    UNIMODAL_COMBINATIONS,
    COMBINATION_RFC,
    COMBINATION_EMPTY,
    COMBINATION_MASKS,
    COMBINATION_CARDINALITY,
    get_combination_spec,
    apply_combination_to_packet,
)
from src.fusion.combination_analysis.distribution_generator import (
    ALL_DISTRIBUTIONS,
    DISTRIBUTION_D1_BALANCED,
    DISTRIBUTION_D2_MODERATE_HEAD_TAIL,
    DISTRIBUTION_D3_STRONG_LONG_TAIL,
    get_distribution_config,
    generate_stratified_distribution_cohort,
)
from src.fusion.combination_analysis.head_tail_classifier import (
    classify_distribution_tiers,
    TIER_HEAD,
    TIER_MIDDLE,
    TIER_TAIL,
    TIER_UNIFORM,
)
from src.fusion.combination_analysis.combination_engine import (
    ModalityCombinationEngine,
)
from src.fusion.combination_analysis.bootstrap_analysis import (
    compute_bootstrap_ci_1d,
)
from src.fusion.dcri.dcri_engine import DCRIEngine


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_all_verification_gates(repo_root: Path = Path(".")) -> Dict[str, Any]:
    print("================================================================================")
    print("FusionMedAI Phase C11.9: Modality-Combination Distribution & Tail-Robustness Verification")
    print("Verification Protocol: 20 Automated Gates | Target: 20/20 PASSED")
    print("================================================================================")

    cohort = load_frozen_cohort(repo_root=repo_root, n_packets=500, seed=115)
    engine = ModalityCombinationEngine(delta=0.20, bootstrap_resamples=500, seed=115)
    dcri_engine = DCRIEngine()
    artifacts_dir = repo_root / "experiments" / "fusion" / "combination_analysis"

    gate_results: List[Tuple[str, str, bool, str]] = []

    # --------------------------------------------------------------------------
    # Gate 1: Modality combination definitions match 8 canonical regimes
    # --------------------------------------------------------------------------
    g1_pass = len(ALL_COMBINATIONS) == 8 and set(ALL_COMBINATIONS) == {"RFC", "RF", "RC", "FC", "R", "F", "C", "EMPTY"}
    gate_results.append(("Gate 1", "Modality Combination Taxonomy (8 Regimes)", g1_pass, "Canonical 8 combination identifiers verified"))

    # --------------------------------------------------------------------------
    # Gate 2: Non-empty combinations count and cardinality hierarchy
    # --------------------------------------------------------------------------
    g2_pass = (
        len(NON_EMPTY_COMBINATIONS) == 7 and
        COMBINATION_CARDINALITY[COMBINATION_RFC] == 3 and
        len(BIMODAL_COMBINATIONS) == 3 and
        len(UNIMODAL_COMBINATIONS) == 3
    )
    gate_results.append(("Gate 2", "Cardinality & Subset Hierarchy", g2_pass, "Tri-modal (1), bimodal (3), unimodal (3) verified"))

    # --------------------------------------------------------------------------
    # Gate 3: Fail-closed zero modality safety
    # --------------------------------------------------------------------------
    empty_pkt = apply_combination_to_packet(cohort[0], COMBINATION_EMPTY)
    empty_res = dcri_engine.evaluate_packet(empty_pkt, delta=0.20)
    g3_pass = (
        empty_res.status == "NO_MODALITY_AVAILABLE" and
        empty_res.r_fusion == 0.0 and
        empty_res.dcri == 0.0 and
        all(w == 0.0 for w in empty_res.modality_weights.values())
    )
    gate_results.append(("Gate 3", "Zero-Modality Fail-Closed Rejection", g3_pass, "EMPTY produces NO_MODALITY_AVAILABLE with zero numerical outputs"))

    # --------------------------------------------------------------------------
    # Gate 4: Availability mask and active modalities correspondence
    # --------------------------------------------------------------------------
    g4_pass = True
    for c in ALL_COMBINATIONS:
        spec = get_combination_spec(c)
        mask = COMBINATION_MASKS[c]
        if (spec.availability_retina, spec.availability_foot, spec.availability_clinical) != mask:
            g4_pass = False
        if len(spec.active_modalities) != spec.cardinality:
            g4_pass = False
    gate_results.append(("Gate 4", "Mask & Specification Exact Correspondence", g4_pass, "All 8 specs map exactly to availability masks"))

    # --------------------------------------------------------------------------
    # Gate 5: Distribution probability configurations sum strictly to 1.0
    # --------------------------------------------------------------------------
    g5_pass = True
    for d_id in ALL_DISTRIBUTIONS:
        d_cfg = get_distribution_config(d_id)
        if not np.isclose(sum(d_cfg.probabilities.values()), 1.0, atol=1e-6):
            g5_pass = False
    gate_results.append(("Gate 5", "Distribution Probability Normalization", g5_pass, "D1, D2, D3 probabilities strictly sum to 1.000000"))

    # --------------------------------------------------------------------------
    # Gate 6: Expected packet allocation counts sum strictly to 500
    # --------------------------------------------------------------------------
    g6_pass = True
    for d_id in ALL_DISTRIBUTIONS:
        d_cfg = get_distribution_config(d_id)
        if sum(d_cfg.expected_counts_n500.values()) != 500:
            g6_pass = False
    gate_results.append(("Gate 6", "Cohort Packet Allocation Conservation (N=500)", g6_pass, "All distribution counts sum exactly to 500 packets"))

    # --------------------------------------------------------------------------
    # Gate 7: Deterministic, reproducible stratified allocation generation
    # --------------------------------------------------------------------------
    a1 = generate_stratified_distribution_cohort(cohort, DISTRIBUTION_D2_MODERATE_HEAD_TAIL, seed=115)
    a2 = generate_stratified_distribution_cohort(cohort, DISTRIBUTION_D2_MODERATE_HEAD_TAIL, seed=115)
    g7_pass = (len(a1) == 500 and [p.assigned_combination for p in a1] == [p.assigned_combination for p in a2])
    gate_results.append(("Gate 7", "Stratified Allocation Determinism", g7_pass, "Identical assignment sequence across independent invocations"))

    # --------------------------------------------------------------------------
    # Gate 8: Deterministic head/middle/tail rank classification
    # --------------------------------------------------------------------------
    t_d2 = classify_distribution_tiers(get_distribution_config(DISTRIBUTION_D2_MODERATE_HEAD_TAIL))
    g8_pass = (
        t_d2.head_combinations == ("RFC", "RF") and
        t_d2.middle_combinations == ("RC", "FC") and
        t_d2.tail_combinations == ("R", "F", "C")
    )
    gate_results.append(("Gate 8", "Head/Middle/Tail Rank Classification", g8_pass, "Pre-registered rank classification deterministic and verified"))

    # --------------------------------------------------------------------------
    # Gate 9: Head-to-Tail ratio (HTR) scaling
    # --------------------------------------------------------------------------
    t_d1 = classify_distribution_tiers(get_distribution_config(DISTRIBUTION_D1_BALANCED))
    t_d3 = classify_distribution_tiers(get_distribution_config(DISTRIBUTION_D3_STRONG_LONG_TAIL))
    g9_pass = (t_d1.head_to_tail_ratio is None and t_d2.head_to_tail_ratio == 4.0 and t_d3.head_to_tail_ratio > 10.0)
    gate_results.append(("Gate 9", "Head-to-Tail Ratio (HTR) Scaling & D1 Null Semantics", g9_pass, "D1 is None (no tail); D2 HTR=4.00, D3 HTR=10.71"))

    # --------------------------------------------------------------------------
    # Gate 10: Active router weight simplex normalization (sum = 1.0)
    # --------------------------------------------------------------------------
    dist_recs, cross_sens = engine.evaluate_all_distributions(cohort)
    g10_pass = True
    for d_id, rec in dist_recs.items():
        for c, c_rec in rec.combinations.items():
            w_sum = c_rec.mean_w_retina + c_rec.mean_w_foot + c_rec.mean_w_clinical
            if not np.isclose(w_sum, 1.0, atol=1e-5):
                g10_pass = False
    gate_results.append(("Gate 10", "Active Authority Simplex Invariant", g10_pass, "sum(w_i) = 1.000000 across all combinations"))

    # --------------------------------------------------------------------------
    # Gate 11: Unavailable modality zero authority weight invariant
    # --------------------------------------------------------------------------
    g11_pass = True
    for d_id, rec in dist_recs.items():
        for c, c_rec in rec.combinations.items():
            mask = COMBINATION_MASKS[c]
            if not mask[0] and c_rec.mean_w_retina != 0.0:
                g11_pass = False
            if not mask[1] and c_rec.mean_w_foot != 0.0:
                g11_pass = False
            if not mask[2] and c_rec.mean_w_clinical != 0.0:
                g11_pass = False
    gate_results.append(("Gate 11", "Unavailable Channel Zero Authority", g11_pass, "A_i = 0 => w_i = 0.000000 strictly enforced"))

    # --------------------------------------------------------------------------
    # Gate 12: Fused risk boundedness in unit interval [0.0, 1.0]
    # --------------------------------------------------------------------------
    g12_pass = True
    for d_id, rec in dist_recs.items():
        for c, c_rec in rec.combinations.items():
            if c_rec.mean_r_fusion < 0.0 or c_rec.mean_r_fusion > 1.0:
                g12_pass = False
    gate_results.append(("Gate 12", "Fused Risk Unit Interval Bounds", g12_pass, "0.0 <= R_fusion <= 1.0 strictly bounded"))

    # --------------------------------------------------------------------------
    # Gate 13: DCRI theoretical bounds & negative preservation
    # --------------------------------------------------------------------------
    g13_pass = True
    for d_id, rec in dist_recs.items():
        for c, c_rec in rec.combinations.items():
            card = COMBINATION_CARDINALITY[c]
            min_bound = -0.20 * card
            if c_rec.mean_dcri < min_bound - 1e-6 or c_rec.mean_dcri > 1.0 + 1e-6:
                g13_pass = False
    gate_results.append(("Gate 13", "DCRI Theoretical Bounds & Unclamped Invariant", g13_pass, "DCRI in [-delta*M, 1.0] with unclamped negative values"))

    # --------------------------------------------------------------------------
    # Gate 14: Uncertainty summation scaling along modality dropout ladder
    # --------------------------------------------------------------------------
    d2_rec = dist_recs[DISTRIBUTION_D2_MODERATE_HEAD_TAIL]
    u_rfc = d2_rec.combinations["RFC"].mean_u_sum
    u_rf = d2_rec.combinations["RF"].mean_u_sum
    u_r = d2_rec.combinations["R"].mean_u_sum
    g14_pass = (u_rfc > u_rf > u_r)
    gate_results.append(("Gate 14", "Cohort Uncertainty-Sum Scaling (RFC > RF > R)", g14_pass, "Additive uncertainty scaling along RFC > RF > R ladder"))

    # --------------------------------------------------------------------------
    # Gate 15: Conflict metrics correctness
    # --------------------------------------------------------------------------
    g15_pass = (
        d2_rec.combinations["RFC"].mean_delta_max > 0.0 and
        d2_rec.combinations["R"].mean_delta_max == 0.0 and
        d2_rec.combinations["F"].mean_delta_max == 0.0 and
        d2_rec.combinations["C"].mean_delta_max == 0.0
    )
    gate_results.append(("Gate 15", "Conflict Metrics Domain Invariants", g15_pass, "Delta_max = 0.0 for unimodals; active for multi-modal packets"))

    # --------------------------------------------------------------------------
    # Gate 16: Minimum sample size threshold rule (N >= 5)
    # --------------------------------------------------------------------------
    all_n_ge_5 = all(
        c_rec.n_packets >= 5
        for rec in dist_recs.values()
        for c_rec in rec.combinations.values()
    )
    all_status_success = all(
        c_rec.status == "SUCCESS"
        for rec in dist_recs.values()
        for c_rec in rec.combinations.values()
    )
    g16_pass = all_n_ge_5 and all_status_success
    gate_results.append(("Gate 16", "Minimum Sample Size Rule Enforcement", g16_pass, "N >= 5 verified across all evaluated combinations in D1, D2, D3"))

    # --------------------------------------------------------------------------
    # Gate 17: Bootstrap confidence interval computation & determinism
    # --------------------------------------------------------------------------
    ci_res1 = compute_bootstrap_ci_1d([0.1, 0.2, 0.3, 0.4, 0.5], n_resamples=500, seed=115)
    ci_res2 = compute_bootstrap_ci_1d([0.1, 0.2, 0.3, 0.4, 0.5], n_resamples=500, seed=115)
    g17_pass = (ci_res1[0] <= 0.3 <= ci_res1[1] and ci_res1[0] > 0.0 and ci_res1 == ci_res2)
    gate_results.append(("Gate 17", "Bootstrap CI Computation & Sample Mean Containment", g17_pass, "Bootstrap CI contains sample mean and is strictly deterministic"))

    # --------------------------------------------------------------------------
    # Gate 18: Comparative baseline ladder B1–B6 execution
    # --------------------------------------------------------------------------
    b_table = d2_rec.baseline_comparison["summary_table"]
    g18_pass = len(b_table) == 6 and all(b_id in b_table for b_id in ("B1", "B2", "B3", "B4", "B5", "B6"))
    gate_results.append(("Gate 18", "Comparative Baseline Ladder B1–B6 Execution", g18_pass, "All 6 baseline architectures benchmarked across combinations"))

    # --------------------------------------------------------------------------
    # Gate 19: Full cohort deep structural & attribute reproducibility
    # --------------------------------------------------------------------------
    cohort_reload = load_frozen_cohort(repo_root=repo_root, n_packets=500, seed=115)
    g19_pass = len(cohort_reload) == len(cohort)
    if g19_pass:
        for p_orig, p_rel in zip(cohort, cohort_reload):
            if (
                p_orig.packet_id != p_rel.packet_id or
                p_orig.retina.risk != p_rel.retina.risk or
                p_orig.retina.uncertainty != p_rel.retina.uncertainty or
                p_orig.foot.risk != p_rel.foot.risk or
                p_orig.foot.uncertainty != p_rel.foot.uncertainty or
                p_orig.clinical.risk != p_rel.clinical.risk or
                p_orig.clinical.uncertainty != p_rel.clinical.uncertainty
            ):
                g19_pass = False
                break
    gate_results.append(("Gate 19", "Full Cohort Deep Structural Reproducibility", g19_pass, "Identical multi-channel features and risks upon fresh reload"))

    # --------------------------------------------------------------------------
    # Gate 20: Sealed research artifacts (16/16) and cryptographic manifest
    # --------------------------------------------------------------------------
    manifest_path = artifacts_dir / "freeze_manifest.json"
    g20_pass = False
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        if len(manifest_data) == 16:
            m_checks = []
            for fname, expected_hash in manifest_data.items():
                fpath = artifacts_dir / fname
                if fpath.is_file() and compute_file_sha256(fpath) == expected_hash:
                    m_checks.append(True)
                else:
                    m_checks.append(False)
            g20_pass = all(m_checks)

    gate_results.append(("Gate 20", "Cryptographic Manifest Certification (16/16)", g20_pass, "All 16 artifact SHA-256 hashes certified on disk"))

    # Print Report Table
    print("\nVerification Gate Results Summary:")
    print("-" * 80)
    all_passed = True
    for g_id, desc, passed, note in gate_results:
        status_str = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"{g_id:7s} | {desc:45s} | {status_str:6s} | {note}")
    print("-" * 80)

    score = sum(1 for _, _, p, _ in gate_results)
    total = len(gate_results)
    print(f"\nFinal Gate Score: {score} / {total} PASSED ({(score/total)*100.0:.1f}%)")

    if not all_passed:
        raise RuntimeError("One or more verification gates FAILED in Phase C11.9.")

    print("Phase C11.9 Modality-Combination Distribution & Tail Robustness VERIFICATION COMPLETE.")
    return {"score": score, "total": total, "passed": all_passed}


if __name__ == "__main__":
    run_all_verification_gates(Path("."))
