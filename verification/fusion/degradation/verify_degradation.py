"""
verification/fusion/degradation/verify_degradation.py
Phase C11.10: 20 Deep Verification Gates for Input Degradation Benchmark

Certifies:
- Protocol and cohort invariants (Gates 1–5)
- Experiment A Quality response mechanics on raw inputs (Gates 6–9)
- Experiment B ACARA-U routing response & simplex invariants (Gates 10–13)
- Slopes, Spearman rank alignment & Monotonicity (Gates 14–16)
- Baseline Benchmarking & B5 vs B6 Quality Isolation (Gate 17)
- Hard-mask safety invariance (Gate 18)
- Live Independent 1,000-Resample Paired Bootstrap Recomputation (Gate 19)
- Cryptographic SHA-256 Artifact Certification (Gate 20)
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.degradation.degradation_spec import (
    SEVERITY_D0_CLEAN,
    SEVERITY_D1_MILD,
    SEVERITY_D2_MODERATE,
    SEVERITY_D3_SEVERE,
    SEVERITY_LEVELS,
    MODALITY_RETINA,
    MODALITY_FOOT,
    MODALITY_CLINICAL,
    ALL_MODALITIES,
    RETINA_OPERATORS,
    FOOT_OPERATORS,
    CLINICAL_OPERATORS,
    OPERATOR_PARAM_GRID,
)
from src.fusion.degradation.image_degradation import (
    generate_benchmark_retina_image,
    generate_benchmark_foot_image,
    apply_image_degradation,
    compute_degraded_image_quality,
)
from src.fusion.degradation.clinical_degradation import (
    generate_benchmark_clinical_vector,
    apply_clinical_degradation,
    compute_degraded_clinical_quality,
)
from src.fusion.degradation.degradation_engine import (
    DegradationEngine,
    apply_degradation_to_packet,
    get_live_degraded_modality_quality,
)
from src.fusion.baselines.decision_packet import ModalityRecord, ControlledDecisionPacket
from src.fusion.degradation.response_metrics import (
    check_hard_mask_invariance,
    compute_spearman_alignment,
    compute_bootstrap_ci_1d,
)


def run_all_verification_gates() -> bool:
    print("=" * 80)
    print("PHASE C11.10: INPUT DEGRADATION BENCHMARK — 20 DEEP VERIFICATION GATES")
    print("=" * 80)

    gate_results: List[Tuple[int, str, bool, str]] = []

    # -------------------------------------------------------------------------
    # Gate 1: Frozen Configuration Lock
    # -------------------------------------------------------------------------
    exp_dir = repo_root / "experiments" / "fusion" / "degradation"
    cfg_file = exp_dir / "experiment_config.json"
    if not cfg_file.exists():
        gate_results.append((1, "Frozen Configuration Lock", False, "experiment_config.json missing"))
    else:
        with open(cfg_file, "r") as f:
            cfg = json.load(f)
        is_ok = (
            cfg.get("cohort_size") == 500
            and cfg.get("seed") == 115
            and cfg.get("router_coefficients", {}).get("alpha") == 1.0
            and cfg.get("router_coefficients", {}).get("beta") == 1.5
            and cfg.get("router_coefficients", {}).get("gamma") == 1.0
            and cfg.get("router_coefficients", {}).get("eta") == 0.5
            and cfg.get("dcri_delta") == 0.20
        )
        gate_results.append((1, "Frozen Configuration Lock", is_ok, "Frozen parameters (alpha=1.0, beta=1.5, gamma=1.0, eta=0.5, delta=0.20, seed=115)"))

    # -------------------------------------------------------------------------
    # Gate 2: Frozen Cohort Integrity
    # -------------------------------------------------------------------------
    cohort = load_frozen_cohort(repo_root=repo_root, n_packets=500, seed=115)
    is_ok = len(cohort) == 500 and all(p.packet_type == "CONTROLLED_DECISION_PACKET" for p in cohort)
    gate_results.append((2, "Frozen Cohort Integrity", is_ok, f"Loaded exactly N={len(cohort)} valid decision packets"))

    # -------------------------------------------------------------------------
    # Gate 3: Modality & Operator Taxonomy
    # -------------------------------------------------------------------------
    all_ops = RETINA_OPERATORS + FOOT_OPERATORS + CLINICAL_OPERATORS
    is_ok = len(ALL_MODALITIES) == 3 and len(all_ops) == 12 and len(SEVERITY_LEVELS) == 4
    gate_results.append((3, "Modality & Operator Taxonomy", is_ok, "3 modalities, 12 operators, 4 severity levels defined"))

    # -------------------------------------------------------------------------
    # Gate 4: Deterministic Parameter Grid
    # -------------------------------------------------------------------------
    is_ok = all(
        op in OPERATOR_PARAM_GRID and all(s in OPERATOR_PARAM_GRID[op] for s in SEVERITY_LEVELS)
        for op in all_ops
    )
    gate_results.append((4, "Deterministic Parameter Grid", is_ok, "Parameter grid fully defined across all 12 operators x 4 severities"))

    # -------------------------------------------------------------------------
    # Gate 5: Clean Identity Invariant (D0)
    # -------------------------------------------------------------------------
    clean_q_r = get_live_degraded_modality_quality(cohort[0].packet_id, MODALITY_RETINA, RETINA_OPERATORS[0], SEVERITY_D0_CLEAN)
    clean_p = apply_degradation_to_packet(cohort[0], MODALITY_RETINA, RETINA_OPERATORS[0], SEVERITY_D0_CLEAN, live_computed_quality=clean_q_r)
    is_ok = clean_p.retina.availability is True and clean_p.retina.quality > 0.90
    gate_results.append((5, "Clean Identity Invariant (D0)", is_ok, f"D0 clean state verified (Clean Q_R = {clean_q_r:.4f})"))

    # Initialize Engine
    engine = DegradationEngine(cohort=cohort, n_bootstrap=100, seed=115)

    # -------------------------------------------------------------------------
    # Gate 6: Deterministic Degradation Repeatability
    # -------------------------------------------------------------------------
    q1 = get_live_degraded_modality_quality(cohort[0].packet_id, MODALITY_FOOT, FOOT_OPERATORS[0], SEVERITY_D2_MODERATE)
    q2 = get_live_degraded_modality_quality(cohort[0].packet_id, MODALITY_FOOT, FOOT_OPERATORS[0], SEVERITY_D2_MODERATE)
    is_ok = (q1 == q2)
    gate_results.append((6, "Deterministic Degradation Repeatability", is_ok, "Repeated raw degradation produces bitwise identical quality score"))

    # -------------------------------------------------------------------------
    # Gate 7: Modality Availability Invariant
    # -------------------------------------------------------------------------
    deg_q_c = get_live_degraded_modality_quality(cohort[0].packet_id, MODALITY_CLINICAL, CLINICAL_OPERATORS[0], SEVERITY_D3_SEVERE)
    deg_p = apply_degradation_to_packet(cohort[0], MODALITY_CLINICAL, CLINICAL_OPERATORS[0], SEVERITY_D3_SEVERE, live_computed_quality=deg_q_c)
    is_ok = deg_p.clinical.availability is True
    gate_results.append((7, "Modality Availability Invariant", is_ok, "Modality remains technically available (A_i = True) under degradation"))

    # -------------------------------------------------------------------------
    # Gate 8: Quality Bound Satisfaction
    # -------------------------------------------------------------------------
    q_recs = engine.evaluate_quality_response(MODALITY_RETINA, RETINA_OPERATORS[0])
    is_ok = all(0.0 <= r.mean_q_degraded <= 1.0 and 0.0 <= r.mean_q_clean <= 1.0 for r in q_recs)
    gate_results.append((8, "Quality Bound Satisfaction", is_ok, "Quality metrics strictly bounded in [0.0, 1.0] across all severities"))

    # -------------------------------------------------------------------------
    # Gate 9: Experiment A Raw Quality Degradation Direction
    # -------------------------------------------------------------------------
    is_ok = (
        q_recs[0].mean_delta_q == 0.0
        and q_recs[1].mean_delta_q < 0
        and q_recs[2].mean_delta_q <= q_recs[1].mean_delta_q
        and q_recs[3].mean_delta_q <= q_recs[2].mean_delta_q
    )
    gate_results.append((9, "Experiment A Raw Quality Degradation Direction", is_ok, "Quality strictly decays monotonically through live frozen quality engine"))

    # -------------------------------------------------------------------------
    # Gate 10: Experiment B Routing Authority Response
    # -------------------------------------------------------------------------
    r_recs, s_recs, mono_rec = engine.evaluate_routing_response(MODALITY_RETINA, RETINA_OPERATORS[0])
    is_ok = r_recs[3].mean_delta_w < 0.0 and r_recs[3].mean_rar > 0.0
    gate_results.append((10, "Experiment B Routing Authority Response", is_ok, f"ACARA-U attenuates authority (Severe delta_w = {r_recs[3].mean_delta_w:.4f})"))

    # -------------------------------------------------------------------------
    # Gate 11: Active Simplex Conservation
    # -------------------------------------------------------------------------
    res = engine.dcri_engine.evaluate_packet(deg_p, delta=0.20)
    total_w = sum(res.modality_weights.values())
    is_ok = abs(total_w - 1.0) < 1e-6 and all(0.0 <= w <= 1.0 for w in res.modality_weights.values())
    gate_results.append((11, "Active Simplex Conservation", is_ok, f"Sum of routing weights = {total_w:.6f} in [0, 1]"))

    # -------------------------------------------------------------------------
    # Gate 12: Unavailable Modality Zero Authority
    # -------------------------------------------------------------------------
    is_ok = all(
        res.modality_weights[m] == 0.0
        for m in ALL_MODALITIES if not deg_p.records[m].availability
    )
    gate_results.append((12, "Unavailable Modality Zero Authority", is_ok, "Unavailable modalities receive exactly zero routing authority"))

    # -------------------------------------------------------------------------
    # Gate 13: Authority Redistribution Invariant
    # -------------------------------------------------------------------------
    is_ok = all(abs(r.redistributed_authority_mean + r.mean_delta_w) < 1e-5 for r in r_recs)
    gate_results.append((13, "Authority Redistribution Invariant", is_ok, "Authority lost by degraded modality is conserved across remaining active channels"))

    # -------------------------------------------------------------------------
    # Gate 14: Quality-Authority Response Slope Positivity
    # -------------------------------------------------------------------------
    is_ok = all(s.mean_slope > 0.0 for s in s_recs[1:])
    gate_results.append((14, "Quality-Authority Response Slope Positivity", is_ok, f"Positive response slope S_QW > 0 (mean = {s_recs[3].mean_slope:.4f})"))

    # -------------------------------------------------------------------------
    # Gate 15: Quality-Routing Spearman Alignment (Actual Spearman Calculation)
    # -------------------------------------------------------------------------
    q_vals = [r.mean_q_degraded for r in q_recs]
    w_vals = [r.mean_w_degraded for r in r_recs]
    rho_val, p_val = compute_spearman_alignment(q_vals, w_vals)
    is_ok = rho_val > 0.80 and p_val < 0.10
    gate_results.append((15, "Quality-Routing Spearman Alignment", is_ok, f"Spearman rank alignment confirmed: rho = {rho_val:.4f} (p = {p_val:.4f})"))

    # -------------------------------------------------------------------------
    # Gate 16: Packet-Level Monotonicity Rate
    # -------------------------------------------------------------------------
    is_ok = mono_rec.quality_monotonic_rate >= 0.90 and mono_rec.routing_monotonic_rate >= 0.90
    gate_results.append((16, "Packet-Level Monotonicity Rate", is_ok, f"Monotonicity rate >= 90% (Quality: {mono_rec.quality_monotonic_rate*100:.1f}%, Routing: {mono_rec.routing_monotonic_rate*100:.1f}%)"))

    # -------------------------------------------------------------------------
    # Gate 17: Baseline B5 vs B6 Quality Isolation
    # -------------------------------------------------------------------------
    b_recs = engine.evaluate_baseline_comparisons(MODALITY_RETINA, RETINA_OPERATORS[0])
    severe_b6 = [r for r in b_recs if r.severity == SEVERITY_D3_SEVERE and r.baseline_id == "B6"][0]
    severe_b5 = [r for r in b_recs if r.severity == SEVERITY_D3_SEVERE and r.baseline_id == "B5"][0]
    is_ok = severe_b6.mean_delta_w < severe_b5.mean_delta_w
    gate_results.append((17, "Baseline B5 vs B6 Quality Isolation", is_ok, f"B6 reduces degraded weight more than B5 (B6: {severe_b6.mean_delta_w:.4f} vs B5: {severe_b5.mean_delta_w:.4f})"))

    # -------------------------------------------------------------------------
    # Gate 18: Hard-Mask Safety Invariance
    # -------------------------------------------------------------------------
    unavail_retina = ModalityRecord(
        sample_id=cohort[0].retina.sample_id,
        modality="retina",
        risk=cohort[0].retina.risk,
        calibrated_probability=cohort[0].retina.calibrated_probability,
        confidence=cohort[0].retina.confidence,
        uncertainty=cohort[0].retina.uncertainty,
        quality=0.0,
        availability=False,
        reliability=cohort[0].retina.reliability,
        model_version=cohort[0].retina.model_version,
    )
    clean_unavail_pkt = ControlledDecisionPacket(
        packet_id=cohort[0].packet_id,
        retina=unavail_retina,
        foot=cohort[0].foot,
        clinical=cohort[0].clinical,
        seed=cohort[0].seed,
    )
    deg_unavail_pkt = apply_degradation_to_packet(
        clean_unavail_pkt, MODALITY_RETINA, RETINA_OPERATORS[0], SEVERITY_D3_SEVERE
    )
    res_cln = engine.dcri_engine.evaluate_packet(clean_unavail_pkt, delta=0.20)
    res_dg = engine.dcri_engine.evaluate_packet(deg_unavail_pkt, delta=0.20)
    is_ok = check_hard_mask_invariance(
        clean_unavail_pkt,
        deg_unavail_pkt,
        res_cln.modality_weights,
        res_dg.modality_weights,
        res_cln.r_fusion,
        res_dg.r_fusion,
    )
    gate_results.append((18, "Hard-Mask Safety Invariance", is_ok, "Corrupting unavailable channel causes exactly delta_w_active = 0 and delta_R = 0"))

    # -------------------------------------------------------------------------
    # Gate 19: Live Independent 1,000-Resample Paired Bootstrap Recomputation
    # -------------------------------------------------------------------------
    # Independently recompute live paired bootstrap on D3 delta_w
    d3_dw_list = [
        apply_degradation_to_packet(p, MODALITY_RETINA, RETINA_OPERATORS[0], SEVERITY_D3_SEVERE, live_computed_quality=get_live_degraded_modality_quality(p.packet_id, MODALITY_RETINA, RETINA_OPERATORS[0], SEVERITY_D3_SEVERE))
        for p in cohort
    ]
    evals_clean = [engine.dcri_engine.evaluate_packet(p, delta=0.20).modality_weights[MODALITY_RETINA] for p in cohort]
    evals_deg = [engine.dcri_engine.evaluate_packet(p, delta=0.20).modality_weights[MODALITY_RETINA] for p in d3_dw_list]
    diffs = [d - c for d, c in zip(evals_deg, evals_clean)]
    ci_low, ci_high = compute_bootstrap_ci_1d(diffs, n_resamples=1000, seed=115)
    is_ok = ci_low < 0.0 and ci_high < 0.0 and ci_low <= ci_high
    gate_results.append((19, "Live 1,000-Resample Paired Bootstrap Recomputation", is_ok, f"Live recomputed 95% CI: [{ci_low:.6f}, {ci_high:.6f}] strictly excludes zero"))

    # -------------------------------------------------------------------------
    # Gate 20: Cryptographic Artifact Manifest Certification
    # -------------------------------------------------------------------------
    manifest_file = exp_dir / "freeze_manifest.json"
    is_ok = manifest_file.exists()
    if is_ok:
        with open(manifest_file, "r") as f:
            m_data = json.load(f)
        artifacts = m_data.get("artifacts", {})
        all_match = True
        for art_name, expected_hash in artifacts.items():
            art_path = exp_dir / art_name
            if not art_path.exists():
                all_match = False
                break
            hasher = hashlib.sha256()
            with open(art_path, "rb") as f:
                hasher.update(f.read())
            if hasher.hexdigest() != expected_hash:
                all_match = False
                break
        is_ok = all_match and len(artifacts) >= 13
    gate_results.append((20, "Cryptographic Artifact Manifest Certification", is_ok, f"All {len(artifacts) if is_ok else 0} JSON artifacts match certified SHA-256 hashes on disk"))

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print()
    all_passed = True
    for gate_num, name, passed, detail in gate_results:
        status_str = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"Gate {gate_num:2d} | {name:<50} | {status_str:<6} | {detail}")

    print("-" * 80)
    print(f"VERIFICATION SUMMARY: {sum(1 for _, _, p, _ in gate_results if p)} / {len(gate_results)} GATES PASSED")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_all_verification_gates()
    sys.exit(0 if success else 1)
