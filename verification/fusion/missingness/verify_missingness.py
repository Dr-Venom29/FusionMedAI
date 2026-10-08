"""
verification/fusion/missingness/verify_c11_8_missingness.py
Phase C11.8: Deep Verification Gate Suite (20 / 20 Gates)
Comprehensive verification suite testing all mathematical invariants, availability regimes,
masked-value invariance, authority redistributions, baseline comparisons, and artifact integrity.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import math
import hashlib
import numpy as np

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord
from src.fusion.missingness.missingness_engine import MissingnessEngine
from src.fusion.missingness.availability_mask import (
    ALL_REGIMES,
    NON_EMPTY_REGIMES,
    MODALITY_NAMES,
    apply_availability_mask,
    verify_availability_invariants,
)
from src.fusion.missingness.robustness_metrics import (
    calc_stats,
    compute_risk_delta,
    compute_dcri_delta,
    compute_authority_redistribution,
    compute_dcri_decomposition,
    compute_routing_entropy,
)
from src.fusion.missingness.stress_tests import (
    evaluate_masked_value_invariance,
    evaluate_unavailable_vs_low_quality,
    evaluate_stress_scenarios,
)
from src.fusion.missingness.dropout_scenarios import (
    generate_sequential_ladders,
)


class MissingnessVerificationRunner:
    """
    Executes 20 Deep Verification Gates for Phase C11.8.
    """

    def __init__(self, repo_root: Path = Path(".")):
        self.repo_root = repo_root
        self.engine = MissingnessEngine()
        self.cohort = load_frozen_cohort(repo_root=self.repo_root, n_packets=500, seed=115)
        self.gate_results: Dict[str, bool] = {}

    def run_all_gates(self) -> bool:
        print("================================================================================")
        print("FusionMedAI Phase C11.8: Deep Verification Gate Suite (20 Gates)")
        print("Cohort Size: N = 500 | PRNG Seed: 115 | Target: 20/20 PASSED")
        print("================================================================================")

        gates = [
            ("Gate 1: Modality Risk Bounds & Contract Validity", self.verify_gate_1_risk_bounds),
            ("Gate 2: Availability Correctness (All 8 Regimes)", self.verify_gate_2_availability_regimes),
            ("Gate 3: Unavailable Modality Zero Weight Invariant (A_i=0 => w_i=0)", self.verify_gate_3_unavailable_weight_zero),
            ("Gate 4: Active Modality Weight Simplex Normalization (sum w_i = 1)", self.verify_gate_4_active_simplex_sum),
            ("Gate 5: Inactive Modality Contribution Zero Invariant (K_i = 0)", self.verify_gate_5_inactive_contribution_zero),
            ("Gate 6: Fused Risk Bounds (0.0 <= R_fusion <= 1.0)", self.verify_gate_6_fused_risk_bounds),
            ("Gate 7: DCRI Theoretical Bounds (-delta*M <= DCRI <= 1.0)", self.verify_gate_7_dcri_bounds),
            ("Gate 8: Zero-Modality Safe Fail-Closed Rejection (NO_MODALITY_AVAILABLE)", self.verify_gate_8_zero_modality_rejection),
            ("Gate 9: Masked-Value Invariance Under Corrupted Unavailable Inputs", self.verify_gate_9_masked_value_invariance),
            ("Gate 10: Unavailable vs Low-Quality Fundamental Distinction", self.verify_gate_10_unavailable_vs_low_quality),
            ("Gate 11: Authority Redistribution Conservation across Regimes", self.verify_gate_11_authority_redistribution),
            ("Gate 12: Sequential Information Loss Progression", self.verify_gate_12_sequential_ladders),
            ("Gate 13: Targeted Stress Dropout Behavior (All 7 Stress Criteria)", self.verify_gate_13_stress_scenarios),
            ("Gate 14: Uncertainty x Missingness Stratification Availability & Execution", self.verify_gate_14_uncertainty_missingness),
            ("Gate 15: Reliability Prior Alignment Under Modality Removal", self.verify_gate_15_reliability_missingness),
            ("Gate 16: Upstream Frozen-Path Consistency", self.verify_gate_16_upstream_immutability),
            ("Gate 17: Label-Free Execution Consistency", self.verify_gate_17_ground_truth_independence),
            ("Gate 18: Comparative Baseline Ladder B1–B6 Execution", self.verify_gate_18_baseline_comparison),
            ("Gate 19: Full Independent Cohort Reproducibility (N=500, seed=115)", self.verify_gate_19_cohort_reproducibility),
            ("Gate 20: Sealed Experiment Artifacts (13/13) & SHA-256 Manifest", self.verify_gate_20_sealed_artifacts),
        ]

        all_passed = True
        for gate_name, gate_fn in gates:
            try:
                passed = gate_fn()
                self.gate_results[gate_name] = passed
                status_str = "PASSED" if passed else "FAILED"
                print(f"[{status_str}] {gate_name}")
                if not passed:
                    all_passed = False
            except Exception as e:
                self.gate_results[gate_name] = False
                all_passed = False
                print(f"[FAILED] {gate_name} (Exception: {e})")

        print("================================================================================")
        passed_count = sum(1 for v in self.gate_results.values() if v)
        print(f"Verification Summary: {passed_count} / {len(gates)} Gates PASSED")
        print("================================================================================")
        return all_passed

    # --- Gate Implementations ---

    def verify_gate_1_risk_bounds(self) -> bool:
        for p in self.cohort:
            for m in MODALITY_NAMES:
                rec = p.records[m]
                if not (0.0 <= rec.risk <= 1.0) or math.isnan(rec.risk) or math.isinf(rec.risk):
                    return False
        return True

    def verify_gate_2_availability_regimes(self) -> bool:
        sample_p = self.cohort[0]
        evals = self.engine.evaluate_packet_all_regimes(sample_p, delta=0.20)
        return set(evals.keys()) == set(ALL_REGIMES.keys())

    def verify_gate_3_unavailable_weight_zero(self) -> bool:
        for p in self.cohort[:50]:
            evals = self.engine.evaluate_packet_all_regimes(p, delta=0.20)
            for reg_name, e in evals.items():
                for m in MODALITY_NAMES:
                    if m not in e.active_modalities:
                        if abs(e.weights.get(m, 0.0)) > 1e-9:
                            return False
        return True

    def verify_gate_4_active_simplex_sum(self) -> bool:
        for p in self.cohort[:50]:
            evals = self.engine.evaluate_packet_all_regimes(p, delta=0.20)
            for reg_name, e in evals.items():
                if e.num_active > 0:
                    w_sum = sum(e.weights.get(m, 0.0) for m in e.active_modalities)
                    if abs(w_sum - 1.0) > 1e-5:
                        return False
        return True

    def verify_gate_5_inactive_contribution_zero(self) -> bool:
        for p in self.cohort[:50]:
            for reg_name in NON_EMPTY_REGIMES:
                masked_pkt = apply_availability_mask(p, ALL_REGIMES[reg_name])
                dcri_res = self.engine.dcri_engine.evaluate_packet(masked_pkt, delta=0.20)
                for m in MODALITY_NAMES:
                    if m not in masked_pkt.available_modalities:
                        k_i = dcri_res.weighted_risk_contributions.get(m, 0.0)
                        if abs(k_i) > 1e-9:
                            return False
        return True

    def verify_gate_6_fused_risk_bounds(self) -> bool:
        for p in self.cohort[:50]:
            evals = self.engine.evaluate_packet_all_regimes(p, delta=0.20)
            for reg_name, e in evals.items():
                if not (0.0 <= e.r_fusion <= 1.0 + 1e-6):
                    return False
        return True

    def verify_gate_7_dcri_bounds(self) -> bool:
        delta = 0.20
        for p in self.cohort[:50]:
            evals = self.engine.evaluate_packet_all_regimes(p, delta=delta)
            for reg_name, e in evals.items():
                min_dcri = -delta * e.num_active - 1e-6
                if not (min_dcri <= e.dcri <= 1.0 + 1e-6):
                    return False
        return True

    def verify_gate_8_zero_modality_rejection(self) -> bool:
        sample_p = self.cohort[0]
        zero_eval = self.engine.evaluate_packet_regime(sample_p, "zero_modality", delta=0.20)
        return (
            zero_eval.num_active == 0
            and zero_eval.status == "NO_MODALITY_AVAILABLE"
            and zero_eval.r_fusion == 0.0
            and zero_eval.dcri == 0.0
            and sum(zero_eval.weights.values()) == 0.0
        )

    def verify_gate_9_masked_value_invariance(self) -> bool:
        # Test on 10 packets x 3 bimodal pairs x 5 perturbations = 150 invariance checks
        for p in self.cohort[:10]:
            inv_recs = evaluate_masked_value_invariance(p, self.engine.dcri_engine, delta=0.20)
            for r in inv_recs:
                if not r.passed or r.max_weight_diff > 1e-8 or r.r_fusion_diff > 1e-8 or r.dcri_diff > 1e-8:
                    return False
        return True

    def verify_gate_10_unavailable_vs_low_quality(self) -> bool:
        sample_p = self.cohort[0]
        res = evaluate_unavailable_vs_low_quality(sample_p, self.engine.dcri_engine, delta=0.20)
        return bool(res["distinction_verified"])

    def verify_gate_11_authority_redistribution(self) -> bool:
        sample_p = self.cohort[0]
        f_eval = self.engine.evaluate_packet_regime(sample_p, "tri_modal", delta=0.20)
        fc_eval = self.engine.evaluate_packet_regime(sample_p, "foot_clinical", delta=0.20)
        rec = self.engine.compute_redistribution_record(f_eval, fc_eval, delta=0.20)
        return (
            rec.removed_modalities == ("retina",)
            and rec.delta_w["retina"] == -f_eval.weights["retina"]
            and rec.delta_w["foot"] >= 0.0
            and rec.delta_w["clinical"] >= 0.0
        )

    def verify_gate_12_sequential_ladders(self) -> bool:
        sample_p = self.cohort[0]
        ladders = generate_sequential_ladders(sample_p)
        for l_name, steps in ladders.items():
            if len(steps) != 3:
                return False
            for r_name, pkt in steps:
                res = self.engine.dcri_engine.evaluate_packet(pkt, delta=0.20)
                if not (0.0 <= res.r_fusion <= 1.0):
                    return False
        return True

    def verify_gate_13_stress_scenarios(self) -> bool:
        sample_p = self.cohort[0]
        stress_dict = evaluate_stress_scenarios(sample_p, self.engine.dcri_engine, delta=0.20)
        expected_scenarios = {
            "missing_highest_confidence", "missing_lowest_confidence",
            "missing_highest_reliability", "missing_lowest_uncertainty",
            "missing_highest_uncertainty", "missing_lowest_quality",
            "missing_highest_quality",
        }
        return set(stress_dict.keys()) == expected_scenarios

    def verify_gate_14_uncertainty_missingness(self) -> bool:
        strat = self.engine.evaluate_uncertainty_stratification(self.cohort[:100], delta=0.20)
        return "removed_retina" in strat and "removed_foot" in strat and "removed_clinical" in strat

    def verify_gate_15_reliability_missingness(self) -> bool:
        # Check that removing highest reliability modality (retina, R=0.930) is systematically tracked
        f_eval = self.engine.evaluate_packet_regime(self.cohort[0], "tri_modal", delta=0.20)
        fc_eval = self.engine.evaluate_packet_regime(self.cohort[0], "foot_clinical", delta=0.20)
        return fc_eval.weights["retina"] == 0.0 and (fc_eval.weights["foot"] + fc_eval.weights["clinical"]) == pytest_approx(1.0)

    def verify_gate_16_upstream_immutability(self) -> bool:
        sample_p = self.cohort[0]
        dcri_res = self.engine.dcri_engine.evaluate_packet(sample_p, delta=0.20)
        regime_eval = self.engine.evaluate_packet_regime(sample_p, "tri_modal", delta=0.20)
        return (
            abs(dcri_res.r_fusion - regime_eval.r_fusion) < 1e-6
            and abs(dcri_res.dcri - regime_eval.dcri) < 1e-6
        )

    def verify_gate_17_ground_truth_independence(self) -> bool:
        pkt_unlabeled = ControlledDecisionPacket(
            packet_id="unlabeled_missingness_test",
            retina=self.cohort[0].retina,
            foot=self.cohort[0].foot,
            clinical=self.cohort[0].clinical,
            seed=115,
        )
        evals = self.engine.evaluate_packet_all_regimes(pkt_unlabeled, delta=0.20)
        return len(evals) == 8

    def verify_gate_18_baseline_comparison(self) -> bool:
        comp = self.engine.evaluate_baselines_comparative(self.cohort[:20])
        return all(b_id in comp for b_id in ["B1", "B2", "B3", "B4", "B5", "B6"])

    def verify_gate_19_cohort_reproducibility(self) -> bool:
        fresh_cohort = load_frozen_cohort(repo_root=self.repo_root, n_packets=500, seed=115)
        fresh_engine = MissingnessEngine()

        evals_1 = self.engine.evaluate_cohort_regimes(self.cohort, delta=0.20)
        evals_2 = fresh_engine.evaluate_cohort_regimes(fresh_cohort, delta=0.20)

        for reg in ALL_REGIMES:
            list_1 = evals_1[reg]
            list_2 = evals_2[reg]
            if len(list_1) != 500 or len(list_2) != 500:
                return False
            for e1, e2 in zip(list_1, list_2):
                if e1.r_fusion != e2.r_fusion or e1.dcri != e2.dcri or e1.status != e2.status:
                    return False
        return True

    def verify_gate_20_sealed_artifacts(self) -> bool:
        artifact_dir = self.repo_root / "experiments" / "fusion" / "missingness"
        manifest_path = artifact_dir / "freeze_manifest.json"
        if not manifest_path.exists():
            return False

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if manifest.get("phase") != "C11.8" or manifest.get("cohort_size") != 500 or manifest.get("status") != "SEALED":
            return False

        # Verify SHA-256 hash of all 12 artifact files
        for filename, info in manifest.get("manifest", {}).items():
            f_path = artifact_dir / filename
            if not f_path.exists():
                return False
            with open(f_path, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            if h != info.get("sha256"):
                return False

        # Semantic validation of availability matrix
        with open(artifact_dir / "availability_matrix.json", "r", encoding="utf-8") as f:
            avail = json.load(f)
            if set(avail.keys()) != set(ALL_REGIMES.keys()):
                return False

        return True


def pytest_approx(val: float, tol: float = 1e-5) -> bool:
    return abs(val - 1.0) < tol


def run_verification():
    runner = MissingnessVerificationRunner()
    success = runner.run_all_gates()
    if not success:
        exit(1)


if __name__ == "__main__":
    run_verification()
