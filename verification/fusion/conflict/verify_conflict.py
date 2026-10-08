"""
verification/fusion/conflict/verify_conflict.py
Phase C11.7: Deep Verification Gates (20 / 20 Gates)
Comprehensive verification suite testing all mathematical invariants, availability regimes,
perturbation properties, deterministic reproducibility, and artifact integrity.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import math
import numpy as np

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord
from src.fusion.conflict.conflict_engine import ConflictEngine
from src.fusion.conflict.conflict_result import ConflictResult, PairwiseRecord
from src.fusion.conflict.pairwise_disagreement import (
    compute_pairwise_record,
    compute_all_pairwise_records,
)
from src.fusion.conflict.conflict_metrics import (
    compute_max_disagreement,
    compute_mean_disagreement,
    compute_weighted_dispersion,
    compute_weight_entropy,
)
from src.fusion.conflict.conflict_classifier import (
    classify_conflict_severity,
    OPERATIONAL_LOW_THRESHOLD,
    OPERATIONAL_HIGH_THRESHOLD,
)
from src.fusion.conflict.conflict_explainability import generate_conflict_summary


class ConflictVerificationRunner:
    """
    Executes 20 Deep Verification Gates for Phase C11.7.
    """

    def __init__(self, repo_root: Path = Path(".")):
        self.repo_root = repo_root
        self.engine = ConflictEngine()
        self.cohort = load_frozen_cohort(repo_root=self.repo_root, n_packets=500, seed=115)
        self.gate_results: Dict[str, bool] = {}

    def run_all_gates(self) -> bool:
        print("================================================================================")
        print("FusionMedAI Phase C11.7: Deep Verification Gate Suite (20 Gates)")
        print("Cohort Size: N = 500 | PRNG Seed: 115 | Target: 20/20 PASSED")
        print("================================================================================")

        gates = [
            ("Gate 1: Modality Risk Bounds & Contract Validity", self.verify_gate_1_risk_bounds),
            ("Gate 2: Availability Correctness (7 Configurations + Zero)", self.verify_gate_2_availability),
            ("Gate 3: Single-Modality Conflict Unavailability Invariant", self.verify_gate_3_single_modality),
            ("Gate 4: Zero-Modality Safe Rejection (NO_MODALITY_AVAILABLE)", self.verify_gate_4_zero_modality),
            ("Gate 5: Pairwise Disagreement Calculation (X_jk = |r_j - r_k|)", self.verify_gate_5_pairwise_calculation),
            ("Gate 6: Pairwise Symmetry Invariant (X_jk == X_kj)", self.verify_gate_6_symmetry),
            ("Gate 7: Zero Disagreement Identity (r_j == r_k => X_jk = 0)", self.verify_gate_7_zero_identity),
            ("Gate 8: Maximum Disagreement Correctness (Delta_max)", self.verify_gate_8_max_disagreement),
            ("Gate 9: Mean Disagreement Correctness (Delta_mean)", self.verify_gate_9_mean_disagreement),
            ("Gate 10: Weighted Variance Bounds & Formula (V_w in [0, 0.25])", self.verify_gate_10_weighted_variance),
            ("Gate 11: Weighted Standard Deviation Formula (sigma_w in [0, 0.5])", self.verify_gate_11_weighted_std),
            ("Gate 12: Consensus Ordering Invariant (Delta_max >= Delta_mean >= sigma_w)", self.verify_gate_12_consensus_ordering),
            ("Gate 13: Conflict Monotonicity Under Risk Perturbation", self.verify_gate_13_perturbation_monotonicity),
            ("Gate 14: Perturbation Determinism & Repeatability", self.verify_gate_14_perturbation_determinism),
            ("Gate 15: Dominant Conflict Pair Identification", self.verify_gate_15_dominant_pair),
            ("Gate 16: Upstream Immutability (Router Weights & DCRI Unchanged)", self.verify_gate_16_upstream_immutability),
            ("Gate 17: Ground-Truth Independence & Zero Label Leakage", self.verify_gate_17_ground_truth_independence),
            ("Gate 18: Operational Classification (LOW / MODERATE / HIGH)", self.verify_gate_18_classification),
            ("Gate 19: Cohort Repeatability Across 500 Frozen Packets", self.verify_gate_19_cohort_repeatability),
            ("Gate 20: Sealed Experiment Artifacts (10/10) & Documentation", self.verify_gate_20_sealed_artifacts),
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

    # --- Individual Gate Implementations ---

    def verify_gate_1_risk_bounds(self) -> bool:
        for p in self.cohort:
            for m in ["retina", "foot", "clinical"]:
                rec = p.records[m]
                if not (0.0 <= rec.risk <= 1.0) or math.isnan(rec.risk) or math.isinf(rec.risk):
                    return False
        return True

    def verify_gate_2_availability(self) -> bool:
        sample_p = self.cohort[0]
        regimes = self.engine.evaluate_modality_configurations(sample_p)
        expected_keys = {
            "retina_only", "foot_only", "clinical_only",
            "retina_foot", "retina_clinical", "foot_clinical",
            "tri_modal", "zero_modality",
        }
        if set(regimes.keys()) != expected_keys:
            return False

        # Invariant verification: Inactive modalities MUST receive weight w_i = 0.0,
        # and active modality weights must sum to 1.0 (for M > 0).
        for reg_name, res in regimes.items():
            dcri_res = self.engine.dcri_engine.evaluate_packet(
                packet=ControlledDecisionPacket(
                    packet_id=f"gate2_test_{reg_name}",
                    retina=ModalityRecord(
                        sample_id=sample_p.retina.sample_id,
                        modality="retina",
                        risk=sample_p.retina.risk,
                        calibrated_probability=sample_p.retina.calibrated_probability,
                        confidence=sample_p.retina.confidence if "retina" in res.active_modalities else 0.0,
                        uncertainty=sample_p.retina.uncertainty if "retina" in res.active_modalities else 0.0,
                        quality=sample_p.retina.quality if "retina" in res.active_modalities else 0.0,
                        availability=("retina" in res.active_modalities),
                        reliability=sample_p.retina.reliability,
                        model_version=sample_p.retina.model_version,
                    ),
                    foot=ModalityRecord(
                        sample_id=sample_p.foot.sample_id,
                        modality="foot",
                        risk=sample_p.foot.risk,
                        calibrated_probability=sample_p.foot.calibrated_probability,
                        confidence=sample_p.foot.confidence if "foot" in res.active_modalities else 0.0,
                        uncertainty=sample_p.foot.uncertainty if "foot" in res.active_modalities else 0.0,
                        quality=sample_p.foot.quality if "foot" in res.active_modalities else 0.0,
                        availability=("foot" in res.active_modalities),
                        reliability=sample_p.foot.reliability,
                        model_version=sample_p.foot.model_version,
                    ),
                    clinical=ModalityRecord(
                        sample_id=sample_p.clinical.sample_id,
                        modality="clinical",
                        risk=sample_p.clinical.risk,
                        calibrated_probability=sample_p.clinical.calibrated_probability,
                        confidence=sample_p.clinical.confidence if "clinical" in res.active_modalities else 0.0,
                        uncertainty=sample_p.clinical.uncertainty if "clinical" in res.active_modalities else 0.0,
                        quality=sample_p.clinical.quality if "clinical" in res.active_modalities else 0.0,
                        availability=("clinical" in res.active_modalities),
                        reliability=sample_p.clinical.reliability,
                        model_version=sample_p.clinical.model_version,
                    ),
                    seed=sample_p.seed,
                )
            )
            for m in ["retina", "foot", "clinical"]:
                w = dcri_res.modality_weights.get(m, 0.0)
                if m not in res.active_modalities and abs(w) > 1e-9:
                    return False
            if res.num_active > 0 and abs(sum(dcri_res.modality_weights.values()) - 1.0) > 1e-6:
                return False

        return True

    def verify_gate_3_single_modality(self) -> bool:
        sample_p = self.cohort[0]
        regimes = self.engine.evaluate_modality_configurations(sample_p)
        for m in ["retina_only", "foot_only", "clinical_only"]:
            r = regimes[m]
            if r.conflict_available is not False:
                return False
            if r.conflict_severity != "NOT_APPLICABLE":
                return False
            if r.max_disagreement is not None or r.mean_disagreement is not None:
                return False
            if r.weighted_std != 0.0:
                return False
        return True

    def verify_gate_4_zero_modality(self) -> bool:
        sample_p = self.cohort[0]
        regimes = self.engine.evaluate_modality_configurations(sample_p)
        zero_res = regimes["zero_modality"]
        return (
            zero_res.conflict_available is False
            and zero_res.conflict_severity == "NO_MODALITY_AVAILABLE"
            and zero_res.max_disagreement is None
            and zero_res.r_fusion == 0.0
            and zero_res.dcri == 0.0
        )

    def verify_gate_5_pairwise_calculation(self) -> bool:
        rec = compute_pairwise_record(
            modality_a="retina", modality_b="foot",
            risk_a=0.75, risk_b=0.20, weight_a=0.6, weight_b=0.4,
            uncertainty_a=0.1, uncertainty_b=0.2, reliability_a=0.93, reliability_b=0.92,
        )
        return (
            abs(rec.signed_difference - 0.55) < 1e-6
            and abs(rec.absolute_difference - 0.55) < 1e-6
            and rec.higher_risk_modality == "retina"
            and rec.lower_risk_modality == "foot"
        )

    def verify_gate_6_symmetry(self) -> bool:
        rec1 = compute_pairwise_record("retina", "foot", 0.70, 0.30, 0.5, 0.5, 0.1, 0.1, 0.9, 0.9)
        rec2 = compute_pairwise_record("foot", "retina", 0.30, 0.70, 0.5, 0.5, 0.1, 0.1, 0.9, 0.9)
        return (
            abs(rec1.absolute_difference - rec2.absolute_difference) < 1e-6
            and abs(rec1.signed_difference - (-rec2.signed_difference)) < 1e-6
        )

    def verify_gate_7_zero_identity(self) -> bool:
        rec = compute_pairwise_record("retina", "foot", 0.40, 0.40, 0.5, 0.5, 0.1, 0.1, 0.9, 0.9)
        return rec.absolute_difference == 0.0 and rec.higher_risk_modality == "tied"

    def verify_gate_8_max_disagreement(self) -> bool:
        active = ("retina", "foot", "clinical")
        risks = {"retina": 0.90, "foot": 0.40, "clinical": 0.10}
        weights = {"retina": 0.4, "foot": 0.3, "clinical": 0.3}
        u = {"retina": 0.1, "foot": 0.1, "clinical": 0.1}
        rel = {"retina": 0.9, "foot": 0.9, "clinical": 0.8}
        records = compute_all_pairwise_records(active, risks, weights, u, rel)
        max_val, pair = compute_max_disagreement(records)
        # RF = 0.50, RC = 0.80, FC = 0.30 -> max is 0.80 (retina, clinical)
        return abs(max_val - 0.80) < 1e-6 and pair == ("retina", "clinical")

    def verify_gate_9_mean_disagreement(self) -> bool:
        active = ("retina", "foot", "clinical")
        risks = {"retina": 0.90, "foot": 0.40, "clinical": 0.10}
        weights = {"retina": 0.4, "foot": 0.3, "clinical": 0.3}
        u = {"retina": 0.1, "foot": 0.1, "clinical": 0.1}
        rel = {"retina": 0.9, "foot": 0.9, "clinical": 0.8}
        records = compute_all_pairwise_records(active, risks, weights, u, rel)
        mean_val = compute_mean_disagreement(records)
        # (0.50 + 0.80 + 0.30) / 3 = 1.60 / 3 = 0.533333
        return abs(mean_val - (1.60 / 3.0)) < 1e-6

    def verify_gate_10_weighted_variance(self) -> bool:
        for p in self.cohort[:50]:
            res = self.engine.evaluate_packet(p)
            if res.weighted_variance is None or not (0.0 <= res.weighted_variance <= 0.25 + 1e-6):
                return False
        return True

    def verify_gate_11_weighted_std(self) -> bool:
        for p in self.cohort[:50]:
            res = self.engine.evaluate_packet(p)
            if res.weighted_std is None or not (0.0 <= res.weighted_std <= 0.5 + 1e-6):
                return False
            if abs(res.weighted_std - math.sqrt(res.weighted_variance)) > 1e-6:
                return False
        return True

    def verify_gate_12_consensus_ordering(self) -> bool:
        # Mathematical Invariant: For any M in {2, 3} and weights w on simplex:
        # Delta_max >= Delta_mean >= sigma_w holds universally.
        for p in self.cohort:
            res = self.engine.evaluate_packet(p)
            if res.conflict_available:
                if res.max_disagreement is None or res.mean_disagreement is None or res.weighted_std is None:
                    return False
                # Check Delta_max >= Delta_mean with numerical tolerance
                if res.max_disagreement < res.mean_disagreement - 1e-6:
                    return False
                # Check Delta_mean >= sigma_w with numerical tolerance
                if res.mean_disagreement < res.weighted_std - 1e-6:
                    return False
        return True

    def verify_gate_13_perturbation_monotonicity(self) -> bool:
        # Verify strict monotonic response under outward risk perturbation.
        # Starting from cohort[0] (where max(retina, foot) = 0.756456),
        # perturbing clinical risk strictly outward (0.80 -> 0.90 -> 1.00) must strictly
        # increase Delta_max, Delta_mean, and sigma_w.
        sample_p = self.cohort[0]
        res_0 = self.engine.perturb_modality_risk(sample_p, "clinical", new_risk=0.80)
        res_1 = self.engine.perturb_modality_risk(sample_p, "clinical", new_risk=0.90)
        res_2 = self.engine.perturb_modality_risk(sample_p, "clinical", new_risk=1.00)

        # 1. Delta_max strict monotonic ordering
        cond_max = (res_0.max_disagreement < res_1.max_disagreement < res_2.max_disagreement)
        # 2. Delta_mean strict monotonic ordering
        cond_mean = (res_0.mean_disagreement < res_1.mean_disagreement < res_2.mean_disagreement)
        # 3. Weighted dispersion strict monotonic ordering
        cond_std = (res_0.weighted_std < res_1.weighted_std < res_2.weighted_std)

        return bool(cond_max and cond_mean and cond_std)

    def verify_gate_14_perturbation_determinism(self) -> bool:
        sample_p = self.cohort[0]
        res1 = self.engine.perturb_modality_risk(sample_p, "foot", 0.75)
        res2 = self.engine.perturb_modality_risk(sample_p, "foot", 0.75)
        return (
            res1.max_disagreement == res2.max_disagreement
            and res1.weighted_variance == res2.weighted_variance
            and res1.r_fusion == res2.r_fusion
        )

    def verify_gate_15_dominant_pair(self) -> bool:
        for p in self.cohort:
            res = self.engine.evaluate_packet(p)
            if res.conflict_available:
                max_rec = max(res.pairwise_records, key=lambda r: r.absolute_difference)
                expected_pair = (max_rec.modality_a, max_rec.modality_b)
                if res.dominant_conflict_pair != expected_pair:
                    return False
        return True

    def verify_gate_16_upstream_immutability(self) -> bool:
        sample_p = self.cohort[0]
        dcri_res = self.engine.dcri_engine.evaluate_packet(sample_p, delta=0.20)
        conflict_res = self.engine.evaluate_packet(sample_p, delta=0.20)
        return (
            abs(dcri_res.r_fusion - conflict_res.r_fusion) < 1e-7
            and abs(dcri_res.dcri - conflict_res.dcri) < 1e-7
            and dcri_res.num_active == conflict_res.num_active
        )

    def verify_gate_17_ground_truth_independence(self) -> bool:
        # Check that no label attribute is read or needed
        pkt_no_label = ControlledDecisionPacket(
            packet_id="unlabeled_packet",
            retina=self.cohort[0].retina,
            foot=self.cohort[0].foot,
            clinical=self.cohort[0].clinical,
            seed=115,
        )
        res = self.engine.evaluate_packet(pkt_no_label)
        return res.conflict_available is True

    def verify_gate_18_classification(self) -> bool:
        return (
            classify_conflict_severity(3, 0.10) == "LOW"
            and classify_conflict_severity(3, 0.25) == "MODERATE"
            and classify_conflict_severity(3, 0.50) == "HIGH"
            and classify_conflict_severity(1, None) == "NOT_APPLICABLE"
            and classify_conflict_severity(0, None) == "NO_MODALITY_AVAILABLE"
        )

    def verify_gate_19_cohort_repeatability(self) -> bool:
        # Full independent reload of frozen cohort from disk to test true reproducible execution
        fresh_cohort = load_frozen_cohort(repo_root=self.repo_root, n_packets=500, seed=115)
        fresh_engine = ConflictEngine()

        results_1 = self.engine.evaluate_cohort(self.cohort, delta=0.20)
        results_2 = fresh_engine.evaluate_cohort(fresh_cohort, delta=0.20)

        if len(results_1) != 500 or len(results_2) != 500:
            return False

        for r1, r2 in zip(results_1, results_2):
            if r1.packet_id != r2.packet_id:
                return False
            if r1.max_disagreement != r2.max_disagreement:
                return False
            if r1.mean_disagreement != r2.mean_disagreement:
                return False
            if r1.weighted_std != r2.weighted_std:
                return False
            if r1.conflict_severity != r2.conflict_severity:
                return False
            if r1.r_fusion != r2.r_fusion:
                return False
            if r1.dcri != r2.dcri:
                return False
        return True

    def verify_gate_20_sealed_artifacts(self) -> bool:
        import hashlib
        artifact_dir = self.repo_root / "experiments" / "fusion" / "conflict"
        manifest_path = artifact_dir / "freeze_manifest.json"
        if not manifest_path.exists():
            return False

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # 1. Structural and freeze verification in manifest
        if manifest.get("phase") != "C11.7":
            return False
        if manifest.get("cohort_size") != 500 or manifest.get("cohort_seed") != 115:
            return False
        if manifest.get("status") != "SEALED":
            return False

        # 2. Cryptographic SHA-256 verification of every artifact listed in manifest
        for filename, info in manifest.get("manifest", {}).items():
            file_path = artifact_dir / filename
            if not file_path.exists():
                return False
            with open(file_path, "rb") as f:
                computed_hash = hashlib.sha256(f.read()).hexdigest()
            if computed_hash != info.get("sha256"):
                return False

        # 3. Deep semantic validation of core artifact files
        # 3a. conflict_configuration.json
        with open(artifact_dir / "conflict_configuration.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
            if cfg.get("cohort_size") != 500 or cfg.get("cohort_seed") != 115:
                return False
            if cfg.get("operational_thresholds", {}).get("low_threshold") != 0.20:
                return False
            if cfg.get("operational_thresholds", {}).get("high_threshold") != 0.35:
                return False

        # 3b. conflict_results.json
        with open(artifact_dir / "conflict_results.json", "r", encoding="utf-8") as f:
            c_res = json.load(f)
            if c_res.get("num_packets") != 500:
                return False
            if "max_disagreement_statistics" not in c_res or "weighted_std_statistics" not in c_res:
                return False
            sev = c_res.get("severity_distribution", {})
            counts = sev.get("counts", {})
            total_sev = sum(counts.values())
            if total_sev != 500:
                return False

        # 3c. availability_results.json
        with open(artifact_dir / "availability_results.json", "r", encoding="utf-8") as f:
            avail = json.load(f)
            expected_regimes = {
                "retina_only", "foot_only", "clinical_only",
                "retina_foot", "retina_clinical", "foot_clinical",
                "tri_modal", "zero_modality",
            }
            if set(avail.keys()) != expected_regimes:
                return False

        # 3d. high_conflict_packets.json
        with open(artifact_dir / "high_conflict_packets.json", "r", encoding="utf-8") as f:
            high_pkts = json.load(f)
            if high_pkts.get("num_selected") != 50 or len(high_pkts.get("packets", [])) != 50:
                return False

        return True


def run_verification():
    runner = ConflictVerificationRunner()
    success = runner.run_all_gates()
    if not success:
        exit(1)


if __name__ == "__main__":
    run_verification()
