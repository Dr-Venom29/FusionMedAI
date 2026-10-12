"""
Phase C11.17 Independent Verification Suite: Final Decision-Level Evaluation.

Verifies acceptance gates C17-01 through C17-16:
- C17-01: Artifact Manifest & Checksum Integrity (All runner evaluation artifacts + upstream packet manifest)
- C17-02: Reference B6 Reproduction Invariant (recomputed directly from individual packet records)
- C17-03: Simplex Preservation Across All Evaluated Policies (recomputed from 3,000 packet records)
- C17-04: Availability Mask Enforcement (no unauthorized weight leakage across all 8 regimes)
- C17-05: Fail-Closed EMPTY Availability Handling (zero weights, zero risk, zero DCRI, correct status)
- C17-06: Unimodal Singleton Degeneracy Invariance (router weights locked to unity for singletons)
- C17-07: Unweighted DCRI Penalty Orthogonality & Exact Form (P_U = delta * sum U_i)
- C17-08: Negative DCRI Preservation (exactly 37 negative records unclamped, preserving transparency)
- C17-09: Decision Action Tier Conservation (Tiers 0, 1, 2 partitions strictly sum to N=500)
- C17-10: Baseline Transition Conservation (sum of 3x3 transition matrix equals N=500, reclassification conservation)
- C17-11: Quality Bonus Directionality & Specificity (B6 vs B5 shifts)
- C17-12: Input Degradation Monotonicity / Response (D0 through D3 evaluation)
- C17-13: Calibration Condition Integrity (Calibrated vs Uncalibrated response)
- C17-14: Continuous Discordance & Conflict Stratification (CSS, Delta_max, sigma_w)
- C17-15: Paired Bootstrap Reproducibility (B=1,000, seed=7117, percentile CI checks)
- C17-16: Research Volume Structural Presence & Population Audit (Volume 17 Chapters 01–13 + README)
"""

import sys
import os
import json
import math
import hashlib
import csv
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional, Sequence
import numpy as np

# Portably resolve repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fusion.decision_evaluation.evaluation_metrics import (
    compute_conflict_metrics,
    compute_decision_tier_transitions,
)

EXP_DIR = REPO_ROOT / "experiments" / "fusion" / "decision_evaluation"
DOCS_DIR = REPO_ROOT / "research" / "fusion" / "Volume_17_Decision_Level_Evaluation"

RUNNER_REQUIRED_ARTIFACTS = [
    "02_evaluation_config.json",
    "03_packet_level_results.csv",
    "03_canonical_records.json",
    "04_summary_metrics.json",
    "05_decision_transitions.csv",
    "06_statistical_analysis.json",
    "07_diagnostics.json",
    "08_execution_summary.json",
]

UPSTREAM_PACKET_MANIFEST = REPO_ROOT / "experiments" / "fusion" / "dcri" / "packet_manifest.json"
UPSTREAM_PACKET_MANIFEST_SIZE_BYTES = 821319
UPSTREAM_PACKET_MANIFEST_SHA256 = "1ff37fbeeba33feae7bf14d57d0df4a10df48ac789edeedf9dbd2fa8121382bb"

REQUIRED_DOCS = [
    "README.md",
    "01_Decision_Evaluation_Protocol.md",
    "02_Baseline_Comparative_Analysis.md",
    "03_Modality_Availability_and_Missingness.md",
    "04_Input_Degradation_Robustness.md",
    "05_Cross_Modality_Conflict_Stratification.md",
    "06_Calibration_Condition_Impact.md",
    "07_Epistemic_Uncertainty_and_DCRI_Behavior.md",
    "08_Decision_Action_Tier_Transitions.md",
    "09_Statistical_Synthesis_and_Bootstrap_Inference.md",
    "10_Safety_Invariants_and_Degeneracy_Analysis.md",
    "11_Diagnostic_Failure_Analysis.md",
    "12_Comprehensive_Results_Summary.md",
    "13_Final_Research_Freeze_Report.md",
]

MODALITIES = ["retina", "foot", "clinical"]
REGIME_MODALITY_MAP = {
    "RFC": ["retina", "foot", "clinical"],
    "RF": ["retina", "foot"],
    "RC": ["retina", "clinical"],
    "FC": ["foot", "clinical"],
    "R": ["retina"],
    "F": ["foot"],
    "C": ["clinical"],
    "EMPTY": [],
}


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class GateFailure(Exception):
    pass


class DecisionEvaluationVerifier:
    def __init__(self, repo_root: Path = REPO_ROOT):
        self.repo_root = repo_root
        self.exp_dir = repo_root / "experiments" / "fusion" / "decision_evaluation"
        self.docs_dir = repo_root / "research" / "fusion" / "Volume_17_Decision_Level_Evaluation"
        self.gate_results: Dict[str, Dict[str, Any]] = {}
        self.canonical_data: Optional[Dict[str, Any]] = None

    def log_gate(self, gate_id: str, passed: bool, message: str, details: Optional[Dict[str, Any]] = None):
        self.gate_results[gate_id] = {
            "passed": passed,
            "message": message,
            "details": details or {},
        }
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {gate_id}: {message}")

    def load_canonical_records(self) -> Dict[str, Any]:
        if self.canonical_data is None:
            canonical_path = self.exp_dir / "03_canonical_records.json"
            if not canonical_path.is_file():
                raise FileNotFoundError(f"Canonical records file missing: {canonical_path}")
            with open(canonical_path, "r", encoding="utf-8") as f:
                self.canonical_data = json.load(f)
        return self.canonical_data

    @staticmethod
    def assert_finite_number(value: Any, field_name: str, packet_id: str = "") -> float:
        """
        Validates that a numeric value is present, not None, convertible to float,
        and strictly finite (not NaN, +inf, or -inf). Raises GateFailure if invalid.
        """
        if value is None:
            raise GateFailure(f"Packet {packet_id}: field '{field_name}' is None")
        try:
            val_f = float(value)
        except (ValueError, TypeError) as exc:
            raise GateFailure(f"Packet {packet_id}: field '{field_name}' cannot be converted to float ({value}): {exc}")
        if not math.isfinite(val_f):
            raise GateFailure(f"Packet {packet_id}: field '{field_name}' is non-finite ({val_f})")
        return val_f

    @classmethod
    def assert_finite_fields(cls, record: Dict[str, Any], fields: Sequence[str], packet_id: str = "") -> Dict[str, float]:
        """
        Validates that multiple fields in record are present and strictly finite floats.
        Returns a dictionary mapping field name to extracted float.
        """
        pid = packet_id or str(record.get("packet_id", "UNKNOWN"))
        res = {}
        for f in fields:
            if f not in record:
                raise GateFailure(f"Packet {pid}: missing required field '{f}'")
            res[f] = cls.assert_finite_number(record[f], f, pid)
        return res

    def verify_all_gates(self) -> bool:
        print("\n================================================================================")
        print("Starting Verification Suite for Phase C11.17 Final Decision-Level Evaluation")
        print("================================================================================\n")

        gates = [
            self.verify_c17_01_artifact_integrity,
            self.verify_c17_02_reference_reproduction,
            self.verify_c17_03_simplex_preservation,
            self.verify_c17_04_mask_enforcement,
            self.verify_c17_05_fail_closed_empty,
            self.verify_c17_06_unimodal_degeneracy,
            self.verify_c17_07_unweighted_dcri_penalty,
            self.verify_c17_08_negative_dcri_preservation,
            self.verify_c17_09_decision_tier_conservation,
            self.verify_c17_10_transition_conservation,
            self.verify_c17_11_quality_bonus_directionality,
            self.verify_c17_12_degradation_monotonicity,
            self.verify_c17_13_calibration_condition,
            self.verify_c17_14_conflict_stratification,
            self.verify_c17_15_bootstrap_reproducibility,
            self.verify_c17_16_documentation_presence,
        ]

        all_passed = True
        for gate in gates:
            try:
                gate()
            except GateFailure:
                all_passed = False
            except Exception as e:
                import traceback
                traceback.print_exc()
                all_passed = False

        passed_count = sum(1 for g in self.gate_results.values() if g["passed"])
        total_count = len(gates)
        print("\n================================================================================")
        print(f"Verification Complete: {passed_count}/{total_count} Gates Passed")
        print("================================================================================\n")

        # Emit independent verification report
        self.emit_verification_report(all_passed)

        if all_passed:
            self.emit_freeze_manifest()

        return all_passed

    def emit_verification_report(self, all_passed: bool) -> None:
        report_path = self.exp_dir / "08_verification_report.json"
        b6_reproduction = self.gate_results.get("Gate C17-02: Reference B6 Reproduction Invariant", {}).get("details", {})
        report = {
            "evaluation_phase": "C11.17",
            "verdict": "ACCEPTED" if all_passed else "REJECTED",
            "verification_timestamp": datetime.now(timezone.utc).isoformat(),
            "gates_evaluated": len(self.gate_results),
            "gates_passed": sum(1 for g in self.gate_results.values() if g["passed"]),
            "gate_details": self.gate_results,
            "reference_reproduction": b6_reproduction,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)
        print(f"  Verification report written to {report_path} (verdict: {report['verdict']}).")

    def emit_freeze_manifest(self) -> None:
        freeze_manifest_path = self.exp_dir / "freeze_manifest.json"
        sealed_files = [
            "02_evaluation_config.json",
            "03_packet_level_results.csv",
            "03_canonical_records.json",
            "04_summary_metrics.json",
            "05_decision_transitions.csv",
            "06_statistical_analysis.json",
            "07_diagnostics.json",
            "08_execution_summary.json",
            "08_verification_report.json",
            "09_artifact_manifest.json",
        ]
        file_hashes = {}
        for fname in sealed_files:
            fpath = self.exp_dir / fname
            if fpath.is_file():
                file_hashes[fname] = {
                    "sha256": compute_sha256(fpath),
                    "size_bytes": fpath.stat().st_size,
                }
            else:
                raise FileNotFoundError(f"Missing file during freeze manifest sealing: {fname}")

        # Upstream manifest entry
        if UPSTREAM_PACKET_MANIFEST.is_file():
            file_hashes["upstream_packet_manifest.json"] = {
                "sha256": compute_sha256(UPSTREAM_PACKET_MANIFEST),
                "size_bytes": UPSTREAM_PACKET_MANIFEST.stat().st_size,
            }

        doc_hashes = {}
        for doc in REQUIRED_DOCS:
            dpath = self.docs_dir / doc
            if dpath.is_file():
                doc_hashes[doc] = {
                    "sha256": compute_sha256(dpath),
                    "size_bytes": dpath.stat().st_size,
                }
            else:
                raise FileNotFoundError(f"Missing research document during freeze manifest sealing: {doc}")

        freeze_manifest = {
            "evaluation_phase": "C11.17",
            "freeze_status": "SEALED",
            "freeze_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_sealed_artifacts": len(file_hashes),
            "total_sealed_documents": len(doc_hashes),
            "artifacts": file_hashes,
            "research_documents": doc_hashes,
        }
        with open(freeze_manifest_path, "w", encoding="utf-8") as f:
            json.dump(freeze_manifest, f, indent=2, allow_nan=False)
        print(f"  Final freeze manifest sealed at {freeze_manifest_path} ({len(file_hashes)} artifacts, {len(doc_hashes)} documents).")

    # -------------------------------------------------------------------------
    # Gate C17-01: Artifact Manifest & Checksum Integrity
    # -------------------------------------------------------------------------
    def verify_c17_01_artifact_integrity(self):
        gate_id = "Gate C17-01: Artifact Manifest & Checksum Integrity"
        manifest_path = self.exp_dir / "09_artifact_manifest.json"
        if not manifest_path.is_file():
            self.log_gate(gate_id, False, "Manifest file 09_artifact_manifest.json missing")
            raise GateFailure()

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        missing = []
        checksum_mismatch = []
        size_mismatch = []

        manifest_artifacts = manifest.get("artifacts", {})

        for fname in RUNNER_REQUIRED_ARTIFACTS:
            fpath = self.exp_dir / fname
            if not fpath.is_file():
                missing.append(fname)
                continue
            if fname not in manifest_artifacts:
                missing.append(f"{fname} (not in manifest)")
                continue

            expected_sha = manifest_artifacts[fname]["sha256"]
            expected_size = manifest_artifacts[fname]["size_bytes"]
            actual_sha = compute_sha256(fpath)
            actual_size = fpath.stat().st_size

            if expected_sha != actual_sha:
                checksum_mismatch.append((fname, expected_sha, actual_sha))
            if expected_size != actual_size:
                size_mismatch.append((fname, expected_size, actual_size))

        # Check upstream packet manifest
        if not UPSTREAM_PACKET_MANIFEST.is_file():
            missing.append("upstream_packet_manifest.json")
        else:
            up_size = UPSTREAM_PACKET_MANIFEST.stat().st_size
            up_sha = compute_sha256(UPSTREAM_PACKET_MANIFEST)
            if up_size != UPSTREAM_PACKET_MANIFEST_SIZE_BYTES:
                size_mismatch.append(("upstream_packet_manifest.json", UPSTREAM_PACKET_MANIFEST_SIZE_BYTES, up_size))
            if up_sha != UPSTREAM_PACKET_MANIFEST_SHA256:
                checksum_mismatch.append(("upstream_packet_manifest.json", UPSTREAM_PACKET_MANIFEST_SHA256, up_sha))

        if missing or checksum_mismatch or size_mismatch:
            self.log_gate(
                gate_id,
                False,
                f"Missing: {missing}, Checksum mismatches: {checksum_mismatch}, Size mismatches: {size_mismatch}",
            )
            raise GateFailure()

        self.log_gate(
            gate_id,
            True,
            f"All {len(RUNNER_REQUIRED_ARTIFACTS)} runner artifacts and upstream manifest bitwise verified",
            {"artifacts_verified": len(RUNNER_REQUIRED_ARTIFACTS) + 1},
        )

    # -------------------------------------------------------------------------
    # Gate C17-02: Reference B6 Reproduction Invariant
    # -------------------------------------------------------------------------
    def verify_c17_02_reference_reproduction(self):
        gate_id = "Gate C17-02: Reference B6 Reproduction Invariant"
        canonical = self.load_canonical_records()
        b6_records = canonical.get("baselines", {}).get("B6", [])

        if len(b6_records) != 500:
            self.log_gate(gate_id, False, f"Expected 500 B6 records, found {len(b6_records)}")
            raise GateFailure()

        # Check unique IDs and finite status
        packet_ids = set()
        for r in b6_records:
            pid = r.get("packet_id")
            if pid in packet_ids:
                self.log_gate(gate_id, False, f"Duplicate packet ID detected in B6 records: {pid}")
                raise GateFailure()
            packet_ids.add(pid)
            if r.get("status") != "SUCCESS" or not r.get("is_finite", False):
                self.log_gate(gate_id, False, f"Non-successful or non-finite record: {pid}")
                raise GateFailure()

        mean_rfusion = float(np.mean([float(r["r_fusion"]) for r in b6_records]))
        mean_dcri = float(np.mean([float(r["dcri"]) for r in b6_records]))
        mean_penalty = float(np.mean([float(r["uncertainty_penalty"]) for r in b6_records]))
        neg_count = sum(1 for r in b6_records if float(r["dcri"]) < 0)

        expected_rfusion = 0.28989984657769124
        expected_dcri = 0.22650622777769125
        expected_penalty = 0.0633936188
        expected_negatives = 37

        if (
            abs(mean_rfusion - expected_rfusion) > 1e-6
            or abs(mean_dcri - expected_dcri) > 1e-6
            or abs(mean_penalty - expected_penalty) > 1e-6
            or neg_count != expected_negatives
        ):
            self.log_gate(
                gate_id,
                False,
                f"Mismatch: R={mean_rfusion:.8f} vs {expected_rfusion:.8f}, DCRI={mean_dcri:.8f} vs {expected_dcri:.8f}, Penalty={mean_penalty:.8f} vs {expected_penalty:.8f}, Neg={neg_count} vs {expected_negatives}",
            )
            raise GateFailure()

        self.log_gate(
            gate_id,
            True,
            f"B6 reference matched within predefined tolerance (1e-6): R={mean_rfusion:.6f}, DCRI={mean_dcri:.6f}, P_U={mean_penalty:.6f}, Neg={neg_count}",
            {
                "mean_rfusion": mean_rfusion,
                "mean_dcri": mean_dcri,
                "mean_penalty": mean_penalty,
                "negative_count": neg_count,
            },
        )

    # -------------------------------------------------------------------------
    # Gate C17-03: Simplex Preservation Across All Evaluated Policies
    # -------------------------------------------------------------------------
    def verify_c17_03_simplex_preservation(self):
        gate_id = "Gate C17-03: Simplex Preservation Across All Evaluated Policies"
        canonical = self.load_canonical_records()
        baselines = canonical.get("baselines", {})

        violations = 0
        total_records = 0

        for pol_id, records in baselines.items():
            for r in records:
                total_records += 1
                pid = str(r.get("packet_id", "UNKNOWN"))
                w = r.get("weights", {})
                w_vals = [
                    self.assert_finite_number(w.get(m), f"weights.{m}", pid)
                    for m in MODALITIES
                ]
                w_sum = sum(w_vals)

                if abs(w_sum - 1.0) > 1e-5:
                    violations += 1
                for val in w_vals:
                    if val < -1e-6 or val > 1.0 + 1e-6:
                        violations += 1
                for m in MODALITIES:
                    if m not in r.get("active_modalities", []) and w_vals[MODALITIES.index(m)] > 1e-6:
                        violations += 1

        if violations > 0 or total_records != 3000:
            self.log_gate(gate_id, False, f"Found {violations} simplex violations out of {total_records} records")
            raise GateFailure()

        self.log_gate(gate_id, True, f"Simplex preserved exactly across all {total_records} evaluated policy records (0 violations)")

    # -------------------------------------------------------------------------
    # Gate C17-04: Availability Mask Enforcement Across All Regimes
    # -------------------------------------------------------------------------
    def verify_c17_04_mask_enforcement(self):
        gate_id = "Gate C17-04: Availability Mask Enforcement"
        canonical = self.load_canonical_records()
        regimes_b6 = canonical.get("regimes_b6", {})
        baseline_regimes = canonical.get("baseline_regimes", {})

        expected_regimes = {"RFC", "RF", "RC", "FC", "R", "F", "C", "EMPTY"}
        expected_baseline_policies = {"B1", "B2", "B5"}

        if set(regimes_b6.keys()) != expected_regimes:
            self.log_gate(gate_id, False, f"B6 regime set mismatch: {set(regimes_b6.keys()) ^ expected_regimes}")
            raise GateFailure()

        if set(baseline_regimes.keys()) != expected_baseline_policies:
            self.log_gate(gate_id, False, f"Baseline regime policy set mismatch: {set(baseline_regimes.keys()) ^ expected_baseline_policies}")
            raise GateFailure()

        for pol_id, reg_map in baseline_regimes.items():
            if set(reg_map.keys()) != expected_regimes:
                self.log_gate(gate_id, False, f"Regime set mismatch for {pol_id}: {set(reg_map.keys()) ^ expected_regimes}")
                raise GateFailure()

        reference_b6 = canonical.get("baselines", {}).get("B6", [])
        expected_cohort_ids = {r["packet_id"] for r in reference_b6}
        if len(expected_cohort_ids) != 500:
            self.log_gate(gate_id, False, f"Canonical reference cohort packet count {len(expected_cohort_ids)} != 500")
            raise GateFailure()

        checked_records = 0
        all_regime_collections = [("B6", regimes_b6)]
        for pol_id in sorted(baseline_regimes.keys()):
            all_regime_collections.append((pol_id, baseline_regimes[pol_id]))

        for pol_name, reg_map in all_regime_collections:
            for reg_id in sorted(expected_regimes):
                records = reg_map[reg_id]
                if len(records) != 500:
                    self.log_gate(gate_id, False, f"Regime {reg_id} under {pol_name} has {len(records)} records != 500")
                    raise GateFailure()

                expected_active = set(REGIME_MODALITY_MAP.get(reg_id, []))
                pids = set()

                for r in records:
                    checked_records += 1
                    pid = r.get("packet_id")
                    if not pid or pid in pids:
                        self.log_gate(gate_id, False, f"Duplicate or missing packet {pid} in regime {reg_id} ({pol_name})")
                        raise GateFailure()
                    pids.add(pid)

                    w = r.get("weights", {})
                    # Explicit verifier-side finite number checks
                    w_vals = {
                        m: self.assert_finite_number(w.get(m), f"weights.{m}", pid)
                        for m in MODALITIES
                    }

                    if reg_id == "EMPTY":
                        if r.get("status") != "NO_MODALITY_AVAILABLE" or r.get("num_active") != 0 or r.get("active_modalities") != []:
                            self.log_gate(gate_id, False, f"EMPTY regime metadata violation in {pid} ({pol_name})")
                            raise GateFailure()
                        for m in MODALITIES:
                            if w_vals[m] != 0.0:
                                self.log_gate(gate_id, False, f"Non-zero weight under EMPTY in {pid} ({pol_name})")
                                raise GateFailure()
                    else:
                        if r.get("status") != "SUCCESS":
                            self.log_gate(gate_id, False, f"Non-success status in {pid} under {reg_id} ({pol_name})")
                            raise GateFailure()
                        if r.get("num_active") != len(expected_active):
                            self.log_gate(gate_id, False, f"num_active mismatch in {pid} under {reg_id} ({pol_name})")
                            raise GateFailure()
                        if set(r.get("active_modalities", [])) != expected_active:
                            self.log_gate(gate_id, False, f"active_modalities mismatch in {pid} under {reg_id} ({pol_name})")
                            raise GateFailure()

                        # Finite check on decision metrics
                        self.assert_finite_fields(r, ["r_fusion", "dcri", "uncertainty_penalty"], pid)

                        w_sum = sum(w_vals[m] for m in expected_active)
                        if abs(w_sum - 1.0) > 1e-5:
                            self.log_gate(gate_id, False, f"Simplex violation in {pid} under {reg_id} ({pol_name}): sum={w_sum}")
                            raise GateFailure()

                        for m in MODALITIES:
                            if m not in expected_active and w_vals[m] > 1e-7:
                                self.log_gate(gate_id, False, f"Mask leak violation in {pid} under {reg_id} ({pol_name}): {m}={w_vals[m]}")
                                raise GateFailure()

                # Enforce identical cohort composition across all regimes
                if pids != expected_cohort_ids:
                    diff_missing = expected_cohort_ids - pids
                    diff_extra = pids - expected_cohort_ids
                    self.log_gate(
                        gate_id,
                        False,
                        f"Regime {reg_id} under {pol_name} cohort mismatch: missing {len(diff_missing)}, extra {len(diff_extra)}",
                    )
                    raise GateFailure()

        if checked_records != 16000:
            self.log_gate(gate_id, False, f"Total checked regime records {checked_records} != 16,000 (expected 4 policies * 8 regimes * 500)")
            raise GateFailure()

        self.log_gate(gate_id, True, f"Mask enforcement verified with 500 records per regime and zero leakage across all {checked_records} regime records")

    # -------------------------------------------------------------------------
    # Gate C17-05: Fail-Closed EMPTY Availability Handling
    # -------------------------------------------------------------------------
    def verify_c17_05_fail_closed_empty(self):
        gate_id = "Gate C17-05: Fail-Closed EMPTY Availability Handling"
        canonical = self.load_canonical_records()
        empty_cohorts = [("B6", canonical.get("regimes_b6", {}).get("EMPTY", []))]
        for pol_id, reg_map in canonical.get("baseline_regimes", {}).items():
            empty_cohorts.append((pol_id, reg_map.get("EMPTY", [])))

        total_empty_checked = 0
        for pol_name, empty_records in empty_cohorts:
            if len(empty_records) != 500:
                self.log_gate(gate_id, False, f"Expected 500 EMPTY records for {pol_name}, found {len(empty_records)}")
                raise GateFailure()

            for r in empty_records:
                total_empty_checked += 1
                pid = str(r.get("packet_id", "UNKNOWN"))

                # Explicit verifier-side finite checks: reject NaN, None, or non-finite values in EMPTY records
                rf = self.assert_finite_number(r.get("r_fusion"), "r_fusion", pid)
                dc = self.assert_finite_number(r.get("dcri"), "dcri", pid)
                pu = self.assert_finite_number(r.get("uncertainty_penalty"), "uncertainty_penalty", pid)
                if rf != 0.0 or dc != 0.0 or pu != 0.0:
                    self.log_gate(gate_id, False, f"Non-zero decision metrics under EMPTY in packet {pid} ({pol_name})")
                    raise GateFailure()

                weights = r.get("weights", {})
                for m in MODALITIES:
                    w_val = self.assert_finite_number(weights.get(m), f"weights.{m}", pid)
                    if w_val != 0.0:
                        self.log_gate(gate_id, False, f"Non-zero weight {m}={w_val} under EMPTY in packet {pid} ({pol_name})")
                        raise GateFailure()

                if (
                    r.get("status") != "NO_MODALITY_AVAILABLE"
                    or r.get("decision_tier") != 0
                    or r.get("num_active") != 0
                    or r.get("active_modalities") != []
                ):
                    self.log_gate(gate_id, False, f"EMPTY contract violation in packet {pid} ({pol_name})")
                    raise GateFailure()

        self.log_gate(
            gate_id,
            True,
            f"Fail-closed EMPTY handling verified across all policies ({total_empty_checked} packets checked): R=0, DCRI=0, P_U=0, Tier 0, NO_MODALITY_AVAILABLE",
        )

    # -------------------------------------------------------------------------
    # Gate C17-06: Unimodal Singleton Degeneracy Invariance
    # -------------------------------------------------------------------------
    def verify_c17_06_unimodal_degeneracy(self):
        gate_id = "Gate C17-06: Unimodal Singleton Degeneracy Invariance"
        canonical = self.load_canonical_records()
        singleton_map = {"R": "retina", "F": "foot", "C": "clinical"}

        policies_to_check = [("B6", canonical.get("regimes_b6", {}))]
        for pol_id, reg_map in canonical.get("baseline_regimes", {}).items():
            policies_to_check.append((pol_id, reg_map))

        total_singletons_checked = 0
        for pol_name, reg_map in policies_to_check:
            for reg_id, active_mod in singleton_map.items():
                records = reg_map.get(reg_id, [])
                if len(records) != 500:
                    self.log_gate(gate_id, False, f"Expected 500 records for singleton {reg_id} ({pol_name}), found {len(records)}")
                    raise GateFailure()

                for r in records:
                    total_singletons_checked += 1
                    w = r.get("weights", {})
                    w_active = float(w.get(active_mod, 0.0))
                    if abs(w_active - 1.0) > 1e-6:
                        self.log_gate(gate_id, False, f"Singleton weight != 1.0 in {r['packet_id']} ({pol_name}, {reg_id}): {w_active}")
                        raise GateFailure()
                    for m in MODALITIES:
                        if m != active_mod and float(w.get(m, 0.0)) > 1e-7:
                            self.log_gate(gate_id, False, f"Singleton inactive weight > 0 in {r['packet_id']} ({pol_name}, {reg_id}): {m}={w.get(m)}")
                            raise GateFailure()

                    r_active = float(r.get("input_risks", {}).get(active_mod, 0.0))
                    if abs(float(r.get("r_fusion", 0.0)) - r_active) > 1e-6:
                        self.log_gate(gate_id, False, f"Singleton r_fusion != r_active in {r['packet_id']} ({pol_name}, {reg_id})")
                        raise GateFailure()

        self.log_gate(gate_id, True, f"All singleton availability regimes collapse mathematically to unity routing across all policies ({total_singletons_checked} checked)")

    # -------------------------------------------------------------------------
    # Gate C17-07: Unweighted DCRI Penalty Orthogonality & Exact Form
    # -------------------------------------------------------------------------
    def verify_c17_07_unweighted_dcri_penalty(self):
        gate_id = "Gate C17-07: Unweighted DCRI Penalty Orthogonality & Exact Form"
        canonical = self.load_canonical_records()
        delta = float(canonical.get("delta", 0.10))

        expected_baselines = {"B1", "B2", "B3", "B4", "B5", "B6"}
        expected_regimes = {"RFC", "RF", "RC", "FC", "R", "F", "C", "EMPTY"}
        expected_baseline_pols = {"B1", "B2", "B5"}
        expected_degs = {"D0", "D1", "D2", "D3"}
        expected_cal = {"calibrated", "uncalibrated"}

        if set(canonical.get("baselines", {}).keys()) != expected_baselines:
            self.log_gate(gate_id, False, "baselines collection set mismatch")
            raise GateFailure()
        if set(canonical.get("regimes_b6", {}).keys()) != expected_regimes:
            self.log_gate(gate_id, False, "regimes_b6 collection set mismatch")
            raise GateFailure()
        if set(canonical.get("baseline_regimes", {}).keys()) != expected_baseline_pols:
            self.log_gate(gate_id, False, "baseline_regimes collection set mismatch")
            raise GateFailure()
        for p in expected_baseline_pols:
            if set(canonical.get("baseline_regimes", {}).get(p, {}).keys()) != expected_regimes:
                self.log_gate(gate_id, False, f"baseline_regimes[{p}] collection set mismatch")
                raise GateFailure()
        if set(canonical.get("degradations_b6", {}).keys()) != expected_degs:
            self.log_gate(gate_id, False, "degradations_b6 collection set mismatch")
            raise GateFailure()
        if set(canonical.get("degradations_b5", {}).keys()) != expected_degs:
            self.log_gate(gate_id, False, "degradations_b5 collection set mismatch")
            raise GateFailure()
        if set(canonical.get("calibration", {}).keys()) != expected_cal:
            self.log_gate(gate_id, False, "calibration collection set mismatch")
            raise GateFailure()

        record_sets: List[Tuple[str, List[Dict[str, Any]]]] = []
        for pol_id in sorted(expected_baselines):
            record_sets.append((f"baseline_{pol_id}", canonical["baselines"][pol_id]))
        for reg_id in sorted(expected_regimes):
            record_sets.append((f"regime_b6_{reg_id}", canonical["regimes_b6"][reg_id]))
        for pol_id in sorted(expected_baseline_pols):
            for reg_id in sorted(expected_regimes):
                record_sets.append((f"baseline_{pol_id}_{reg_id}", canonical["baseline_regimes"][pol_id][reg_id]))
        for sev in sorted(expected_degs):
            record_sets.append((f"degradation_b6_{sev}", canonical["degradations_b6"][sev]))
        for sev in sorted(expected_degs):
            record_sets.append((f"degradation_b5_{sev}", canonical["degradations_b5"][sev]))
        for cal_key in sorted(expected_cal):
            record_sets.append((f"calibration_{cal_key}", canonical["calibration"][cal_key]))

        if len(record_sets) != 48:
            self.log_gate(gate_id, False, f"Expected exactly 48 record collections, found {len(record_sets)}")
            raise GateFailure()

        total_checked = 0
        for set_name, records in record_sets:
            if len(records) != 500:
                self.log_gate(gate_id, False, f"Collection {set_name} count {len(records)} != 500")
                raise GateFailure()

            pids = set()
            for r in records:
                total_checked += 1
                pid = r.get("packet_id")
                if not pid or pid in pids:
                    self.log_gate(gate_id, False, f"Missing or duplicate packet_id in {set_name}: {pid}")
                    raise GateFailure()
                pids.add(pid)

                if r.get("status") == "NO_MODALITY_AVAILABLE":
                    rf = self.assert_finite_number(r.get("r_fusion"), "r_fusion", pid)
                    dc = self.assert_finite_number(r.get("dcri"), "dcri", pid)
                    pu = self.assert_finite_number(r.get("uncertainty_penalty"), "uncertainty_penalty", pid)
                    if rf != 0.0 or dc != 0.0 or pu != 0.0:
                        self.log_gate(gate_id, False, f"DCRI non-zero on EMPTY in {set_name} / {pid}")
                        raise GateFailure()
                    continue

                if r.get("status") != "SUCCESS" or not r.get("is_finite", False):
                    self.log_gate(gate_id, False, f"Invalid record status/finite in {set_name} / {pid}")
                    raise GateFailure()

                # Explicit verifier-side finite number checks
                f_fields = self.assert_finite_fields(r, ["r_fusion", "uncertainty_penalty", "dcri", "u_sum"], pid)
                actual_r_fusion = f_fields["r_fusion"]
                actual_penalty = f_fields["uncertainty_penalty"]
                actual_dcri = f_fields["dcri"]
                actual_u_sum = f_fields["u_sum"]

                active_mods = r.get("active_modalities", [])
                uncs = r.get("input_uncertainties", {})
                risks = r.get("input_risks", {})
                weights = r.get("weights", {})

                # Validate weights, risks, uncertainties are finite
                for m in MODALITIES:
                    self.assert_finite_number(weights.get(m), f"weights.{m}", pid)
                    self.assert_finite_number(risks.get(m), f"input_risks.{m}", pid)
                    self.assert_finite_number(uncs.get(m), f"input_uncertainties.{m}", pid)

                # Validate weights simplex on active modalities
                w_sum = sum(float(weights.get(m, 0.0)) for m in active_mods)
                if abs(w_sum - 1.0) > 1e-5:
                    self.log_gate(gate_id, False, f"Simplex violation in {set_name} / {pid}: {w_sum}")
                    raise GateFailure()

                # Recompute r_fusion
                expected_r_fusion = sum(float(weights.get(m, 0.0)) * float(risks.get(m, 0.0)) for m in active_mods)
                if abs(actual_r_fusion - expected_r_fusion) > 1e-5:
                    self.log_gate(gate_id, False, f"r_fusion mismatch in {set_name}/{pid}: {actual_r_fusion} vs {expected_r_fusion}")
                    raise GateFailure()

                expected_u_sum = sum(float(uncs.get(m, 0.0)) for m in active_mods)
                if abs(actual_u_sum - expected_u_sum) > 1e-5:
                    self.log_gate(gate_id, False, f"u_sum mismatch in {set_name}/{pid}: {actual_u_sum} vs {expected_u_sum}")
                    raise GateFailure()

                expected_penalty = delta * expected_u_sum
                if abs(actual_penalty - expected_penalty) > 1e-5:
                    self.log_gate(gate_id, False, f"Penalty mismatch in {set_name}/{pid}: {actual_penalty} vs {expected_penalty}")
                    raise GateFailure()

                expected_dcri = actual_r_fusion - actual_penalty
                if abs(actual_dcri - expected_dcri) > 1e-5:
                    self.log_gate(gate_id, False, f"DCRI mismatch in {set_name}/{pid}: {actual_dcri} vs {expected_dcri}")
                    raise GateFailure()

        if total_checked != 24000:
            self.log_gate(gate_id, False, f"Total checked records {total_checked} != 24,000 (expected 48 collections * 500)")
            raise GateFailure()

        self.log_gate(gate_id, True, f"DCRI formula DCRI = R_fusion - P_U and P_U = delta * sum(U_i) verified across all {total_checked} evaluated records")

    # -------------------------------------------------------------------------
    # Gate C17-08: Negative DCRI Preservation
    # -------------------------------------------------------------------------
    def verify_c17_08_negative_dcri_preservation(self):
        gate_id = "Gate C17-08: Negative DCRI Preservation"
        canonical = self.load_canonical_records()
        baselines = canonical.get("baselines", {})

        b6_records = baselines.get("B6", [])
        neg_b6 = [r for r in b6_records if float(r.get("dcri", 0.0)) < 0]

        if len(neg_b6) != 37:
            self.log_gate(gate_id, False, f"Expected 37 negative DCRI records under B6, found {len(neg_b6)}")
            raise GateFailure()

        min_dcri_b6 = min(float(r["dcri"]) for r in neg_b6)
        if min_dcri_b6 >= 0:
            self.log_gate(gate_id, False, f"Negative DCRI values clamped: min={min_dcri_b6}")
            raise GateFailure()

        # Check negative counts across all policies
        expected_neg_counts = {
            "B1": 223,
            "B2": 0,
            "B3": 1,
            "B4": 1,
            "B5": 35,
            "B6": 37,
        }
        for pol, exp_count in expected_neg_counts.items():
            records = baselines.get(pol, [])
            actual_count = sum(1 for r in records if float(r.get("dcri", 0.0)) < 0)
            if actual_count != exp_count:
                self.log_gate(gate_id, False, f"Policy {pol} negative DCRI count mismatch: {actual_count} vs expected {exp_count}")
                raise GateFailure()

        self.log_gate(
            gate_id,
            True,
            f"Negative DCRI preserved unclamped: B6=37 (min={min_dcri_b6:.6f}), B1=223, B2=0, B3=1, B4=1, B5=35",
            {"negative_counts": expected_neg_counts, "min_dcri_b6": min_dcri_b6},
        )

    # -------------------------------------------------------------------------
    # Gate C17-09: Decision Action Tier Conservation
    # -------------------------------------------------------------------------
    def verify_c17_09_decision_tier_conservation(self):
        gate_id = "Gate C17-09: Decision Action Tier Conservation"
        canonical = self.load_canonical_records()
        baselines = canonical.get("baselines", {})
        tau_1 = float(canonical.get("tau_1", 0.20))
        tau_2 = float(canonical.get("tau_2", 0.40))

        for pol_id, records in baselines.items():
            if len(records) != 500:
                self.log_gate(gate_id, False, f"Policy {pol_id} record count {len(records)} != 500")
                raise GateFailure()

            counts = {0: 0, 1: 0, 2: 0}
            for r in records:
                pid = str(r.get("packet_id", "UNKNOWN"))
                score = self.assert_finite_number(r.get("dcri"), "dcri", pid)
                expected_tier = 0 if score < tau_1 else (1 if score < tau_2 else 2)
                actual_tier = r.get("decision_tier")

                if actual_tier not in (0, 1, 2) or actual_tier != expected_tier:
                    self.log_gate(gate_id, False, f"Tier assignment mismatch for {pid} under {pol_id}: {actual_tier} vs {expected_tier}")
                    raise GateFailure()
                counts[actual_tier] += 1

            if sum(counts.values()) != 500:
                self.log_gate(gate_id, False, f"Tier counts for {pol_id} sum to {sum(counts.values())} != 500")
                raise GateFailure()

        b6_counts = {0: 0, 1: 0, 2: 0}
        for r in baselines.get("B6", []):
            b6_counts[r["decision_tier"]] += 1

        if b6_counts[0] != 232 or b6_counts[1] != 187 or b6_counts[2] != 81:
            self.log_gate(gate_id, False, f"B6 tier counts unexpected: {b6_counts}")
            raise GateFailure()

        self.log_gate(gate_id, True, f"Tier conservation verified for all policies; B6 exactly Routine=232, Assessment=187, Escalation=81")

    # -------------------------------------------------------------------------
    # Gate C17-10: Baseline Transition Conservation
    # -------------------------------------------------------------------------
    def verify_c17_10_transition_conservation(self):
        gate_id = "Gate C17-10: Baseline Transition Conservation"
        canonical = self.load_canonical_records()
        baselines = canonical.get("baselines", {})
        b6_records = baselines.get("B6", [])
        b6_map = {r["packet_id"]: r for r in b6_records}
        if len(b6_map) != 500:
            self.log_gate(gate_id, False, f"B6 unique packet count {len(b6_map)} != 500")
            raise GateFailure()

        expected_reclass = {
            "B1": 184,
            "B2": 81,
            "B3": 57,
            "B4": 64,
            "B5": 3,
        }

        for pol_id, exp_reclass in expected_reclass.items():
            cand_records = baselines.get(pol_id, [])
            cand_map = {r["packet_id"]: r for r in cand_records}
            if set(cand_map.keys()) != set(b6_map.keys()) or len(cand_map) != 500:
                self.log_gate(gate_id, False, f"Packet ID mismatch between {pol_id} and B6")
                raise GateFailure()

            matrix = [[0 for _ in range(3)] for _ in range(3)]
            unchanged = 0
            reclassified = 0

            # Deterministic iteration over sorted packet IDs
            for pid in sorted(b6_map.keys()):
                r_cand = cand_map[pid]
                r_b6 = b6_map[pid]
                c_tier = r_cand["decision_tier"]
                b6_tier = r_b6["decision_tier"]
                matrix[b6_tier][c_tier] += 1
                if c_tier == b6_tier:
                    unchanged += 1
                else:
                    reclassified += 1

            total = unchanged + reclassified
            if total != 500 or sum(sum(row) for row in matrix) != 500:
                self.log_gate(gate_id, False, f"Transition matrix for {pol_id} does not sum to 500 (sum={total})")
                raise GateFailure()

            if reclassified != exp_reclass:
                self.log_gate(gate_id, False, f"Policy {pol_id} reclassifications: {reclassified} vs expected {exp_reclass}")
                raise GateFailure()

        self.log_gate(gate_id, True, "All 3x3 transition matrices conserve N=500 and reclassification counts")

    # -------------------------------------------------------------------------
    # Gate C17-11: Quality Bonus Directionality & Specificity
    # -------------------------------------------------------------------------
    def verify_c17_11_quality_bonus_directionality(self):
        gate_id = "Gate C17-11: Quality Bonus Directionality & Specificity"
        canonical = self.load_canonical_records()
        baselines = canonical.get("baselines", {})
        b6_records = baselines.get("B6", [])
        b5_records = baselines.get("B5", [])

        if len(b6_records) != 500 or len(b5_records) != 500:
            self.log_gate(gate_id, False, f"Record count mismatch: B6={len(b6_records)}, B5={len(b5_records)}")
            raise GateFailure()

        b6_by_id = {r["packet_id"]: r for r in b6_records}
        b5_by_id = {r["packet_id"]: r for r in b5_records}
        if set(b6_by_id.keys()) != set(b5_by_id.keys()) or len(b6_by_id) != 500:
            self.log_gate(gate_id, False, "B6 and B5 packet IDs do not match or are not unique (N=500)")
            raise GateFailure()

        pids = sorted(b6_by_id.keys())
        diffs = [float(b6_by_id[pid]["dcri"]) - float(b5_by_id[pid]["dcri"]) for pid in pids]
        point_est = float(np.mean(diffs))

        # Expected point estimate = 0.002653
        if abs(point_est - 0.002653) > 1e-5:
            self.log_gate(gate_id, False, f"Point estimate mismatch: {point_est:.6f} vs expected 0.002653")
            raise GateFailure()

        # Paired bootstrap
        n = len(diffs)
        rng = np.random.default_rng(7117)
        indices = rng.integers(0, n, size=(1000, n))
        resample_means = np.mean(np.array(diffs)[indices], axis=1)
        ci_lower = float(np.percentile(resample_means, 2.5))
        ci_upper = float(np.percentile(resample_means, 97.5))

        # Directionality: Quality bonus increases DCRI and 95% CI strictly excludes zero
        if ci_lower <= 0.0 <= ci_upper or ci_lower <= 0.0:
            self.log_gate(gate_id, False, f"Quality bonus difference does not strictly exceed zero: 95% CI=[{ci_lower:.6f}, {ci_upper:.6f}]")
            raise GateFailure()

        self.log_gate(
            gate_id,
            True,
            f"Quality bonus statistically significant: point={point_est:+.6f}, 95% CI=[{ci_lower:+.6f}, {ci_upper:+.6f}]",
            {"point_estimate": point_est, "ci_lower": ci_lower, "ci_upper": ci_upper},
        )

    # -------------------------------------------------------------------------
    # Gate C17-12: Input Degradation Monotonicity / Response
    # -------------------------------------------------------------------------
    def verify_c17_12_degradation_monotonicity(self):
        gate_id = "Gate C17-12: Input Degradation Monotonicity / Response"
        canonical = self.load_canonical_records()
        degradations = canonical.get("degradations_b6", {})

        expected_severities = ["D0", "D1", "D2", "D3"]
        d0_pids = None
        for sev in expected_severities:
            if sev not in degradations:
                self.log_gate(gate_id, False, f"Degradation severity {sev} missing from canonical records")
                raise GateFailure()
            records = degradations[sev]
            if len(records) != 500:
                self.log_gate(gate_id, False, f"Degradation severity {sev} has {len(records)} records != 500")
                raise GateFailure()

            pids = set()
            for r in records:
                pid = r.get("packet_id")
                if not pid or pid in pids:
                    self.log_gate(gate_id, False, f"Duplicate or missing packet_id {pid} under {sev}")
                    raise GateFailure()
                pids.add(pid)

                if r.get("status") != "SUCCESS" or not r.get("is_finite", False):
                    self.log_gate(gate_id, False, f"Invalid record {pid} under degradation {sev}")
                    raise GateFailure()

                # Explicit verifier-side finite number checks
                f_fields = self.assert_finite_fields(r, ["uncertainty_penalty", "dcri", "r_fusion"], pid)
                delta = float(canonical.get("delta", 0.10))
                u_dict = r.get("input_uncertainties", {})
                u_sum_calc = sum(
                    self.assert_finite_number(u_dict.get(m), f"input_uncertainties.{m}", pid)
                    for m in MODALITIES
                )
                pu_calc = delta * u_sum_calc
                dcri_calc = f_fields["r_fusion"] - pu_calc
                if abs(f_fields["uncertainty_penalty"] - pu_calc) > 1e-6:
                    self.log_gate(gate_id, False, f"Penalty formula mismatch for {pid} under {sev}")
                    raise GateFailure()
                if abs(f_fields["dcri"] - dcri_calc) > 1e-6:
                    self.log_gate(gate_id, False, f"DCRI formula mismatch for {pid} under {sev}")
                    raise GateFailure()

            if d0_pids is None:
                d0_pids = pids
            elif pids != d0_pids:
                self.log_gate(gate_id, False, f"Degradation {sev} packet ID set does not match reference D0 ({len(pids ^ d0_pids)} mismatches)")
                raise GateFailure()

        # Cohort-level monotonicity: mean P_U strictly increases, mean DCRI strictly decreases
        mean_pus = {}
        mean_dcris = {}
        for sev in expected_severities:
            mean_pus[sev] = float(np.mean([float(r["uncertainty_penalty"]) for r in degradations[sev]]))
            mean_dcris[sev] = float(np.mean([float(r["dcri"]) for r in degradations[sev]]))

        # Verify strict monotonic penalty surge: P_U(D0) < P_U(D1) < P_U(D2) < P_U(D3)
        for i in range(len(expected_severities) - 1):
            s_curr, s_next = expected_severities[i], expected_severities[i + 1]
            if mean_pus[s_next] <= mean_pus[s_curr]:
                self.log_gate(gate_id, False, f"Mean uncertainty penalty not strictly increasing: {s_curr}={mean_pus[s_curr]:.6f} >= {s_next}={mean_pus[s_next]:.6f}")
                raise GateFailure()
            if mean_dcris[s_next] >= mean_dcris[s_curr]:
                self.log_gate(gate_id, False, f"Mean DCRI not strictly decreasing: {s_curr}={mean_dcris[s_curr]:.6f} <= {s_next}={mean_dcris[s_next]:.6f}")
                raise GateFailure()

        # Verify exact protocol expectations
        exp_pus = {"D0": 0.063394, "D1": 0.071391, "D2": 0.083231, "D3": 0.101519}
        exp_dcris = {"D0": 0.226506, "D1": 0.212011, "D2": 0.193754, "D3": 0.175799}
        for sev in expected_severities:
            if abs(mean_pus[sev] - exp_pus[sev]) > 1e-5:
                self.log_gate(gate_id, False, f"Penalty mean mismatch for {sev}: {mean_pus[sev]:.6f} vs {exp_pus[sev]:.6f}")
                raise GateFailure()
            if abs(mean_dcris[sev] - exp_dcris[sev]) > 1e-5:
                self.log_gate(gate_id, False, f"DCRI mean mismatch for {sev}: {mean_dcris[sev]:.6f} vs {exp_dcris[sev]:.6f}")
                raise GateFailure()

        self.log_gate(
            gate_id,
            True,
            f"Degradation monotonicity verified across D0–D3: P_U strictly increasing [{mean_pus['D0']:.4f}->{mean_pus['D3']:.4f}], DCRI strictly decreasing [{mean_dcris['D0']:.4f}->{mean_dcris['D3']:.4f}]",
            {"mean_pus": mean_pus, "mean_dcris": mean_dcris},
        )

    # -------------------------------------------------------------------------
    # Gate C17-13: Calibration Condition Integrity
    # -------------------------------------------------------------------------
    def verify_c17_13_calibration_condition(self):
        gate_id = "Gate C17-13: Calibration Condition Integrity"
        canonical = self.load_canonical_records()
        cal = canonical.get("calibration", {})
        cal_pkts = cal.get("calibrated", [])
        uncal_pkts = cal.get("uncalibrated", [])

        if len(cal_pkts) != 500 or len(uncal_pkts) != 500:
            self.log_gate(gate_id, False, f"Calibration record counts mismatch: cal={len(cal_pkts)}, uncal={len(uncal_pkts)}")
            raise GateFailure()

        cal_by_id = {r["packet_id"]: r for r in cal_pkts}
        uncal_by_id = {r["packet_id"]: r for r in uncal_pkts}
        if set(cal_by_id.keys()) != set(uncal_by_id.keys()) or len(cal_by_id) != 500:
            self.log_gate(gate_id, False, "Calibration packet IDs do not match or are not unique (N=500)")
            raise GateFailure()

        # Check finite and SUCCESS
        for pid in cal_by_id:
            rc = cal_by_id[pid]
            ru = uncal_by_id[pid]
            if rc.get("status") != "SUCCESS" or not rc.get("is_finite", False):
                self.log_gate(gate_id, False, f"Non-success or non-finite calibrated record {pid}")
                raise GateFailure()
            if ru.get("status") != "SUCCESS" or not ru.get("is_finite", False):
                self.log_gate(gate_id, False, f"Non-success or non-finite uncalibrated record {pid}")
                raise GateFailure()
            self.assert_finite_fields(rc, ["dcri", "r_fusion", "uncertainty_penalty"], pid)
            self.assert_finite_fields(ru, ["dcri", "r_fusion", "uncertainty_penalty"], pid)

        pids = sorted(cal_by_id.keys())
        diffs = [float(cal_by_id[pid]["dcri"]) - float(uncal_by_id[pid]["dcri"]) for pid in pids]
        point_est = float(np.mean(diffs))

        # Expected point estimate = +0.004615
        if abs(point_est - 0.004615) > 1e-5:
            self.log_gate(gate_id, False, f"Calibration shift point estimate mismatch: {point_est:.6f} vs 0.004615")
            raise GateFailure()

        # Bootstrap
        n = len(diffs)
        rng = np.random.default_rng(7117)
        indices = rng.integers(0, n, size=(1000, n))
        resample_means = np.mean(np.array(diffs)[indices], axis=1)
        ci_lower = float(np.percentile(resample_means, 2.5))
        ci_upper = float(np.percentile(resample_means, 97.5))

        # Statistical directionality (strictly excludes zero) and exact CI reproduction
        if ci_lower <= 0.0 <= ci_upper:
            self.log_gate(gate_id, False, f"Calibration shift crosses zero: [{ci_lower:.6f}, {ci_upper:.6f}]")
            raise GateFailure()
        if abs(ci_lower - 0.003544) > 1e-5 or abs(ci_upper - 0.005752) > 1e-5:
            self.log_gate(gate_id, False, f"Calibration CI mismatch: [{ci_lower:.6f}, {ci_upper:.6f}] vs expected [0.003544, 0.005752]")
            raise GateFailure()

        # Independently recompute tier transitions and invariance
        matrix = [[0 for _ in range(3)] for _ in range(3)]
        unchanged = 0
        reclassified = 0
        upgrades = 0
        downgrades = 0

        for pid in pids:
            u_tier = uncal_by_id[pid]["decision_tier"]
            c_tier = cal_by_id[pid]["decision_tier"]
            matrix[u_tier][c_tier] += 1
            if u_tier == c_tier:
                unchanged += 1
            else:
                reclassified += 1
                if c_tier > u_tier:
                    upgrades += 1
                else:
                    downgrades += 1

        if unchanged != 484 or reclassified != 16 or upgrades != 12 or downgrades != 4:
            self.log_gate(
                gate_id,
                False,
                f"Transition mismatch: unchanged={unchanged} (exp 484), reclass={reclassified} (exp 16), up={upgrades} (exp 12), down={downgrades} (exp 4)",
            )
            raise GateFailure()

        expected_matrix = [
            [231, 10, 0],
            [1, 174, 2],
            [0, 3, 79],
        ]
        if matrix != expected_matrix:
            self.log_gate(gate_id, False, f"Transition matrix mismatch: {matrix} vs expected {expected_matrix}")
            raise GateFailure()

        self.log_gate(
            gate_id,
            True,
            f"Calibration condition verified: point={point_est:+.6f}, 95% CI=[{ci_lower:+.6f}, {ci_upper:+.6f}], unchanged={unchanged}/500 (96.8%), reclassified={reclassified}",
            {"point_estimate": point_est, "ci_lower": ci_lower, "ci_upper": ci_upper, "matrix": matrix},
        )

    # -------------------------------------------------------------------------
    # Gate C17-14: Continuous Discordance & Conflict Stratification
    # -------------------------------------------------------------------------
    def verify_c17_14_conflict_stratification(self):
        gate_id = "Gate C17-014: Continuous Discordance & Conflict Stratification"
        canonical = self.load_canonical_records()
        b6_records = canonical.get("baselines", {}).get("B6", [])

        if len(b6_records) != 500:
            self.log_gate(gate_id, False, f"B6 record count {len(b6_records)} != 500")
            raise GateFailure()

        strata_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
        strata_tiers = {
            "LOW": {0: 0, 1: 0, 2: 0},
            "MODERATE": {0: 0, 1: 0, 2: 0},
            "HIGH": {0: 0, 1: 0, 2: 0},
        }

        for r in b6_records:
            pid = str(r.get("packet_id", "UNKNOWN"))
            f_fields = self.assert_finite_fields(r, ["r_fusion", "css", "delta_max", "sigma_w"], pid)
            risks = r.get("input_risks", {})
            weights = r.get("weights", {})

            # Explicitly validate finite inputs
            r_vals = [
                self.assert_finite_number(risks.get(m), f"input_risks.{m}", pid)
                for m in MODALITIES
            ]
            w_vals = [
                self.assert_finite_number(weights.get(m), f"weights.{m}", pid)
                for m in MODALITIES
            ]
            r_fusion = f_fields["r_fusion"]

            # Distinct pairwise modality differences: (retina, foot), (retina, clinical), (foot, clinical)
            delta_rf = abs(r_vals[0] - r_vals[1])
            delta_rc = abs(r_vals[0] - r_vals[2])
            delta_fc = abs(r_vals[1] - r_vals[2])
            pairwise = [delta_rf, delta_rc, delta_fc]
            delta_max = max(pairwise)

            sigma_w = math.sqrt(sum(w_vals[i] * ((r_vals[i] - r_fusion) ** 2) for i in range(3)))

            cat = "LOW" if delta_max < 0.20 else ("MODERATE" if delta_max < 0.35 else "HIGH")

            if abs(f_fields["css"] - delta_max) > 1e-6:
                self.log_gate(gate_id, False, f"CSS mismatch for packet {pid}: {f_fields['css']} vs {delta_max}")
                raise GateFailure()
            if abs(f_fields["delta_max"] - delta_max) > 1e-6:
                self.log_gate(gate_id, False, f"delta_max mismatch for packet {pid}: {f_fields['delta_max']} vs {delta_max}")
                raise GateFailure()
            if abs(f_fields["sigma_w"] - sigma_w) > 1e-6:
                self.log_gate(gate_id, False, f"sigma_w mismatch for packet {pid}: {f_fields['sigma_w']} vs {sigma_w}")
                raise GateFailure()
            if r.get("conflict_category") != cat:
                self.log_gate(gate_id, False, f"Category mismatch for packet {pid}: {r.get('conflict_category')} vs {cat}")
                raise GateFailure()

            tier = r.get("decision_tier")
            strata_counts[cat] += 1
            strata_tiers[cat][tier] += 1

        if strata_counts["LOW"] != 45 or strata_counts["MODERATE"] != 92 or strata_counts["HIGH"] != 363:
            self.log_gate(gate_id, False, f"Conflict strata counts unexpected: {strata_counts}")
            raise GateFailure()

        exp_tiers = {
            "LOW": {0: 43, 1: 2, 2: 0},
            "MODERATE": {0: 74, 1: 17, 2: 1},
            "HIGH": {1: 168, 0: 115, 2: 80},
        }
        for cat in ["LOW", "MODERATE", "HIGH"]:
            for t_idx in [0, 1, 2]:
                if strata_tiers[cat][t_idx] != exp_tiers[cat][t_idx]:
                    self.log_gate(gate_id, False, f"Per-stratum tier distribution mismatch for {cat} Tier {t_idx}: {strata_tiers[cat][t_idx]} vs {exp_tiers[cat][t_idx]}")
                    raise GateFailure()

        # Check escalation rate monotonicity across conflict strata:
        esc_low = strata_tiers["LOW"][2] / strata_counts["LOW"]
        esc_mod = strata_tiers["MODERATE"][2] / strata_counts["MODERATE"]
        esc_high = strata_tiers["HIGH"][2] / strata_counts["HIGH"]
        if not (esc_low < esc_mod < esc_high):
            self.log_gate(gate_id, False, f"Escalation rate not monotonically increasing with conflict: LOW={esc_low:.4f}, MOD={esc_mod:.4f}, HIGH={esc_high:.4f}")
            raise GateFailure()

        self.log_gate(
            gate_id,
            True,
            f"Conflict stratification verified: LOW=45 (esc={esc_low:.1%}), MODERATE=92 (esc={esc_mod:.1%}), HIGH=363 (esc={esc_high:.1%}) with strict monotonic escalation",
            strata_counts,
        )

    # -------------------------------------------------------------------------
    # Gate C17-15: Paired Bootstrap Reproducibility
    # -------------------------------------------------------------------------
    def verify_c17_15_bootstrap_reproducibility(self):
        gate_id = "Gate C17-15: Paired Bootstrap Reproducibility"
        stats_path = self.exp_dir / "06_statistical_analysis.json"
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        if stats.get("bootstrap_rounds") != 1000 or stats.get("bootstrap_seed") != 7117:
            self.log_gate(gate_id, False, "Bootstrap configuration mismatch in statistical analysis")
            raise GateFailure()

        canonical = self.load_canonical_records()
        baselines = canonical.get("baselines", {})
        b6_records = baselines.get("B6", [])
        b6_by_id = {r["packet_id"]: r for r in b6_records}

        if len(b6_by_id) != 500:
            self.log_gate(gate_id, False, f"B6 unique packet count {len(b6_by_id)} != 500")
            raise GateFailure()

        pids = sorted(b6_by_id.keys())
        n = len(pids)

        for pol_id in ["B1", "B2", "B3", "B4", "B5"]:
            cand_records = baselines.get(pol_id, [])
            cand_by_id = {r["packet_id"]: r for r in cand_records}
            if set(cand_by_id.keys()) != set(b6_by_id.keys()) or len(cand_by_id) != 500:
                self.log_gate(gate_id, False, f"Packet ID mismatch between {pol_id} and B6")
                raise GateFailure()

            diffs = [float(b6_by_id[pid]["dcri"]) - float(cand_by_id[pid]["dcri"]) for pid in pids]

            rng = np.random.default_rng(7117)
            indices = rng.integers(0, n, size=(1000, n))
            resample_means = np.mean(np.array(diffs)[indices], axis=1)
            ci_lower = float(np.percentile(resample_means, 2.5))
            ci_upper = float(np.percentile(resample_means, 97.5))

            expected_ci = stats["baseline_comparisons"][pol_id]["diff_dcri"]
            if abs(ci_lower - expected_ci["ci_lower"]) > 1e-5 or abs(ci_upper - expected_ci["ci_upper"]) > 1e-5:
                self.log_gate(gate_id, False, f"Bootstrap CI mismatch for {pol_id}: [{ci_lower:.6f}, {ci_upper:.6f}] vs [{expected_ci['ci_lower']}, {expected_ci['ci_upper']}]")
                raise GateFailure()

        self.log_gate(gate_id, True, "Bootstrap reproducibility verified across B1–B5 (B=1,000, seed=7117, alpha=0.05)")

    # -------------------------------------------------------------------------
    # Gate C17-16: Research Volume Structural Presence & Population Audit
    # -------------------------------------------------------------------------
    def verify_c17_16_documentation_presence(self):
        gate_id = "Gate C17-16: Research Volume Structural Presence & Population Audit"
        if not self.docs_dir.is_dir():
            self.log_gate(gate_id, False, f"Docs directory missing: {self.docs_dir}")
            raise GateFailure()

        missing = []
        underpopulated = []
        forbidden_found = []

        forbidden_tokens = ["TODO", "TBD", "PLACEHOLDER", "Lorem ipsum"]

        for doc in REQUIRED_DOCS:
            doc_path = self.docs_dir / doc
            if not doc_path.is_file():
                missing.append(doc)
                continue
            content = doc_path.read_text(encoding="utf-8")
            if len(content.strip()) < 500:
                underpopulated.append((doc, len(content.strip())))
            for tok in forbidden_tokens:
                if tok in content:
                    forbidden_found.append((doc, tok))

        if missing or underpopulated or forbidden_found:
            self.log_gate(
                gate_id,
                False,
                f"Missing docs: {missing}, Under-populated docs: {underpopulated}, Forbidden tokens: {forbidden_found}",
            )
            raise GateFailure()

        self.log_gate(
            gate_id,
            True,
            f"All {len(REQUIRED_DOCS)} research volume documents verified, fully populated and free of placeholder tokens",
        )


if __name__ == "__main__":
    verifier = DecisionEvaluationVerifier()
    success = verifier.verify_all_gates()
    sys.exit(0 if success else 1)
