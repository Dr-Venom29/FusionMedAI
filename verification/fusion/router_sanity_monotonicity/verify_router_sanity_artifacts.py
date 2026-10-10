import sys
import json
import math
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

EXPECTED_ARTIFACT_FILES = ("protocol.json", "test_results.json", "summary.json")

EXPECTED_MUTATION_TESTS = {
    "test_mutation_1_inverted_uncertainty_sign",
    "test_mutation_2_inverted_confidence_sign",
    "test_mutation_3_artifact_inactive_weight_leakage",
    "test_mutation_4_artifact_broken_manifest_hash",
    "test_mutation_5_artifact_corrupt_frozen_coefficients",
    "test_mutation_6_artifact_missing_production_boundary_field",
    "test_mutation_7_artifact_mutation_count_mismatch",
    "test_mutation_8_artifact_malformed_numeric_field",
    "test_mutation_9_artifact_summary_status_mismatch",
    "test_mutation_10_artifact_summary_phase_mismatch",
    "test_mutation_11_artifact_gates_passed_count_mismatch",
    "test_mutation_12_artifact_manifest_hashes_null",
    "test_mutation_13_artifact_malformed_root_array",
    "test_mutation_14_artifact_summary_scorecard_contradiction",
}


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def safe_float(val: Any) -> Optional[float]:
    """Safely converts a value to finite float or returns None (rejecting booleans)."""
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def safe_int(val: Any) -> Optional[int]:
    """Safely converts a value to int (rejecting booleans, floats, and non-integral strings)."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            if "." in val or "e" in val.lower():
                return None
            return int(val)
        except (ValueError, TypeError):
            return None
    return None


def safe_dict(val: Any) -> Dict[str, Any]:
    """Safely returns dict or empty dict if not a dict."""
    return val if isinstance(val, dict) else {}


def fmt_float(val: Any, fmt: str = ".2e") -> str:
    """Safely formats a float or returns 'N/A' if invalid."""
    f = safe_float(val)
    return f"{f:{fmt}}" if f is not None else "N/A"


class RouterSanityArtifactVerifier:
    """
    Independent deep verifier for C11.15 frozen artifacts.
    Evaluates recorded evidence across artifacts, verifies SHA-256 manifest integrity,
    and cross-checks consistency across protocol, results, summary, and manifest.
    """

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.protocol_file = self.artifacts_dir / "protocol.json"
        self.test_results_file = self.artifacts_dir / "test_results.json"
        self.summary_file = self.artifacts_dir / "summary.json"
        self.manifest_file = self.artifacts_dir / "freeze_manifest.json"

    def run_all_gates(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """Executes the 16 deep verification gates (S15-00 through S15-15)."""
        gates = []

        # Gate 0: File Existence of all expected artifacts and manifest
        files_exist = (
            self.protocol_file.is_file()
            and self.test_results_file.is_file()
            and self.summary_file.is_file()
            and self.manifest_file.is_file()
        )
        if not files_exist:
            missing = [
                f for f in list(EXPECTED_ARTIFACT_FILES) + ["freeze_manifest.json"]
                if not (self.artifacts_dir / f).is_file()
            ]
            return False, [{
                "id": "S15-00",
                "name": "File Existence Gate",
                "passed": False,
                "details": f"Missing required artifact files: {missing}",
            }]

        # Load artifacts safely
        try:
            with open(self.protocol_file, "r", encoding="utf-8") as f:
                protocol = json.load(f)
            with open(self.test_results_file, "r", encoding="utf-8") as f:
                test_results = json.load(f)
            with open(self.summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
            with open(self.manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            return False, [{
                "id": "S15-00",
                "name": "JSON Parsing Gate",
                "passed": False,
                "details": f"Failed to parse artifact JSON: {str(e)}",
            }]

        # Validate root schemas (must all be JSON objects / dictionaries)
        for fname, obj in [("protocol.json", protocol), ("test_results.json", test_results), ("summary.json", summary), ("freeze_manifest.json", manifest)]:
            if not isinstance(obj, dict):
                return False, [{
                    "id": "S15-00",
                    "name": "JSON Root Schema Gate",
                    "passed": False,
                    "details": f"Artifact '{fname}' root must be a JSON object, got {type(obj).__name__}",
                }]

        # Gate 1: Cryptographic Hash Integrity & Cross-Artifact Consistency
        artifact_hashes = manifest.get("artifact_hashes")
        if not isinstance(artifact_hashes, dict):
            return False, [{
                "id": "S15-00",
                "name": "Cryptographic Hash & Cross-Artifact Consistency",
                "passed": False,
                "details": f"Manifest 'artifact_hashes' must be a JSON object, got {type(artifact_hashes).__name__}",
            }]

        hash_keys = set(artifact_hashes.keys())
        expected_keys = set(EXPECTED_ARTIFACT_FILES)
        keys_match = (hash_keys == expected_keys) and (len(hash_keys) == len(EXPECTED_ARTIFACT_FILES))
        matched_count = 0
        hash_details = {}

        for fname in EXPECTED_ARTIFACT_FILES:
            if fname in artifact_hashes and isinstance(artifact_hashes[fname], str):
                actual_hash = compute_sha256(self.artifacts_dir / fname)
                expected_hash = artifact_hashes[fname]
                matches = (actual_hash == expected_hash)
                if matches:
                    matched_count += 1
                hash_details[fname] = {"matches": matches, "expected": expected_hash[:10], "actual": actual_hash[:10]}

        hashes_pass = keys_match and (matched_count == len(EXPECTED_ARTIFACT_FILES))

        # Cross-Artifact Consistency Validation
        summary_phase_ok = (summary.get("phase") == "C11.15")
        protocol_phase_ok = (protocol.get("phase") == "C11.15")
        manifest_phase_ok = (manifest.get("phase") == "C11.15")
        test_results_phase_ok = (test_results.get("phase") == "C11.15")

        summary_status_ok = (summary.get("evaluation_status") == "PASSED")
        summary_scorecard = summary.get("scorecard")
        expected_criteria = {f"S15-{i:02d}" for i in range(1, 16)}

        scorecard_reconciliation_ok = True
        reconciliation_mismatches = []

        if isinstance(summary_scorecard, dict):
            scorecard_keys = set(summary_scorecard.keys())
            scorecard_complete = (scorecard_keys == expected_criteria)
            scorecard_passed_count = sum(
                1 for v in summary_scorecard.values()
                if isinstance(v, dict) and v.get("passed") is True
            )
            scorecard_all_pass = scorecard_complete and (scorecard_passed_count == 15)

            # Direct 1-to-1 comparison of summary scorecard against test_results.json
            cfg_ev = safe_dict(test_results.get("configuration_evaluations"))
            mono_ev = safe_dict(test_results.get("monotonicity_evaluations"))
            inv_ev = safe_dict(test_results.get("invariants_evaluations"))
            edge_ev = safe_dict(test_results.get("edge_cases_evaluations"))
            pipe_ev = safe_dict(test_results.get("pipeline_evaluations"))
            reg_ev = safe_dict(test_results.get("regression_evaluations"))
            mut_ev = safe_dict(test_results.get("mutation_evaluations"))

            # Derive underlying evaluated status map
            expected_status_map = {
                "S15-01": cfg_ev.get("passed") is True and cfg_ev.get("matches_expected") is True,
                "S15-02": safe_dict(mono_ev.get("confidence")).get("status") == "PASS",
                "S15-03": safe_dict(mono_ev.get("reliability")).get("status") == "PASS",
                "S15-04": safe_dict(mono_ev.get("uncertainty")).get("status") == "PASS",
                "S15-05": safe_dict(mono_ev.get("quality")).get("status") == "PASS",
                "S15-06": safe_dict(inv_ev.get("simplex_conservation")).get("passed") is True,
                "S15-07": safe_dict(inv_ev.get("hard_availability_masking")).get("passed") is True,
                "S15-08": safe_dict(inv_ev.get("softmax_shift_invariance")).get("passed") is True,
                "S15-09": safe_dict(inv_ev.get("modality_permutation_invariance")).get("passed") is True,
                "S15-10": (
                    safe_dict(inv_ev.get("numerical_stability")).get("passed") is True
                    and safe_dict(edge_ev.get("input_validation_rejection")).get("all_invalid_inputs_rejected") is True
                ),
                "S15-11": safe_dict(edge_ev.get("empty_modality_handling")).get("passed") is True,
                "S15-12": safe_dict(edge_ev.get("masked_value_invariance")).get("passed") is True,
                "S15-13": safe_dict(pipe_ev.get("decoupling_verification")).get("passed") is True,
                "S15-14": reg_ev.get("passed") is True,
                "S15-15": mut_ev.get("passed") is True,
            }

            for cid in sorted(expected_criteria):
                sc_entry = summary_scorecard.get(cid)
                sc_pass = sc_entry.get("passed") if isinstance(sc_entry, dict) else None
                expected_pass = expected_status_map.get(cid)
                if sc_pass is not expected_pass or sc_pass is not True:
                    scorecard_reconciliation_ok = False
                    reconciliation_mismatches.append(f"{cid}: scorecard={sc_pass} vs test_results={expected_pass}")
        else:
            scorecard_keys = set()
            scorecard_complete = False
            scorecard_passed_count = 0
            scorecard_all_pass = False
            scorecard_reconciliation_ok = False

        gates_passed_str = summary.get("gates_passed")
        gates_passed_ok = (
            gates_passed_str == "15/15"
            and scorecard_passed_count == 15
        )

        cross_artifact_ok = (
            summary_phase_ok
            and protocol_phase_ok
            and manifest_phase_ok
            and test_results_phase_ok
            and summary_status_ok
            and scorecard_complete
            and scorecard_all_pass
            and scorecard_reconciliation_ok
            and gates_passed_ok
        )

        gate_00_pass = bool(hashes_pass and cross_artifact_ok)
        gates.append({
            "id": "S15-00",
            "name": "Cryptographic Hash & Cross-Artifact Consistency",
            "passed": gate_00_pass,
            "details": f"{matched_count}/3 hashes verified, summary_status={summary.get('evaluation_status')}, phase_ok={summary_phase_ok}, scorecard={scorecard_passed_count}/15 reconciled" if gate_00_pass else f"Integrity failure: hashes_pass={hashes_pass}, cross_artifact_ok={cross_artifact_ok}, mismatches={reconciliation_mismatches}",
        })

        # Gate 2: S15-01 Frozen Reference Configuration Integrity
        frozen_cfg = safe_dict(protocol.get("frozen_configuration"))
        coeffs = safe_dict(frozen_cfg.get("coefficients"))
        delta = safe_float(frozen_cfg.get("delta"))
        cohort_n = safe_int(frozen_cfg.get("benchmark_cohort_n"))
        cohort_seed = safe_int(frozen_cfg.get("benchmark_cohort_seed"))
        config_eval = safe_dict(test_results.get("configuration_evaluations"))

        theta_ok = (
            safe_float(coeffs.get("alpha")) == 1.0
            and safe_float(coeffs.get("beta")) == 1.5
            and safe_float(coeffs.get("gamma")) == 1.0
            and safe_float(coeffs.get("eta")) == 0.5
            and delta == 0.10
            and cohort_n == 500
            and cohort_seed == 115
            and config_eval.get("passed") is True
            and config_eval.get("matches_expected") is True
        )
        gates.append({
            "id": "S15-01",
            "name": "Frozen Reference Configuration Integrity (Theta_0, delta=0.10)",
            "passed": theta_ok,
            "details": f"Theta_0={coeffs}, delta={delta}, eval_passed={config_eval.get('passed')}",
        })

        # Gate 3: S15-02 Confidence Monotonicity
        conf_mono = safe_dict(safe_dict(test_results.get("monotonicity_evaluations")).get("confidence"))
        passed_trials = safe_int(conf_mono.get("passed_trials"))
        total_trials = safe_int(conf_mono.get("total_trials"))
        conf_pass = (
            conf_mono.get("status") == "PASS"
            and passed_trials is not None
            and total_trials is not None
            and passed_trials == total_trials
            and total_trials == 144
        )
        gates.append({
            "id": "S15-02",
            "name": "Confidence Monotonicity (d w_i / d C_i > 0)",
            "passed": conf_pass,
            "details": f"{passed_trials}/{total_trials} trials passed",
        })

        # Gate 4: S15-03 Reliability Monotonicity
        rel_mono = safe_dict(safe_dict(test_results.get("monotonicity_evaluations")).get("reliability"))
        passed_trials = safe_int(rel_mono.get("passed_trials"))
        total_trials = safe_int(rel_mono.get("total_trials"))
        rel_pass = (
            rel_mono.get("status") == "PASS"
            and passed_trials is not None
            and total_trials is not None
            and passed_trials == total_trials
            and total_trials == 144
        )
        gates.append({
            "id": "S15-03",
            "name": "Reliability Monotonicity (d w_i / d R_i > 0 in kernel fixture)",
            "passed": rel_pass,
            "details": f"{passed_trials}/{total_trials} trials passed",
        })

        # Gate 5: S15-04 Uncertainty Monotonicity
        unc_mono = safe_dict(safe_dict(test_results.get("monotonicity_evaluations")).get("uncertainty"))
        passed_trials = safe_int(unc_mono.get("passed_trials"))
        total_trials = safe_int(unc_mono.get("total_trials"))
        unc_pass = (
            unc_mono.get("status") == "PASS"
            and passed_trials is not None
            and total_trials is not None
            and passed_trials == total_trials
            and total_trials == 144
        )
        gates.append({
            "id": "S15-04",
            "name": "Uncertainty Monotonicity (d w_i / d U_i < 0)",
            "passed": unc_pass,
            "details": f"{passed_trials}/{total_trials} trials passed",
        })

        # Gate 6: S15-05 Quality Monotonicity
        qual_mono = safe_dict(safe_dict(test_results.get("monotonicity_evaluations")).get("quality"))
        passed_trials = safe_int(qual_mono.get("passed_trials"))
        total_trials = safe_int(qual_mono.get("total_trials"))
        qual_pass = (
            qual_mono.get("status") == "PASS"
            and passed_trials is not None
            and total_trials is not None
            and passed_trials == total_trials
            and total_trials == 144
        )
        gates.append({
            "id": "S15-05",
            "name": "Quality Monotonicity (d w_i / d Q_i > 0)",
            "passed": qual_pass,
            "details": f"{passed_trials}/{total_trials} trials passed",
        })

        # Gate 7: S15-06 Simplex Conservation
        simplex = safe_dict(safe_dict(test_results.get("invariants_evaluations")).get("simplex_conservation"))
        max_sum_dev = safe_float(simplex.get("max_sum_deviation"))
        min_weight = safe_float(simplex.get("min_weight_observed"))
        viols = safe_int(simplex.get("violation_count"))
        simplex_pass = (
            simplex.get("passed") is True
            and max_sum_dev is not None and max_sum_dev < 1e-10
            and min_weight is not None and min_weight >= 0.0
            and viols is not None and viols == 0
        )
        gates.append({
            "id": "S15-06",
            "name": "Simplex Conservation (sum w_i = 1.0 within 1e-10, w_i >= 0)",
            "passed": simplex_pass,
            "details": f"max_dev={fmt_float(max_sum_dev)}, min_w={fmt_float(min_weight, '.4f')}, viols={viols}",
        })

        # Gate 8: S15-07 Hard Availability Masking
        masking = safe_dict(safe_dict(test_results.get("invariants_evaluations")).get("hard_availability_masking"))
        viols = safe_int(masking.get("violation_count"))
        total_evals = safe_int(masking.get("total_evaluations"))
        masking_pass = (
            masking.get("passed") is True
            and viols is not None and viols == 0
            and total_evals is not None and total_evals > 0
        )
        gates.append({
            "id": "S15-07",
            "name": "Hard Availability Masking (A_i = 0 => w_i = 0.0 exact)",
            "passed": masking_pass,
            "details": f"{viols} violations across {total_evals} evals",
        })

        # Gate 9: S15-08 Softmax Shift Invariance
        shift = safe_dict(safe_dict(test_results.get("invariants_evaluations")).get("softmax_shift_invariance"))
        max_shift_dev = safe_float(shift.get("max_shift_deviation"))
        viols = safe_int(shift.get("violation_count"))
        shift_pass = (
            shift.get("passed") is True
            and shift.get("production_route_shift_passed") is True
            and shift.get("reference_kernel_shift_passed") is True
            and max_shift_dev is not None and max_shift_dev < 1e-12
            and viols is not None and viols == 0
        )
        gates.append({
            "id": "S15-08",
            "name": "Softmax Shift Invariance (z_i + c => w_i invariant within 1e-12)",
            "passed": shift_pass,
            "details": f"max_shift_dev={fmt_float(max_shift_dev)}, route_shift={shift.get('production_route_shift_passed')}",
        })

        # Gate 10: S15-09 Modality Permutation Invariance
        perm = safe_dict(safe_dict(test_results.get("invariants_evaluations")).get("modality_permutation_invariance"))
        max_perm_dev = safe_float(perm.get("max_permutation_deviation"))
        viols = safe_int(perm.get("violation_count"))
        perm_pass = (
            perm.get("passed") is True
            and perm.get("reference_kernel_perm_passed") is True
            and max_perm_dev is not None and max_perm_dev < 1e-12
            and viols is not None and viols == 0
        )
        gates.append({
            "id": "S15-09",
            "name": "Reference Normalization Kernel Permutation Invariance (Order Independence)",
            "passed": perm_pass,
            "details": f"max_perm_dev={fmt_float(max_perm_dev)}, kernel_perm={perm.get('reference_kernel_perm_passed')}",
        })

        # Gate 11: S15-10 Numerical Stability & Input Validation Rejection
        stab = safe_dict(safe_dict(test_results.get("invariants_evaluations")).get("numerical_stability"))
        input_val = safe_dict(safe_dict(test_results.get("edge_cases_evaluations")).get("input_validation_rejection"))
        passed_cases = safe_int(input_val.get("passed_cases"))
        total_cases = safe_int(input_val.get("total_cases"))
        stab_pass = (
            stab.get("passed") is True
            and stab.get("production_route_boundary_passed") is True
            and stab.get("reference_kernel_extreme_logits_passed") is True
            and input_val.get("all_invalid_inputs_rejected") is True
            and passed_cases == 19
            and total_cases == 19
        )
        gates.append({
            "id": "S15-10",
            "name": "Numerical Stability & Input Validation Rejection",
            "passed": stab_pass,
            "details": f"route_boundary={stab.get('production_route_boundary_passed')}, extreme_kernel={stab.get('reference_kernel_extreme_logits_passed')}, input_val_cases={passed_cases}/{total_cases}",
        })

        # Gate 12: S15-11 Empty Modality Fail-Closed Handling
        empty = safe_dict(safe_dict(test_results.get("edge_cases_evaluations")).get("empty_modality_handling"))
        num_active = safe_int(empty.get("num_active"))
        empty_pass = (
            empty.get("passed") is True
            and empty.get("status") == "NO_MODALITY_AVAILABLE"
            and num_active == 0
        )
        gates.append({
            "id": "S15-11",
            "name": "Empty Modality Fail-Closed Handling (NO_MODALITY_AVAILABLE)",
            "passed": empty_pass,
            "details": f"status={empty.get('status')}, num_active={num_active}",
        })

        # Gate 13: S15-12 Masked-Value Corruption Invariance
        corrupt = safe_dict(safe_dict(test_results.get("edge_cases_evaluations")).get("masked_value_invariance"))
        max_active_dev = safe_float(corrupt.get("max_active_weight_dev"))
        viols = safe_int(corrupt.get("violation_count"))
        corrupt_pass = (
            corrupt.get("passed") is True
            and max_active_dev is not None and max_active_dev < 1e-12
            and viols is not None and viols == 0
        )
        gates.append({
            "id": "S15-12",
            "name": "Masked-Value Corruption Invariance",
            "passed": corrupt_pass,
            "details": f"max_active_weight_dev={fmt_float(max_active_dev)}, violations={viols}",
        })

        # Gate 14: S15-13 Pipeline Separation & Property Decoupling
        pipe = safe_dict(safe_dict(test_results.get("pipeline_evaluations")).get("decoupling_verification"))
        w_delta = safe_float(pipe.get("weight_delta"))
        sc_a_risk = safe_float(pipe.get("scenario_a_risk_delta"))
        sc_b_risk = safe_float(pipe.get("scenario_b_risk_delta"))
        sc_a_dcri = safe_float(pipe.get("scenario_a_dcri_delta"))
        sc_b_dcri = safe_float(pipe.get("scenario_b_dcri_delta"))
        pipe_pass = (
            pipe.get("passed") is True
            and w_delta is not None and w_delta > 0.0
            and sc_a_risk is not None and sc_a_risk > 0.0
            and sc_b_risk is not None and sc_b_risk < 0.0
            and sc_a_dcri is not None and sc_a_dcri > 0.0
            and sc_b_dcri is not None and sc_b_dcri < 0.0
            and pipe.get("delta_matches_frozen") is True
        )
        gates.append({
            "id": "S15-13",
            "name": "Pipeline Stage Separation & Property Decoupling",
            "passed": pipe_pass,
            "details": f"weight_delta={fmt_float(w_delta, '.4f')}, sc_a_risk={fmt_float(sc_a_risk, '.4f')}, sc_b_risk={fmt_float(sc_b_risk, '.4f')}",
        })

        # Gate 15: S15-14 Regression Invariance Execution Check
        regression = safe_dict(test_results.get("regression_evaluations"))
        tests_run = safe_int(regression.get("tests_run"))
        failures = safe_int(regression.get("failures"))
        errors = safe_int(regression.get("errors"))
        skipped = safe_int(regression.get("skipped"))
        regression_pass = (
            regression.get("passed") is True
            and tests_run == 22
            and failures == 0
            and errors == 0
            and skipped == 0
        )
        gates.append({
            "id": "S15-14",
            "name": "Regression Invariance (C11.14 Suite Execution Evidence)",
            "passed": regression_pass,
            "details": f"tests_run={tests_run}/22, failures={failures}, errors={errors}",
        })

        # Gate 16: S15-15 Mutation Detection Execution Check
        mutation = safe_dict(test_results.get("mutation_evaluations"))
        mutations_tested = safe_int(mutation.get("mutations_tested"))
        expected_mutations = safe_int(mutation.get("expected_mutations"))
        raw_names = mutation.get("executed_test_names")
        if isinstance(raw_names, list) and all(isinstance(x, str) for x in raw_names):
            executed_names = set(raw_names)
        else:
            executed_names = set()

        failures = safe_int(mutation.get("failures"))
        errors = safe_int(mutation.get("errors"))
        skipped = safe_int(mutation.get("skipped"))
        names_match = (mutation.get("names_match") is True) and (executed_names == EXPECTED_MUTATION_TESTS)

        mutation_pass = (
            mutation.get("passed") is True
            and mutations_tested == len(EXPECTED_MUTATION_TESTS)
            and expected_mutations == len(EXPECTED_MUTATION_TESTS)
            and names_match
            and failures == 0
            and errors == 0
            and skipped == 0
        )
        gates.append({
            "id": "S15-15",
            "name": "Verifier Mutation Fault Detection Evidence",
            "passed": mutation_pass,
            "details": f"mutations_tested={mutations_tested}/{len(EXPECTED_MUTATION_TESTS)}, names_match={names_match}, failures={failures}, errors={errors}",
        })

        all_passed = all(g["passed"] for g in gates)
        return all_passed, gates

    def generate_certification_report(self, all_passed: bool, gates: List[Dict[str, Any]]) -> Path:
        """Writes an independent formal certification report recording verification outcome, hashes, and gate results."""
        from datetime import datetime, timezone
        report_path = self.artifacts_dir / "certification_report.json"
        
        artifact_hashes = {}
        for fname in list(EXPECTED_ARTIFACT_FILES) + ["freeze_manifest.json"]:
            fpath = self.artifacts_dir / fname
            if fpath.is_file():
                artifact_hashes[fname] = compute_sha256(fpath)
            else:
                artifact_hashes[fname] = "MISSING"

        report = {
            "phase": "C11.15",
            "title": "ACARA-U Residual Router Sanity & Monotonicity Formal Certification Record",
            "verification_status": "CERTIFIED_AND_SEALED" if all_passed else "VERIFICATION_FAILED",
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "verifier_version": "1.0.0",
            "total_gates": len(gates),
            "passed_gates": sum(1 for g in gates if g.get("passed")),
            "verified_artifact_hashes": artifact_hashes,
            "gate_results": gates,
            "envelope_parameters": {
                "reference_coefficients": {"alpha": 1.0, "beta": 1.5, "gamma": 1.0, "eta": 0.5},
                "uncertainty_discount_delta": 0.10,
                "benchmark_cohort_n": 500,
                "benchmark_cohort_seed": 115,
            },
        }

        # Atomic write with allow_nan=False
        temp_file = report_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)
        temp_file.replace(report_path)
        return report_path


def main():
    root = Path(__file__).resolve().parents[3]
    artifacts_dir = root / "experiments" / "fusion" / "router_sanity_monotonicity" / "results"
    print("=" * 90)
    print("  PHASE C11.15: ACARA-U ROUTER SANITY & MONOTONICITY DEEP ARTIFACT VERIFICATION")
    print("=" * 90)
    print(f"Artifacts Directory: {artifacts_dir}\n")

    verifier = RouterSanityArtifactVerifier(artifacts_dir)
    all_passed, gates = verifier.run_all_gates()
    cert_path = verifier.generate_certification_report(all_passed, gates)

    print(f"{'Gate':<12} | {'Criterion Name':<50} | {'Status':<8} | {'Details'}")
    print("-" * 105)
    for g in gates:
        status = "PASS" if g["passed"] else "FAIL"
        print(f"{g.get('id', 'GATE'):<12} | {g['name']:<50} | {status:<8} | {g.get('details', '')}")
    print("-" * 105)

    passed_count = sum(1 for g in gates if g["passed"])
    total_count = len(gates)
    print(f"\nVerification Summary: {passed_count}/{total_count} Gates Passed")
    print(f"Certification Record Generated: {cert_path}")

    if all_passed:
        print("\n>>> ALL C11.15 GATES PASSED: ARTIFACT INTEGRITY CERTIFIED AND SEALED <<<")
        sys.exit(0)
    else:
        print("\n>>> VERIFICATION FAILED: ONE OR MORE GATES FAILED <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
