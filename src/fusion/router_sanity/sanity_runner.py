"""
FusionMedAI - Phase C11.15: Router Sanity & Monotonicity Runner
Orchestrates complete evaluation across:
1. Isolated Single-Input Monotonicity (Confidence, Reliability, Uncertainty, Quality)
2. Foundational Mathematical Invariants (Simplex, Masking, Shift Invariance, Permutation, Stability)
3. Degenerate & Edge Case Handling (EMPTY Fail-Closed, Masked-Value Corruption Invariance, Validation)
4. Pipeline Contract Verification (Stage Separation & Decoupling with production DCRI downstream)
5. Executed Regression Suite Verification (C11.14 Suite Execution with exact count assertion)
6. Executed Verifier Mutation Suite (Fault Injections Detection with exact count assertion)
7. Generation of 4 Frozen JSON Artifacts + Freeze Manifest (with explicit lifecycle status)
"""

import json
import hashlib
import io
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router_sanity.sanity_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    ROUTER_COEFFS_REF,
    DELTA_FROZEN,
    BENCHMARK_COHORT_N,
    BENCHMARK_COHORT_SEED,
    FLOAT_TOLERANCE,
    STRICT_MONOTONIC_EPS,
    SHIFT_INVARIANCE_TOLERANCE,
    ACTIVE_REGIMES,
    ALL_REGIMES,
    INTERIOR_GRID,
    BOUNDARY_GRID,
    PERTURBATION_STEPS,
    ACCEPTANCE_CRITERIA,
)
from src.fusion.router_sanity.isolated_monotonicity import IsolatedMonotonicityEvaluator
from src.fusion.router_sanity.invariant_checker import RouterInvariantChecker
from src.fusion.router_sanity.edge_case_evaluator import EdgeCaseEvaluator
from src.fusion.router_sanity.pipeline_contract_checker import PipelineContractChecker


def to_serializable(val: Any) -> Any:
    """Recursively converts numpy and non-standard types to pure JSON serializable objects."""
    import numpy as np
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    elif isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, (np.floating, float)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return [to_serializable(x) for x in val.tolist()]
    elif isinstance(val, dict):
        return {str(k): to_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [to_serializable(x) for x in val]
    return val


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


class RouterSanityRunner:
    """Orchestrates Phase C11.15 execution and artifact generation."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (
            Path(__file__).resolve().parents[3]
            / "experiments"
            / "fusion"
            / "router_sanity_monotonicity"
            / "results"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.monotonicity_evaluator = IsolatedMonotonicityEvaluator()
        self.invariant_checker = RouterInvariantChecker()
        self.edge_case_evaluator = EdgeCaseEvaluator()
        self.pipeline_checker = PipelineContractChecker()

    def check_reference_configuration(self) -> Dict[str, Any]:
        """Validates that frozen Theta_0 and Delta match sealed parameters."""
        expected_coeffs = {"alpha": 1.0, "beta": 1.5, "gamma": 1.0, "eta": 0.5}
        expected_delta = 0.10
        coeffs_match = (ROUTER_COEFFS_REF == expected_coeffs)
        delta_match = (DELTA_FROZEN == expected_delta)
        instance_coeffs = RouterCoefficients(
            alpha=ALPHA_REF,
            beta=BETA_REF,
            gamma=GAMMA_REF,
            eta=ETA_REF,
        ).to_dict()
        instance_match = (instance_coeffs == expected_coeffs)

        passed = coeffs_match and delta_match and instance_match
        return {
            "criterion_id": "S15-01",
            "name": ACCEPTANCE_CRITERIA["S15-01"],
            "passed": passed,
            "coefficients": ROUTER_COEFFS_REF,
            "delta": DELTA_FROZEN,
            "instance_coefficients": instance_coeffs,
            "matches_expected": passed,
        }

    def execute_regression_suite(self) -> Dict[str, Any]:
        """Programmatically executes the C11.14 test suite to verify regression invariance with exact counts."""
        from verification.fusion.dcri_policy_analysis.test_policy_determinism import TestPolicyDeterminism
        from verification.fusion.dcri_policy_analysis.test_policy_thresholds import TestPolicyThresholds
        from verification.fusion.dcri_policy_analysis.test_regime_invariance import TestRegimePolicyInvariance
        from verification.fusion.dcri_policy_analysis.test_verifier_mutations import TestVerifierMutations as C14Mutations
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for tc in [TestPolicyDeterminism, TestPolicyThresholds, TestRegimePolicyInvariance, C14Mutations]:
            suite.addTests(loader.loadTestsFromTestCase(tc))
        
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=0)
        result = runner.run(suite)
        
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        expected_tests = 22
        
        passed = (tests_run == expected_tests) and (failures == 0) and (errors == 0) and (skipped == 0)

        return {
            "criterion_id": "S15-14",
            "passed": passed,
            "tests_run": tests_run,
            "expected_tests": expected_tests,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "suite_dir": "verification/fusion/dcri_policy_analysis",
        }

    def execute_mutation_suite(self) -> Dict[str, Any]:
        """Programmatically executes the C11.15 mutation testing suite to verify fault detection with exact named tests and counts."""
        from verification.fusion.router_sanity_monotonicity.test_verifier_mutations import TestVerifierMutations as C15Mutations
        
        expected_test_names = {
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
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        suite.addTests(loader.loadTestsFromTestCase(C15Mutations))
        
        executed_test_names = {test._testMethodName for test in suite}
        names_match = (executed_test_names == expected_test_names)
        
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=0)
        result = runner.run(suite)
        
        mutations_tested = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        expected_mutations = len(expected_test_names)
        
        passed = (
            names_match
            and (mutations_tested == expected_mutations)
            and (failures == 0)
            and (errors == 0)
            and (skipped == 0)
        )

        return {
            "criterion_id": "S15-15",
            "passed": passed,
            "mutations_tested": mutations_tested,
            "expected_mutations": expected_mutations,
            "executed_test_names": sorted(list(executed_test_names)),
            "names_match": names_match,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "suite_file": "verification/fusion/router_sanity_monotonicity/test_verifier_mutations.py",
        }

    def _atomic_write_json(self, target_path: Path, data: Any) -> None:
        """Atomically writes pure JSON with allow_nan=False to prevent non-finite token leakage."""
        temp_path = target_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(to_serializable(data), f, indent=2, allow_nan=False)
        temp_path.replace(target_path)

    def run_all(self) -> Dict[str, Any]:
        """Runs the complete suite and builds all result dictionaries."""
        # 1. Reference configuration check
        config_results = self.check_reference_configuration()

        # 2. Monotonicity evaluations
        mono_results = self.monotonicity_evaluator.run_all()

        # 3. Invariant evaluations
        inv_results = self.invariant_checker.run_all_invariant_checks()

        # 4. Edge case evaluations
        edge_results = self.edge_case_evaluator.run_all_edge_case_checks()

        # 5. Pipeline contract evaluations
        pipe_results = self.pipeline_checker.run_all_pipeline_checks()

        # 6. Executed Regression Suite
        regression_results = self.execute_regression_suite()

        # 7. Executed Mutation Suite
        mutation_results = self.execute_mutation_suite()

        # Strict S15-10 calculation matching verifier contract
        stab = inv_results["numerical_stability"]
        input_val = edge_results["input_validation_rejection"]
        s15_10_passed = bool(
            stab.get("passed") is True
            and stab.get("production_route_boundary_passed") is True
            and stab.get("reference_kernel_extreme_logits_passed") is True
            and input_val.get("all_invalid_inputs_rejected") is True
            and input_val.get("passed_cases") == 19
            and input_val.get("total_cases") == 19
        )

        # 8. Build Scorecard for S15-01 through S15-15
        scorecard = {
            "S15-01": {
                "name": ACCEPTANCE_CRITERIA["S15-01"],
                "passed": config_results["passed"],
                "details": config_results,
            },
            "S15-02": {
                "name": ACCEPTANCE_CRITERIA["S15-02"],
                "passed": mono_results["confidence"]["status"] == "PASS",
                "details": {
                    "total_trials": mono_results["confidence"]["total_trials"],
                    "passed_trials": mono_results["confidence"]["passed_trials"],
                    "status": mono_results["confidence"]["status"],
                },
            },
            "S15-03": {
                "name": ACCEPTANCE_CRITERIA["S15-03"],
                "passed": mono_results["reliability"]["status"] == "PASS",
                "details": {
                    "total_trials": mono_results["reliability"]["total_trials"],
                    "passed_trials": mono_results["reliability"]["passed_trials"],
                    "status": mono_results["reliability"]["status"],
                },
            },
            "S15-04": {
                "name": ACCEPTANCE_CRITERIA["S15-04"],
                "passed": mono_results["uncertainty"]["status"] == "PASS",
                "details": {
                    "total_trials": mono_results["uncertainty"]["total_trials"],
                    "passed_trials": mono_results["uncertainty"]["passed_trials"],
                    "status": mono_results["uncertainty"]["status"],
                },
            },
            "S15-05": {
                "name": ACCEPTANCE_CRITERIA["S15-05"],
                "passed": mono_results["quality"]["status"] == "PASS",
                "details": {
                    "total_trials": mono_results["quality"]["total_trials"],
                    "passed_trials": mono_results["quality"]["passed_trials"],
                    "status": mono_results["quality"]["status"],
                },
            },
            "S15-06": {
                "name": ACCEPTANCE_CRITERIA["S15-06"],
                "passed": inv_results["simplex_conservation"]["passed"],
                "details": {
                    "max_sum_deviation": inv_results["simplex_conservation"]["max_sum_deviation"],
                    "min_weight_observed": inv_results["simplex_conservation"]["min_weight_observed"],
                },
            },
            "S15-07": {
                "name": ACCEPTANCE_CRITERIA["S15-07"],
                "passed": inv_results["hard_availability_masking"]["passed"],
                "details": {"violations": inv_results["hard_availability_masking"]["violation_count"]},
            },
            "S15-08": {
                "name": ACCEPTANCE_CRITERIA["S15-08"],
                "passed": inv_results["softmax_shift_invariance"]["passed"],
                "details": {"max_shift_deviation": inv_results["softmax_shift_invariance"]["max_shift_deviation"]},
            },
            "S15-09": {
                "name": ACCEPTANCE_CRITERIA["S15-09"],
                "passed": inv_results["modality_permutation_invariance"]["passed"],
                "details": {"max_perm_deviation": inv_results["modality_permutation_invariance"]["max_permutation_deviation"]},
            },
            "S15-10": {
                "name": ACCEPTANCE_CRITERIA["S15-10"],
                "passed": s15_10_passed,
                "details": {
                    "numerical_stability_passed": stab.get("passed"),
                    "production_route_boundary_passed": stab.get("production_route_boundary_passed"),
                    "extreme_kernel_passed": stab.get("reference_kernel_extreme_logits_passed"),
                    "input_validation_passed": input_val.get("all_invalid_inputs_rejected"),
                    "input_validation_cases_passed": f"{input_val.get('passed_cases')}/{input_val.get('total_cases')}",
                },
            },
            "S15-11": {
                "name": ACCEPTANCE_CRITERIA["S15-11"],
                "passed": edge_results["empty_modality_handling"]["passed"],
                "details": {"status": edge_results["empty_modality_handling"]["status"]},
            },
            "S15-12": {
                "name": ACCEPTANCE_CRITERIA["S15-12"],
                "passed": edge_results["masked_value_invariance"]["passed"],
                "details": {"max_dev": edge_results["masked_value_invariance"]["max_active_weight_dev"]},
            },
            "S15-13": {
                "name": ACCEPTANCE_CRITERIA["S15-13"],
                "passed": pipe_results["decoupling_verification"]["passed"],
                "details": pipe_results["decoupling_verification"],
            },
            "S15-14": {
                "name": ACCEPTANCE_CRITERIA["S15-14"],
                "passed": regression_results["passed"],
                "details": regression_results,
            },
            "S15-15": {
                "name": ACCEPTANCE_CRITERIA["S15-15"],
                "passed": mutation_results["passed"],
                "details": mutation_results,
            },
        }

        all_criteria_passed = (
            all(item["passed"] for item in scorecard.values())
            and edge_results["all_edge_cases_passed"]
            and inv_results["all_invariants_passed"]
            and mono_results["all_passed"]
            and config_results["passed"]
            and regression_results["passed"]
            and mutation_results["passed"]
        )

        return {
            "all_passed": all_criteria_passed,
            "scorecard": scorecard,
            "configuration": config_results,
            "monotonicity": mono_results,
            "invariants": inv_results,
            "edge_cases": edge_results,
            "pipeline": pipe_results,
            "regression": regression_results,
            "mutation": mutation_results,
        }

    def generate_and_save_artifacts(self) -> Dict[str, Path]:
        """Generates all 4 JSON artifacts atomically and writes them to disk."""
        evaluation_data = self.run_all()
        timestamp = datetime.now(timezone.utc).isoformat()

        # 1. Protocol JSON
        protocol = {
            "phase": "C11.15",
            "title": "Residual Router Sanity & Monotonicity Analysis",
            "generated_at": timestamp,
            "frozen_configuration": {
                "coefficients": ROUTER_COEFFS_REF,
                "delta": DELTA_FROZEN,
                "benchmark_cohort_n": BENCHMARK_COHORT_N,
                "benchmark_cohort_seed": BENCHMARK_COHORT_SEED,
            },
            "numerical_tolerances": {
                "float_tolerance": FLOAT_TOLERANCE,
                "strict_monotonic_eps": STRICT_MONOTONIC_EPS,
                "shift_invariance_tolerance": SHIFT_INVARIANCE_TOLERANCE,
            },
            "active_regimes": list(ACTIVE_REGIMES),
            "all_regimes": list(ALL_REGIMES),
            "test_grids": {
                "interior_grid": list(INTERIOR_GRID),
                "boundary_grid": list(BOUNDARY_GRID),
                "perturbation_steps": list(PERTURBATION_STEPS),
            },
            "acceptance_criteria": ACCEPTANCE_CRITERIA,
        }
        protocol_path = self.output_dir / "protocol.json"
        self._atomic_write_json(protocol_path, protocol)

        # 2. Test Results JSON
        test_results = {
            "phase": "C11.15",
            "generated_at": timestamp,
            "configuration_evaluations": evaluation_data["configuration"],
            "monotonicity_evaluations": evaluation_data["monotonicity"],
            "invariants_evaluations": evaluation_data["invariants"],
            "edge_cases_evaluations": evaluation_data["edge_cases"],
            "pipeline_evaluations": evaluation_data["pipeline"],
            "regression_evaluations": evaluation_data["regression"],
            "mutation_evaluations": evaluation_data["mutation"],
        }
        test_results_path = self.output_dir / "test_results.json"
        self._atomic_write_json(test_results_path, test_results)

        # 3. Summary JSON (distinguishes generation status from independent verification)
        summary = {
            "phase": "C11.15",
            "title": "ACARA-U Router Sanity & Monotonicity Certification Summary",
            "generated_at": timestamp,
            "evaluation_status": "PASSED" if evaluation_data["all_passed"] else "FAILED",
            "verification_status": "PENDING_INDEPENDENT_VERIFICATION",
            "gates_passed": f"{sum(1 for c in evaluation_data['scorecard'].values() if c['passed'])}/{len(evaluation_data['scorecard'])}",
            "scorecard": evaluation_data["scorecard"],
            "key_metrics": {
                "configuration_verified": evaluation_data["configuration"]["passed"],
                "confidence_monotonic_tests_passed": evaluation_data["monotonicity"]["confidence"]["status"] == "PASS",
                "reliability_monotonic_tests_passed": evaluation_data["monotonicity"]["reliability"]["status"] == "PASS",
                "uncertainty_monotonic_tests_passed": evaluation_data["monotonicity"]["uncertainty"]["status"] == "PASS",
                "quality_monotonic_tests_passed": evaluation_data["monotonicity"]["quality"]["status"] == "PASS",
                "simplex_conservation_passed": evaluation_data["invariants"]["simplex_conservation"]["passed"],
                "hard_masking_passed": evaluation_data["invariants"]["hard_availability_masking"]["passed"],
                "shift_invariance_passed": evaluation_data["invariants"]["softmax_shift_invariance"]["passed"],
                "permutation_invariance_passed": evaluation_data["invariants"]["modality_permutation_invariance"]["passed"],
                "numerical_stability_passed": evaluation_data["invariants"]["numerical_stability"]["passed"],
                "empty_handling_passed": evaluation_data["edge_cases"]["empty_modality_handling"]["passed"],
                "masked_invariance_passed": evaluation_data["edge_cases"]["masked_value_invariance"]["passed"],
                "pipeline_decoupling_passed": evaluation_data["pipeline"]["decoupling_verification"]["passed"],
                "regression_invariance_passed": evaluation_data["regression"]["passed"],
                "mutation_detection_passed": evaluation_data["mutation"]["passed"],
            },
        }
        summary_path = self.output_dir / "summary.json"
        self._atomic_write_json(summary_path, summary)

        # 4. Freeze Manifest JSON
        hashes = {
            "protocol.json": compute_sha256(protocol_path),
            "test_results.json": compute_sha256(test_results_path),
            "summary.json": compute_sha256(summary_path),
        }

        manifest = {
            "phase": "C11.15",
            "experiment_id": "router_sanity_monotonicity",
            "frozen_at": timestamp,
            "manifest_status": "GENERATED_PENDING_INDEPENDENT_VERIFIER",
            "artifact_hashes": hashes,
            "verification_envelope": {
                "total_criteria": len(ACCEPTANCE_CRITERIA),
                "passed_criteria": sum(1 for c in evaluation_data["scorecard"].values() if c["passed"]),
                "coefficients": ROUTER_COEFFS_REF,
                "delta": DELTA_FROZEN,
            },
        }
        manifest_path = self.output_dir / "freeze_manifest.json"
        self._atomic_write_json(manifest_path, manifest)

        return {
            "protocol": protocol_path,
            "test_results": test_results_path,
            "summary": summary_path,
            "manifest": manifest_path,
        }


if __name__ == "__main__":
    runner = RouterSanityRunner()
    paths = runner.generate_and_save_artifacts()
    print("Generated C11.15 Artifacts:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
