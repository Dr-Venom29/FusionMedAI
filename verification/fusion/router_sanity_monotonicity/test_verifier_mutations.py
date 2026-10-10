"""
FusionMedAI - Phase C11.15: Verifier Mutation Testing Suite
Validates S15-15: Verifies that the verification checks and artifact verifier
reliably detect injected intentional faults across router kernels and artifact files.
"""

import unittest
import shutil
import tempfile
import json
import hashlib
from pathlib import Path
from typing import Dict, Tuple

from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.router_input import RouterInput
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router_sanity.isolated_monotonicity import IsolatedMonotonicityEvaluator
from verification.fusion.router_sanity_monotonicity.verify_router_sanity_artifacts import (
    RouterSanityArtifactVerifier,
    compute_sha256,
)


class InvertedUncertaintyRouter(ACARAUv2Router):
    """Mutated router where uncertainty penalty is inverted into an uncertainty bonus (+ gamma * U)."""
    def compute_logits(self, router_input: RouterInput) -> Tuple[Dict[str, float], Dict[str, float]]:
        raw_logits = {}
        masked_logits = {}
        alpha = float(self.coefficients.alpha)
        beta = float(self.coefficients.beta)
        gamma = float(self.coefficients.gamma)
        eta = float(self.coefficients.eta)

        for mod_name, ch in router_input.channels.items():
            # Injected Fault: + gamma * uncertainty instead of - gamma * uncertainty
            z_i = (
                alpha * float(ch.confidence)
                + beta * float(ch.reliability)
                + gamma * float(ch.uncertainty)
                + eta * float(ch.quality)
            )
            raw_logits[mod_name] = float(z_i)
            masked_logits[mod_name] = float(z_i) if ch.availability else float("-inf")
        return raw_logits, masked_logits


class InvertedConfidenceRouter(ACARAUv2Router):
    """Mutated router where confidence bonus is inverted into a confidence penalty (- alpha * C)."""
    def compute_logits(self, router_input: RouterInput) -> Tuple[Dict[str, float], Dict[str, float]]:
        raw_logits = {}
        masked_logits = {}
        alpha = float(self.coefficients.alpha)
        beta = float(self.coefficients.beta)
        gamma = float(self.coefficients.gamma)
        eta = float(self.coefficients.eta)

        for mod_name, ch in router_input.channels.items():
            # Injected Fault: - alpha * confidence instead of + alpha * confidence
            z_i = (
                -alpha * float(ch.confidence)
                + beta * float(ch.reliability)
                - gamma * float(ch.uncertainty)
                + eta * float(ch.quality)
            )
            raw_logits[mod_name] = float(z_i)
            masked_logits[mod_name] = float(z_i) if ch.availability else float("-inf")
        return raw_logits, masked_logits


def _build_valid_baseline_fixtures(target_dir: Path) -> None:
    """Constructs minimal valid JSON artifacts and matching freeze_manifest in target_dir."""
    protocol = {
        "phase": "C11.15",
        "title": "Residual Router Sanity & Monotonicity Analysis",
        "frozen_configuration": {
            "coefficients": {"alpha": 1.0, "beta": 1.5, "gamma": 1.0, "eta": 0.5},
            "delta": 0.10,
            "benchmark_cohort_n": 500,
            "benchmark_cohort_seed": 115,
        },
        "numerical_tolerances": {
            "float_tolerance": 1e-7,
            "strict_monotonic_eps": 1e-9,
            "shift_invariance_tolerance": 1e-12,
        },
        "active_regimes": ["R", "F", "C", "RF", "RC", "FC", "RFC"],
        "all_regimes": ["EMPTY", "R", "F", "C", "RF", "RC", "FC", "RFC"],
    }
    with open(target_dir / "protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    test_results = {
        "phase": "C11.15",
        "configuration_evaluations": {
            "criterion_id": "S15-01",
            "passed": True,
            "matches_expected": True,
            "coefficients": {"alpha": 1.0, "beta": 1.5, "gamma": 1.0, "eta": 0.5},
            "delta": 0.10,
        },
        "monotonicity_evaluations": {
            "confidence": {"status": "PASS", "passed_trials": 144, "total_trials": 144},
            "reliability": {"status": "PASS", "passed_trials": 144, "total_trials": 144},
            "uncertainty": {"status": "PASS", "passed_trials": 144, "total_trials": 144},
            "quality": {"status": "PASS", "passed_trials": 144, "total_trials": 144},
        },
        "invariants_evaluations": {
            "simplex_conservation": {"passed": True, "max_sum_deviation": 0.0, "min_weight_observed": 0.0, "violation_count": 0},
            "hard_availability_masking": {"passed": True, "violation_count": 0, "total_evaluations": 144},
            "softmax_shift_invariance": {
                "passed": True,
                "production_route_shift_passed": True,
                "reference_kernel_shift_passed": True,
                "max_shift_deviation": 0.0,
                "violation_count": 0,
            },
            "modality_permutation_invariance": {
                "passed": True,
                "reference_kernel_perm_passed": True,
                "max_permutation_deviation": 0.0,
                "violation_count": 0,
            },
            "numerical_stability": {
                "passed": True,
                "production_route_boundary_passed": True,
                "reference_kernel_extreme_logits_passed": True,
                "extreme_kernel_passed": True,
            },
        },
        "edge_cases_evaluations": {
            "input_validation_rejection": {"all_invalid_inputs_rejected": True, "passed_cases": 19, "total_cases": 19},
            "empty_modality_handling": {"passed": True, "status": "NO_MODALITY_AVAILABLE", "num_active": 0},
            "masked_value_invariance": {"passed": True, "max_active_weight_dev": 0.0, "violation_count": 0},
        },
        "pipeline_evaluations": {
            "decoupling_verification": {
                "passed": True,
                "weight_delta": 0.1,
                "scenario_a_risk_delta": 0.05,
                "scenario_b_risk_delta": -0.05,
                "scenario_a_dcri_delta": 0.05,
                "scenario_b_dcri_delta": -0.05,
                "delta_matches_frozen": True,
            }
        },
        "regression_evaluations": {"passed": True, "tests_run": 22, "failures": 0, "errors": 0, "skipped": 0},
        "mutation_evaluations": {
            "passed": True,
            "mutations_tested": 14,
            "expected_mutations": 14,
            "executed_test_names": [
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
            ],
            "names_match": True,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
        },
    }
    with open(target_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    scorecard = {
        f"S15-{i:02d}": {
            "name": f"Acceptance Criterion S15-{i:02d}",
            "passed": True,
        }
        for i in range(1, 16)
    }

    summary = {
        "phase": "C11.15",
        "title": "ACARA-U Router Sanity & Monotonicity Certification Summary",
        "evaluation_status": "PASSED",
        "verification_status": "PENDING_INDEPENDENT_VERIFICATION",
        "gates_passed": "15/15",
        "scorecard": scorecard,
    }
    with open(target_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    manifest = {
        "phase": "C11.15",
        "experiment_id": "router_sanity_monotonicity",
        "manifest_status": "GENERATED_PENDING_INDEPENDENT_VERIFIER",
        "artifact_hashes": {
            "protocol.json": compute_sha256(target_dir / "protocol.json"),
            "test_results.json": compute_sha256(target_dir / "test_results.json"),
            "summary.json": compute_sha256(target_dir / "summary.json"),
        },
    }
    with open(target_dir / "freeze_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


class TestVerifierMutations(unittest.TestCase):
    """Verifies that injected algorithmic or artifact faults are caught by verification gates."""

    def setUp(self):
        # Always generate fresh, isolated baseline fixtures in temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="c1115_mutations_"))
        _build_valid_baseline_fixtures(self.temp_dir)

    def tearDown(self):
        # Ensure temporary directory is cleanly removed
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _sync_manifest_hash(self, filename: str) -> None:
        """Helper to recompute hash in temporary manifest so test fails on semantic gate, not hash mismatch."""
        target = self.temp_dir / filename
        manifest_file = self.temp_dir / "freeze_manifest.json"
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["artifact_hashes"][filename] = compute_sha256(target)
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def test_mutation_1_inverted_uncertainty_sign(self):
        """Mutation 1: Inverting uncertainty sign causes monotonicity failure and artifact verifier rejection at Gate S15-04."""
        mutated_router = InvertedUncertaintyRouter()
        evaluator = IsolatedMonotonicityEvaluator(router=mutated_router)
        res = evaluator.evaluate_uncertainty_monotonicity()
        self.assertEqual(res["status"], "FAIL", "Mutation 1 (+ gamma * U) was not detected by evaluator!")
        self.assertLess(res["passed_trials"], res["total_trials"])

        # Inject mutated result into test_results.json and verify rejection by artifact verifier
        results_file = self.temp_dir / "test_results.json"
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["monotonicity_evaluations"]["uncertainty"] = res
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self._sync_manifest_hash("test_results.json")

        verifier = RouterSanityArtifactVerifier(self.temp_dir)
        all_passed, gates = verifier.run_all_gates()
        self.assertFalse(all_passed, "Artifact verifier did not reject Mutation 1!")
        gate_04 = next((g for g in gates if g.get("id") == "S15-04"), None)
        self.assertIsNotNone(gate_04)
        self.assertFalse(gate_04["passed"])

    def test_mutation_2_inverted_confidence_sign(self):
        """Mutation 2: Inverting confidence sign causes monotonicity failure and artifact verifier rejection at Gate S15-02."""
        mutated_router = InvertedConfidenceRouter()
        evaluator = IsolatedMonotonicityEvaluator(router=mutated_router)
        res = evaluator.evaluate_confidence_monotonicity()
        self.assertEqual(res["status"], "FAIL", "Mutation 2 (- alpha * C) was not detected by evaluator!")
        self.assertLess(res["passed_trials"], res["total_trials"])

        # Inject mutated result into test_results.json and verify rejection by artifact verifier
        results_file = self.temp_dir / "test_results.json"
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["monotonicity_evaluations"]["confidence"] = res
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self._sync_manifest_hash("test_results.json")

        verifier = RouterSanityArtifactVerifier(self.temp_dir)
        all_passed, gates = verifier.run_all_gates()
        self.assertFalse(all_passed, "Artifact verifier did not reject Mutation 2!")
        gate_02 = next((g for g in gates if g.get("id") == "S15-02"), None)
        self.assertIsNotNone(gate_02)
        self.assertFalse(gate_02["passed"])

    def test_mutation_3_artifact_inactive_weight_leakage(self):
        """Mutation 3: Mutating test_results.json with hard masking violation is caught by artifact verifier at Gate S15-07."""
        results_file = self.temp_dir / "test_results.json"
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Inject masking failure
        data["invariants_evaluations"]["hard_availability_masking"]["passed"] = False
        data["invariants_evaluations"]["hard_availability_masking"]["violation_count"] = 1
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self._sync_manifest_hash("test_results.json")

        verifier = RouterSanityArtifactVerifier(self.temp_dir)
        all_passed, gates = verifier.run_all_gates()
        
        self.assertFalse(all_passed, "Mutation 3 (masking violation) was not caught by verifier!")
        gate_07 = next((g for g in gates if g.get("id") == "S15-07"), None)
        self.assertIsNotNone(gate_07)
        self.assertFalse(gate_07["passed"])

    def test_mutation_4_artifact_broken_manifest_hash(self):
        """Mutation 4: Altering an artifact hash in manifest causes Gate S15-00 to fail."""
        manifest_file = self.temp_dir / "freeze_manifest.json"
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        # Corrupt hash
        manifest["artifact_hashes"]["summary.json"] = "0" * 64
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        verifier = RouterSanityArtifactVerifier(self.temp_dir)
        all_passed, gates = verifier.run_all_gates()
        
        self.assertFalse(all_passed, "Mutation 4 (corrupt hash) was not caught by verifier!")
        gate_00 = next((g for g in gates if g.get("id") == "S15-00"), None)
        self.assertIsNotNone(gate_00)
        self.assertFalse(gate_00["passed"])

    def test_mutation_5_artifact_corrupt_frozen_coefficients(self):
        """Mutation 5: Altering frozen coefficients in protocol.json causes Gate S15-01 to fail."""
        protocol_file = self.temp_dir / "protocol.json"
        with open(protocol_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Corrupt coefficient
        data["frozen_configuration"]["coefficients"]["alpha"] = 2.0
        with open(protocol_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self._sync_manifest_hash("protocol.json")

        verifier = RouterSanityArtifactVerifier(self.temp_dir)
        all_passed, gates = verifier.run_all_gates()
        
        self.assertFalse(all_passed, "Mutation 5 (corrupt coefficients) was not caught by verifier!")
        gate_01 = next((g for g in gates if g.get("id") == "S15-01"), None)
        self.assertIsNotNone(gate_01)
        self.assertFalse(gate_01["passed"])

    def test_mutation_6_artifact_missing_production_boundary_field(self):
        """Mutation 6: Missing production_route_boundary_passed in test_results.json fails Gate S15-10."""
        results_file = self.temp_dir / "test_results.json"
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Remove production_route_boundary_passed evidence entirely
        if "production_route_boundary_passed" in data["invariants_evaluations"]["numerical_stability"]:
            del data["invariants_evaluations"]["numerical_stability"]["production_route_boundary_passed"]
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self._sync_manifest_hash("test_results.json")

        verifier = RouterSanityArtifactVerifier(self.temp_dir)
        all_passed, gates = verifier.run_all_gates()
        
        self.assertFalse(all_passed, "Mutation 6 (missing boundary field) was not caught by verifier!")
        gate_10 = next((g for g in gates if g.get("id") == "S15-10"), None)
        self.assertIsNotNone(gate_10)
        self.assertFalse(gate_10["passed"])

    def test_mutation_7_artifact_mutation_count_mismatch(self):
        """Mutation 7: Recording fewer mutations tested in test_results.json fails Gate S15-15."""
        results_file = self.temp_dir / "test_results.json"
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data["mutation_evaluations"]["mutations_tested"] = 5
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self._sync_manifest_hash("test_results.json")

        verifier = RouterSanityArtifactVerifier(self.temp_dir)
        all_passed, gates = verifier.run_all_gates()
        
        self.assertFalse(all_passed, "Mutation 7 (mutation count mismatch) was not caught by verifier!")
        gate_15 = next((g for g in gates if g.get("id") == "S15-15"), None)
        self.assertIsNotNone(gate_15)
        self.assertFalse(gate_15["passed"])

    def test_mutation_8_artifact_malformed_numeric_field(self):
        """Mutation 8: Malformed non-numeric field in test_results.json fails gate cleanly without crashing."""
        results_file = self.temp_dir / "test_results.json"
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Corrupt numeric field with non-numeric string
        data["invariants_evaluations"]["simplex_conservation"]["max_sum_deviation"] = "malformed_string"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self._sync_manifest_hash("test_results.json")

        verifier = RouterSanityArtifactVerifier(self.temp_dir)
        all_passed, gates = verifier.run_all_gates()
        
        self.assertFalse(all_passed, "Mutation 8 (malformed numeric field) did not fail verification cleanly!")
        gate_06 = next((g for g in gates if g.get("id") == "S15-06"), None)
        self.assertIsNotNone(gate_06)
        self.assertFalse(gate_06["passed"])

    def test_mutation_9_artifact_summary_status_mismatch(self):
        """Mutation 9: Altering summary.json status to FAILED fails cross-artifact consistency at Gate S15-00."""
        summary_file = self.temp_dir / "summary.json"
        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data["evaluation_status"] = "FAILED"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self._sync_manifest_hash("summary.json")

        verifier = RouterSanityArtifactVerifier(self.temp_dir)
        all_passed, gates = verifier.run_all_gates()
        
        self.assertFalse(all_passed, "Mutation 9 (summary status mismatch) was not caught by verifier!")
        gate_00 = next((g for g in gates if g.get("id") == "S15-00"), None)
        self.assertIsNotNone(gate_00)
        self.assertFalse(gate_00["passed"])

    def test_mutation_10_artifact_summary_phase_mismatch(self):
        """Mutation 10: Altering summary.json phase to C11.14 fails cross-artifact consistency at Gate S15-00."""
        summary_file = self.temp_dir / "summary.json"
        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data["phase"] = "C11.14"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self._sync_manifest_hash("summary.json")

        verifier = RouterSanityArtifactVerifier(self.temp_dir)
        all_passed, gates = verifier.run_all_gates()
        
        self.assertFalse(all_passed, "Mutation 10 (summary phase mismatch) was not caught by verifier!")
        gate_00 = next((g for g in gates if g.get("id") == "S15-00"), None)
        self.assertIsNotNone(gate_00)
        self.assertFalse(gate_00["passed"])

    def test_mutation_11_artifact_gates_passed_count_mismatch(self):
        """Mutation 11: Altering summary.json gates_passed string fails cross-artifact consistency at Gate S15-00."""
        summary_file = self.temp_dir / "summary.json"
        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data["gates_passed"] = "14/15"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self._sync_manifest_hash("summary.json")

        verifier = RouterSanityArtifactVerifier(self.temp_dir)
        all_passed, gates = verifier.run_all_gates()
        
        self.assertFalse(all_passed, "Mutation 11 (gates passed mismatch) was not caught by verifier!")
        gate_00 = next((g for g in gates if g.get("id") == "S15-00"), None)
        self.assertIsNotNone(gate_00)
        self.assertFalse(gate_00["passed"])

    def test_mutation_12_artifact_manifest_hashes_null(self):
        """Mutation 12: Null artifact_hashes object in freeze_manifest.json fails schema validation cleanly."""
        manifest_file = self.temp_dir / "freeze_manifest.json"
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        manifest["artifact_hashes"] = None
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        verifier = RouterSanityArtifactVerifier(self.temp_dir)
        all_passed, gates = verifier.run_all_gates()
        
        self.assertFalse(all_passed, "Mutation 12 (null artifact_hashes) was not caught cleanly!")
        gate_00 = next((g for g in gates if g.get("id") == "S15-00"), None)
        self.assertIsNotNone(gate_00)
        self.assertFalse(gate_00["passed"])

    def test_mutation_13_artifact_malformed_root_array(self):
        """Mutation 13: Protocol JSON whose root is a list fails schema validation cleanly."""
        protocol_file = self.temp_dir / "protocol.json"
        with open(protocol_file, "w", encoding="utf-8") as f:
            json.dump(["invalid", "root", "list"], f, indent=2)

        self._sync_manifest_hash("protocol.json")

        verifier = RouterSanityArtifactVerifier(self.temp_dir)
        all_passed, gates = verifier.run_all_gates()
        
        self.assertFalse(all_passed, "Mutation 13 (root list schema error) was not caught cleanly!")
        gate_00 = next((g for g in gates if g.get("id") == "S15-00"), None)
        self.assertIsNotNone(gate_00)
        self.assertFalse(gate_00["passed"])

    def test_mutation_14_artifact_summary_scorecard_contradiction(self):
        """Mutation 14: Contradiction between summary scorecard and test_results causes Gate S15-00 to fail."""
        summary_file = self.temp_dir / "summary.json"
        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Invert one scorecard criterion pass flag to create a contradiction with test_results
        data["scorecard"]["S15-06"]["passed"] = False
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self._sync_manifest_hash("summary.json")

        verifier = RouterSanityArtifactVerifier(self.temp_dir)
        all_passed, gates = verifier.run_all_gates()
        
        self.assertFalse(all_passed, "Mutation 14 (summary scorecard contradiction) was not caught by verifier!")
        gate_00 = next((g for g in gates if g.get("id") == "S15-00"), None)
        self.assertIsNotNone(gate_00)
        self.assertFalse(gate_00["passed"])


if __name__ == "__main__":
    unittest.main()
