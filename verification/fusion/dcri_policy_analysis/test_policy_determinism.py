"""
FusionMedAI - Phase C11.14: Unit Tests for Policy Determinism and Invariants
"""

import unittest
from pathlib import Path

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.dcri_policy.policy_config import (
    DELTA_FROZEN,
    TAU_1_DEFAULT,
    TAU_2_DEFAULT,
)
from src.fusion.dcri_policy.policy_engine import PolicyEvaluator


class TestPolicyDeterminism(unittest.TestCase):
    """Tests bitwise determinism and paired consistency of policy evaluations."""

    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[3]
        cls.cohort = load_frozen_cohort(repo_root=repo_root, n_packets=500, seed=115)
        cls.evaluator = PolicyEvaluator(delta=DELTA_FROZEN)

    def test_bitwise_determinism_across_runs(self):
        """Validates that repeated runs on the frozen cohort produce bitwise identical policy outcomes and per-packet outputs."""
        states_1 = self.evaluator.precompute_packet_base_state(self.cohort)
        res_1 = self.evaluator.evaluate_policies(states_1, TAU_1_DEFAULT, TAU_2_DEFAULT)

        states_2 = self.evaluator.precompute_packet_base_state(self.cohort)
        res_2 = self.evaluator.evaluate_policies(states_2, TAU_1_DEFAULT, TAU_2_DEFAULT)

        # Aggregate metrics
        self.assertEqual(res_1.policy_a_counts, res_2.policy_a_counts)
        self.assertEqual(res_1.policy_b_counts, res_2.policy_b_counts)
        self.assertEqual(res_1.transition_matrix, res_2.transition_matrix)
        self.assertEqual(res_1.reclassification_count, res_2.reclassification_count)
        self.assertEqual(res_1.downgraded_count, res_2.downgraded_count)
        self.assertEqual(res_1.upgraded_count, res_2.upgraded_count)

        # Exhaustive per-packet comparisons across all 500 packets
        self.assertEqual(len(res_1.per_packet_details), len(res_2.per_packet_details))
        for p1, p2 in zip(res_1.per_packet_details, res_2.per_packet_details):
            self.assertEqual(p1["packet_id"], p2["packet_id"])
            self.assertEqual(p1["r_fusion"], p2["r_fusion"])
            self.assertEqual(p1["dcri"], p2["dcri"])
            self.assertEqual(p1["u_sum"], p2["u_sum"])
            self.assertEqual(p1["action_fused_risk"], p2["action_fused_risk"])
            self.assertEqual(p1["action_dcri"], p2["action_dcri"])
            self.assertEqual(p1["action_shift"], p2["action_shift"])
            self.assertEqual(p1["is_reclassified"], p2["is_reclassified"])

    def test_monotonic_action_assignment(self):
        """Validates that DCRI score is strictly <= R_fusion on all packets and never causes an upgrade."""
        states = self.evaluator.precompute_packet_base_state(self.cohort)
        res = self.evaluator.evaluate_policies(states, TAU_1_DEFAULT, TAU_2_DEFAULT)

        for p in res.per_packet_details:
            self.assertLessEqual(p["dcri"], p["r_fusion"] + 1e-12)
            self.assertLessEqual(p["action_dcri"], p["action_fused_risk"])

    def test_full_serialization_byte_determinism(self):
        """Validates that repeated runs produce byte-for-byte identical canonical JSON serializations."""
        import json
        states_1 = self.evaluator.precompute_packet_base_state(self.cohort)
        res_1 = self.evaluator.evaluate_policies(states_1, TAU_1_DEFAULT, TAU_2_DEFAULT)
        bytes_1 = json.dumps(res_1.to_dict(), sort_keys=True, indent=2).encode("utf-8")

        states_2 = self.evaluator.precompute_packet_base_state(self.cohort)
        res_2 = self.evaluator.evaluate_policies(states_2, TAU_1_DEFAULT, TAU_2_DEFAULT)
        bytes_2 = json.dumps(res_2.to_dict(), sort_keys=True, indent=2).encode("utf-8")

        self.assertEqual(
            bytes_1,
            bytes_2,
            "Canonical policy-result serialization differs across repeated runs",
        )

    def test_full_pipeline_export_byte_determinism(self):
        """Validates that running the complete PolicyAnalysisExperimentRunner pipeline into two separate directories produces byte-identical files across all 5 JSON artifacts and manifest."""
        import tempfile
        import shutil
        from src.fusion.dcri_policy.policy_runner import PolicyAnalysisExperimentRunner

        repo_root = Path(__file__).resolve().parents[3]
        temp_dir_1 = tempfile.mkdtemp()
        temp_dir_2 = tempfile.mkdtemp()
        try:
            runner_1 = PolicyAnalysisExperimentRunner(repo_root=repo_root, output_dir=Path(temp_dir_1))
            runner_1.run_all()

            runner_2 = PolicyAnalysisExperimentRunner(repo_root=repo_root, output_dir=Path(temp_dir_2))
            runner_2.run_all()

            artifact_names = [
                "policy_config.json",
                "threshold_results.json",
                "regime_results.json",
                "robustness_results.json",
                "statistical_results.json",
                "freeze_manifest.json",
            ]

            for name in artifact_names:
                file_1 = Path(temp_dir_1) / name
                file_2 = Path(temp_dir_2) / name
                self.assertTrue(file_1.is_file(), f"{name} missing in run 1")
                self.assertTrue(file_2.is_file(), f"{name} missing in run 2")

                bytes_1 = file_1.read_bytes()
                bytes_2 = file_2.read_bytes()
                self.assertEqual(bytes_1, bytes_2, f"Byte mismatch detected in full pipeline export for {name}")
        finally:
            shutil.rmtree(temp_dir_1, ignore_errors=True)
            shutil.rmtree(temp_dir_2, ignore_errors=True)

    def test_serialization_mutation_detected(self):
        """Validates that any synthetic alteration in canonical serialization is caught by assertion."""
        import json
        states = self.evaluator.precompute_packet_base_state(self.cohort)
        res = self.evaluator.evaluate_policies(states, TAU_1_DEFAULT, TAU_2_DEFAULT)
        d1 = res.to_dict()
        d2 = res.to_dict()
        d2["reclassifications"]["total_reclassified_count"] += 1
        bytes_1 = json.dumps(d1, sort_keys=True, indent=2).encode("utf-8")
        bytes_2 = json.dumps(d2, sort_keys=True, indent=2).encode("utf-8")
        with self.assertRaises(AssertionError):
            self.assertEqual(bytes_1, bytes_2)


if __name__ == "__main__":
    unittest.main()
