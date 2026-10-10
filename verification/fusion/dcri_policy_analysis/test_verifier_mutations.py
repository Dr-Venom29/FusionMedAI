"""
FusionMedAI - Phase C11.14: Verifier Mutation Testing Suite
Tests that deliberate mutations to manifests, checksums, file sizes, configuration fields,
summary insights, artifact list lengths, schema keys, and statistical metrics correctly
and specifically trigger failures in Gate 12, with manifest checksums synchronized
where appropriate to isolate deep semantic and schema reconciliation checks.
"""

import unittest
import json
import shutil
import tempfile
from pathlib import Path

from verification.fusion.dcri_policy_analysis.verify_policy_artifacts import (
    PolicyVerificationSuite,
    get_repo_root,
    compute_sha256_file,
)


class TestVerifierMutations(unittest.TestCase):
    """
    Validates failure detection mechanics of Gate 12 under synthetic mutations.
    """

    @classmethod
    def setUpClass(cls):
        cls.repo_root = get_repo_root()
        cls.orig_results_dir = cls.repo_root / "experiments" / "fusion" / "dcri_policy_analysis" / "results"

    def setUp(self):
        # Create a fresh isolated copy of the results directory in a temp dir
        self.temp_dir = tempfile.mkdtemp()
        self.test_results_dir = Path(self.temp_dir) / "results"
        shutil.copytree(self.orig_results_dir, self.test_results_dir)

        # Instantiate suite pointed to temp results
        self.suite = PolicyVerificationSuite(repo_root=self.repo_root)
        self.suite.results_dir = self.test_results_dir

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def update_manifest_for_file(self, filename: str):
        """
        Updates the test freeze_manifest.json with the mutated file's new size and SHA-256 hash.
        This ensures manifest integrity checks pass, isolating the field-level semantic reconciliation.
        """
        manifest_path = self.test_results_dir / "freeze_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        filepath = self.test_results_dir / filename
        manifest["artifacts"][filename]["size_bytes"] = filepath.stat().st_size
        manifest["artifacts"][filename]["sha256"] = compute_sha256_file(filepath)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def test_unmodified_passes(self):
        """Validates that the unmodified copy passes Gate 12."""
        self.suite.verify_gate_12()

    def test_mutation_manifest_file_size_fails(self):
        """Validates that altering a recorded file size in the manifest causes Gate 12 to fail at Part 1."""
        manifest_path = self.test_results_dir / "freeze_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        manifest["artifacts"]["policy_config.json"]["size_bytes"] += 1
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        with self.assertRaises(ValueError) as ctx:
            self.suite.verify_gate_12()
        self.assertIn("File size mismatch", str(ctx.exception))

    def test_mutation_manifest_hash_fails(self):
        """Validates that altering an artifact checksum in the manifest causes Gate 12 to fail at Part 1."""
        manifest_path = self.test_results_dir / "freeze_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        manifest["artifacts"]["policy_config.json"]["sha256"] = "0" * 64
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        with self.assertRaises(ValueError) as ctx:
            self.suite.verify_gate_12()
        self.assertIn("SHA-256 hash mismatch", str(ctx.exception))

    def test_mutation_config_field_fails(self):
        """Validates that altering a frozen parameter causes Gate 12 to fail at field reconciliation even with valid manifest."""
        config_path = self.test_results_dir / "policy_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config["frozen_parameters"]["delta"] = 0.20
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Update manifest to isolate semantic verification
        self.update_manifest_for_file("policy_config.json")

        with self.assertRaises(AssertionError) as ctx:
            self.suite.verify_gate_12()
        self.assertIn("delta mismatch", str(ctx.exception))

    def test_mutation_schema_extra_key_fails(self):
        """Validates that injecting an extra key into nested schema causes Gate 12 to fail even with valid manifest."""
        config_path = self.test_results_dir / "policy_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config["frozen_parameters"]["extra_unauthorized_key"] = "leak"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        self.update_manifest_for_file("policy_config.json")

        with self.assertRaises(AssertionError) as ctx:
            self.suite.verify_gate_12()
        self.assertIn("frozen_parameters schema mismatch", str(ctx.exception))

    def test_mutation_summary_insights_fails(self):
        """Validates that altering summary_insights causes Gate 12 to fail at field reconciliation even with valid manifest."""
        thresh_path = self.test_results_dir / "threshold_results.json"
        with open(thresh_path, "r", encoding="utf-8") as f:
            thresh = json.load(f)

        thresh["summary_insights"]["nominal_reclassification_rate"] = 0.50
        with open(thresh_path, "w", encoding="utf-8") as f:
            json.dump(thresh, f, indent=2)

        # Update manifest to isolate semantic verification
        self.update_manifest_for_file("threshold_results.json")

        with self.assertRaises(AssertionError) as ctx:
            self.suite.verify_gate_12()
        self.assertIn("summary_insights reclass rate mismatch", str(ctx.exception))

    def test_mutation_list_length_fails(self):
        """Validates that altering threshold_sweep_results list length causes Gate 12 to fail even with valid manifest."""
        thresh_path = self.test_results_dir / "threshold_results.json"
        with open(thresh_path, "r", encoding="utf-8") as f:
            thresh = json.load(f)

        thresh["threshold_sweep_results"].pop()  # Drop 1 element
        with open(thresh_path, "w", encoding="utf-8") as f:
            json.dump(thresh, f, indent=2)

        # Update manifest to isolate semantic verification
        self.update_manifest_for_file("threshold_results.json")

        with self.assertRaises(AssertionError) as ctx:
            self.suite.verify_gate_12()
        self.assertIn("threshold sweep count != 25", str(ctx.exception))

    def test_mutation_regime_mean_dcri_fails(self):
        """Validates that altering regime mean DCRI causes Gate 12 to fail even with valid manifest."""
        reg_path = self.test_results_dir / "regime_results.json"
        with open(reg_path, "r", encoding="utf-8") as f:
            reg = json.load(f)

        reg["regime_results"]["R"]["mean_dcri"] += 0.05
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)

        self.update_manifest_for_file("regime_results.json")

        with self.assertRaises(AssertionError) as ctx:
            self.suite.verify_gate_12()
        self.assertIn("Regime R mean_dcri mismatch", str(ctx.exception))

    def test_mutation_statistical_wilson_ci_fails(self):
        """Validates that altering statistical Wilson CI causes Gate 12 to fail even with valid manifest."""
        stat_path = self.test_results_dir / "statistical_results.json"
        with open(stat_path, "r", encoding="utf-8") as f:
            stat = json.load(f)

        stat["statistical_summary"]["reclassifications"]["wilson_95_ci"] = [0.10, 0.30]
        with open(stat_path, "w", encoding="utf-8") as f:
            json.dump(stat, f, indent=2)

        self.update_manifest_for_file("statistical_results.json")

        with self.assertRaises(AssertionError) as ctx:
            self.suite.verify_gate_12()
        self.assertIn("reclass wilson CI mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
