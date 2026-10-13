"""
FusionMedAI - Phase C11: Canonical Redesigned Cohort Policy S Verification
Verifies that canonical_safeguarded_policy_results.json accurately reproduces
safeguarded decision-policy metrics directly from the canonical manifest
(01_canonical_redesigned_packet_manifest.json, SHA-256 cf877d74e306...) with zero divergence.
"""

import unittest
import json
from pathlib import Path

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.dcri_policy.policy_config import (
    DELTA_FROZEN,
    TAU_1_DEFAULT,
    TAU_2_DEFAULT,
    AMBIGUITY_U_SUM_THRESHOLD,
    AMBIGUITY_MIN_QUALITY_THRESHOLD,
)
from src.fusion.dcri_policy.policy_engine import PolicyEvaluator
from verification.fusion.dcri_policy_analysis.verify_policy_artifacts import compute_sha256_file


class TestCanonicalSafeguardPolicy(unittest.TestCase):
    """
    Validates canonical cohort Policy S artifact integrity and exact recomputation.
    """

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[3]
        cls.canonical_dir = cls.repo_root / "experiments" / "fusion" / "canonical_redesigned_rerun"
        cls.manifest_path = cls.canonical_dir / "01_canonical_redesigned_packet_manifest.json"
        cls.results_path = cls.canonical_dir / "canonical_safeguarded_policy_results.json"

        cls.manifest_sha = compute_sha256_file(cls.manifest_path)
        cls.cohort = load_frozen_cohort(
            repo_root=cls.repo_root,
            n_packets=500,
            seed=115,
            manifest_path=cls.manifest_path,
        )
        cls.evaluator = PolicyEvaluator(delta=DELTA_FROZEN)
        cls.base_states = cls.evaluator.precompute_packet_base_state(cls.cohort)

    def test_canonical_manifest_sha_integrity(self):
        """Verifies the canonical manifest hash matches expected cf877d74..."""
        expected_sha = "cf877d74e306d6dbc3ac7d9e2485b1c46da4cd64a8e35a82fe68274854dc4cd4"
        self.assertEqual(self.manifest_sha, expected_sha)

    def test_canonical_artifact_recomputation(self):
        """Recomputes Policy S on canonical cohort and reconciles every field in the artifact."""
        with open(self.results_path, "r", encoding="utf-8") as f:
            artifact = json.load(f)

        self.assertEqual(artifact["manifest_sha256"], self.manifest_sha)
        se = artifact["safeguarded_evaluation"]

        # Run independent in-memory evaluation
        recomputed = self.evaluator.evaluate_safeguarded_policy(
            self.base_states,
            tau_1=TAU_1_DEFAULT,
            tau_2=TAU_2_DEFAULT,
            u_sum_threshold=AMBIGUITY_U_SUM_THRESHOLD,
            min_quality_threshold=AMBIGUITY_MIN_QUALITY_THRESHOLD,
        )

        # 1. Configuration fields
        self.assertEqual(se["tau_1"], TAU_1_DEFAULT)
        self.assertEqual(se["tau_2"], TAU_2_DEFAULT)
        self.assertEqual(se["delta"], DELTA_FROZEN)
        self.assertEqual(se["u_sum_threshold"], AMBIGUITY_U_SUM_THRESHOLD)
        self.assertEqual(se["min_quality_threshold"], AMBIGUITY_MIN_QUALITY_THRESHOLD)
        self.assertEqual(se["n_packets"], 500)

        # 2. Base Risk (Policy B)
        self.assertEqual(se["policy_b_base_risk"]["counts"], recomputed.policy_b_counts)
        self.assertEqual(recomputed.policy_b_counts, {
            "routine_review": 192,
            "additional_assessment": 225,
            "escalation": 83,
        })
        self.assertEqual(se["policy_b_base_risk"]["percentages"], {
            "routine_review": 38.4,
            "additional_assessment": 45.0,
            "escalation": 16.6,
        })

        # 3. Unconstrained DCRI (Policy A)
        self.assertEqual(se["policy_a_unconstrained_dcri"]["counts"], recomputed.policy_a_counts)
        self.assertEqual(recomputed.policy_a_counts, {
            "routine_review": 262,
            "additional_assessment": 181,
            "escalation": 57,
        })
        self.assertEqual(se["policy_a_unconstrained_dcri"]["percentages"], {
            "routine_review": 52.4,
            "additional_assessment": 36.2,
            "escalation": 11.4,
        })

        # 4. Safeguarded DCRI (Policy S)
        self.assertEqual(se["policy_s_safeguarded_dcri"]["counts"], recomputed.policy_s_counts)
        self.assertEqual(recomputed.policy_s_counts, {
            "routine_review": 193,
            "additional_assessment": 225,
            "escalation": 82,
        })
        self.assertEqual(se["policy_s_safeguarded_dcri"]["percentages"], {
            "routine_review": 38.6,
            "additional_assessment": 45.0,
            "escalation": 16.4,
        })

        # 5. Safeguard Metrics and Count Conservation
        sm = se["safeguard_metrics"]
        self.assertEqual(sm["attempted_downgrades_count"], 96)
        self.assertAlmostEqual(sm["attempted_downgrades_rate"], 96 / 500, places=6)
        self.assertEqual(sm["blocked_downgrades_count"], 94)
        self.assertAlmostEqual(sm["blocked_downgrades_rate"], 94 / 500, places=6)
        self.assertEqual(sm["permitted_downgrades_count"], 2)
        self.assertAlmostEqual(sm["permitted_downgrades_rate"], 2 / 500, places=6)
        self.assertEqual(sm["attempted_downgrades_count"], sm["blocked_downgrades_count"] + sm["permitted_downgrades_count"])

        self.assertAlmostEqual(sm["blocked_ratio_vs_attempted"], 94 / 96, places=5)
        self.assertEqual(sm["ambiguous_flags_count"], 432)
        self.assertAlmostEqual(sm["ambiguous_flags_rate"], 432 / 500, places=6)
        self.assertEqual(sm["reclass_count_s_vs_b"], 2)
        self.assertAlmostEqual(sm["reclass_rate_s_vs_b"], 2 / 500, places=6)
        self.assertEqual(sm["downgrades_s_vs_b"], 2)
        self.assertAlmostEqual(sm["downgrades_rate_s_vs_b"], 2 / 500, places=6)


    def test_canonical_packet_stream_digest(self):
        """Verifies cryptographic content digest across all 500 canonical packets."""
        import hashlib
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        h = hashlib.sha256()
        for pkt in manifest_data.get("sample_packets", []):
            h.update(json.dumps(pkt, sort_keys=True).encode("utf-8"))
        computed_stream_digest = h.hexdigest()
        expected_digest = "65d7c53c3238128d34ce3efe4b121d656c9450df7b0eed48fc8cdb4a09090ea2"
        self.assertEqual(computed_stream_digest, expected_digest)

    def test_canonical_mathematical_invariants(self):
        """
        Verifies core mathematical invariants on the canonical cohort:
        1. Simplex preservation: sum(w_i) == 1.0 within float tolerance.
        2. Exact analytical DCRI formula: DCRI == R_fusion - delta * U_sum (< 1e-14).
        3. Monotonic non-inflation: DCRI <= R_fusion on every packet (zero upgrades).
        4. Negative score safety: any DCRI < 0 maps strictly to Tier 0 (Routine Review).
        """
        for state in self.base_states:
            # 1. Simplex
            sum_w = sum(state["weights"].values())
            self.assertAlmostEqual(sum_w, 1.0, delta=1e-12)

            # 2. Exact DCRI formula
            r_fusion = state["r_fusion"]
            dcri = state["dcri"]
            u_sum = state["u_sum"]
            expected_dcri = r_fusion - DELTA_FROZEN * u_sum
            self.assertAlmostEqual(dcri, expected_dcri, delta=1e-14)

            # 3. Monotonic non-inflation
            self.assertLessEqual(dcri, r_fusion + 1e-12)

            # 4. Negative DCRI
            if dcri < 0.0:
                from src.fusion.dcri_policy.policy_engine import assign_action, ACTION_ROUTINE_REVIEW
                tier = assign_action(dcri, TAU_1_DEFAULT, TAU_2_DEFAULT)
                self.assertEqual(tier, ACTION_ROUTINE_REVIEW)

    def test_canonical_per_packet_safeguard_invariants(self):
        """
        Exhaustively verifies per-packet safeguard decisions on all 500 canonical cases:
        1. When ambiguous and DCRI downgrades: tier reverts to base, blocked=True, flag emitted.
        2. When clear and DCRI downgrades: tier matches DCRI, blocked=False, no flag.
        3. Monotonic non-inflation under Policy S: T_safe <= T_base on every packet.
        """
        recomputed = self.evaluator.evaluate_safeguarded_policy(
            self.base_states,
            tau_1=TAU_1_DEFAULT,
            tau_2=TAU_2_DEFAULT,
            u_sum_threshold=AMBIGUITY_U_SUM_THRESHOLD,
            min_quality_threshold=AMBIGUITY_MIN_QUALITY_THRESHOLD,
        )

        for p in recomputed.per_packet_details:
            p_base = p["action_base"]
            p_dcri = p["action_dcri"]
            p_safe = p["action_safeguarded"]
            is_amb = p["is_ambiguous"]
            is_att = p["is_attempted_downgrade"]
            is_blk = p["downgrade_blocked"]
            flag = p["flag"]

            self.assertEqual(is_att, (p_dcri < p_base))
            if is_att and is_amb:
                self.assertTrue(is_blk)
                self.assertEqual(p_safe, p_base)
                self.assertIsNotNone(flag)
            elif is_att and not is_amb:
                self.assertFalse(is_blk)
                self.assertEqual(p_safe, p_dcri)
            else:
                self.assertFalse(is_blk)
                self.assertEqual(p_safe, p_dcri)

            self.assertLessEqual(p_safe, p_base)


if __name__ == "__main__":
    unittest.main()


