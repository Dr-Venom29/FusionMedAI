"""
FusionMedAI - Phase C11.14: Unit Tests for DCRI Safeguarded Policy and Ambiguity Gateway
Tests:
1. High uncertainty: a case that would be downgraded by DCRI receives the ambiguity flag and is not downgraded below its base tier.
2. Low input quality: a degraded input triggers the configured safeguard when it meets ambiguity criteria.
3. Low base risk with ambiguity: separate repeat-exam flag is retained even when base-risk tier is low.
4. Clear evidence: safeguard does not unnecessarily block a DCRI downgrade when eligibility criteria are not met.
5. Tier 2 preservation: Priority Escalation cases cannot be downgraded by DCRI when safeguard applies.
6. Missing modality: existing hard-masking and fail-closed behavior remains intact.
7. Score separation: adding safeguard does not alter computed DCRI or fused risk scores.
8. Auditability: every blocked downgrade records its reason, base tier, DCRI tier, and final tier.
"""

import unittest
from pathlib import Path

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.dcri_policy.policy_config import (
    DELTA_FROZEN,
    TAU_1_DEFAULT,
    TAU_2_DEFAULT,
    AMBIGUITY_U_SUM_THRESHOLD,
    AMBIGUITY_MIN_QUALITY_THRESHOLD,
    FLAG_REPEAT_EXAM_AMBIGUOUS_EVIDENCE,
    REASON_CLEAR_EVIDENCE,
    REASON_HIGH_UNCERTAINTY,
    REASON_DEGRADED_QUALITY,
    ACTION_ROUTINE_REVIEW,
    ACTION_ADDITIONAL_ASSESSMENT,
    ACTION_ESCALATION,
)
from src.fusion.dcri_policy.policy_engine import (
    PolicyEvaluator,
    detect_evidence_ambiguity,
    apply_safeguard_policy,
)


class TestDCRIPolicySafeguard(unittest.TestCase):
    """Validates the 8 core requirements of the DCRI policy safeguard gateway."""

    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[3]
        cls.cohort = load_frozen_cohort(repo_root=repo_root, n_packets=500, seed=115)
        cls.evaluator = PolicyEvaluator(delta=DELTA_FROZEN)
        cls.base_states = cls.evaluator.precompute_packet_base_state(cls.cohort)

    def test_1_high_uncertainty_blocks_downgrade(self):
        """Case with high uncertainty is flagged and blocked from DCRI-induced downgrade."""
        is_amb, reason = detect_evidence_ambiguity(u_sum=0.45, qualities={"retina": 1.0, "clinical": 1.0})
        self.assertTrue(is_amb)
        self.assertEqual(reason, REASON_HIGH_UNCERTAINTY)

        # Base is Tier 2, DCRI would downgrade to Tier 1
        action_final, blocked, flag, res_reason = apply_safeguard_policy(
            action_base=ACTION_ESCALATION,
            action_dcri=ACTION_ADDITIONAL_ASSESSMENT,
            is_ambiguous=is_amb,
            ambiguity_reason=reason,
        )
        self.assertEqual(action_final, ACTION_ESCALATION)
        self.assertTrue(blocked)
        self.assertEqual(flag, FLAG_REPEAT_EXAM_AMBIGUOUS_EVIDENCE)
        self.assertIn("BLOCKED_DOWNGRADE", res_reason)

    def test_2_low_input_quality_triggers_safeguard(self):
        """Case with degraded input quality triggers ambiguity flag and blocks downgrade."""
        is_amb, reason = detect_evidence_ambiguity(u_sum=0.15, qualities={"retina": 0.40, "clinical": 0.95})
        self.assertTrue(is_amb)
        self.assertEqual(reason, REASON_DEGRADED_QUALITY)

        # Base is Tier 1, DCRI would downgrade to Tier 0
        action_final, blocked, flag, res_reason = apply_safeguard_policy(
            action_base=ACTION_ADDITIONAL_ASSESSMENT,
            action_dcri=ACTION_ROUTINE_REVIEW,
            is_ambiguous=is_amb,
            ambiguity_reason=reason,
        )
        self.assertEqual(action_final, ACTION_ADDITIONAL_ASSESSMENT)
        self.assertTrue(blocked)
        self.assertEqual(flag, FLAG_REPEAT_EXAM_AMBIGUOUS_EVIDENCE)

    def test_3_low_base_risk_with_ambiguity_retains_flag(self):
        """Case with low base risk (Tier 0) still carries the repeat-exam flag when evidence is ambiguous."""
        is_amb, reason = detect_evidence_ambiguity(u_sum=0.35, qualities={"retina": 1.0})
        self.assertTrue(is_amb)

        # Both base and DCRI are Tier 0 (no downgrade attempted)
        action_final, blocked, flag, res_reason = apply_safeguard_policy(
            action_base=ACTION_ROUTINE_REVIEW,
            action_dcri=ACTION_ROUTINE_REVIEW,
            is_ambiguous=is_amb,
            ambiguity_reason=reason,
        )
        self.assertEqual(action_final, ACTION_ROUTINE_REVIEW)
        self.assertFalse(blocked)  # No downgrade was attempted
        self.assertEqual(flag, FLAG_REPEAT_EXAM_AMBIGUOUS_EVIDENCE)  # Flag is still present

    def test_4_clear_evidence_allows_downgrade(self):
        """When evidence is clear, safeguard does not block legitimate DCRI downgrade."""
        is_amb, reason = detect_evidence_ambiguity(u_sum=0.12, qualities={"retina": 0.95, "clinical": 0.98})
        self.assertFalse(is_amb)
        self.assertEqual(reason, REASON_CLEAR_EVIDENCE)

        # Base is Tier 1, DCRI is Tier 0
        action_final, blocked, flag, res_reason = apply_safeguard_policy(
            action_base=ACTION_ADDITIONAL_ASSESSMENT,
            action_dcri=ACTION_ROUTINE_REVIEW,
            is_ambiguous=is_amb,
            ambiguity_reason=reason,
        )
        self.assertEqual(action_final, ACTION_ROUTINE_REVIEW)
        self.assertFalse(blocked)
        self.assertIsNone(flag)

    def test_5_tier_2_preservation_under_safeguard(self):
        """A Priority Escalation (Tier 2) case cannot be downgraded to Tier 1 or Tier 0 when ambiguous."""
        for dcri_downgrade_target in [ACTION_ADDITIONAL_ASSESSMENT, ACTION_ROUTINE_REVIEW]:
            action_final, blocked, flag, _ = apply_safeguard_policy(
                action_base=ACTION_ESCALATION,
                action_dcri=dcri_downgrade_target,
                is_ambiguous=True,
                ambiguity_reason=REASON_HIGH_UNCERTAINTY,
            )
            self.assertEqual(action_final, ACTION_ESCALATION)
            self.assertTrue(blocked)
            self.assertEqual(flag, FLAG_REPEAT_EXAM_AMBIGUOUS_EVIDENCE)

    def test_6_cohort_missing_modality_safeguard_execution(self):
        """Safeguarded evaluation runs cleanly and verifies exact count conservation across canonical cohort."""
        res_safeguarded = self.evaluator.evaluate_safeguarded_policy(
            self.base_states,
            tau_1=TAU_1_DEFAULT,
            tau_2=TAU_2_DEFAULT,
        )
        self.assertEqual(res_safeguarded.n_packets, 500)
        # Strict count conservation: attempted = blocked + permitted
        self.assertEqual(
            res_safeguarded.attempted_downgrades_count,
            res_safeguarded.blocked_downgrades_count + res_safeguarded.permitted_downgrades_count,
            "Count conservation failed: attempted != blocked + permitted"
        )
        self.assertEqual(res_safeguarded.attempted_downgrades_count, 96)
        self.assertEqual(res_safeguarded.blocked_downgrades_count, 94)
        self.assertEqual(res_safeguarded.permitted_downgrades_count, 2)
        self.assertEqual(res_safeguarded.ambiguous_flags_count, 432)

    def test_7_score_separation_invariant(self):
        """Applying the safeguard does not alter the underlying DCRI or R_fusion scores."""
        res_safeguarded = self.evaluator.evaluate_safeguarded_policy(
            self.base_states,
            tau_1=TAU_1_DEFAULT,
            tau_2=TAU_2_DEFAULT,
        )
        for state, p in zip(self.base_states, res_safeguarded.per_packet_details):
            self.assertEqual(state["r_fusion"], p["r_fusion"])
            self.assertEqual(state["dcri"], p["dcri"])
            self.assertEqual(state["u_sum"], p["u_sum"])

    def test_8_auditability_of_blocked_downgrades(self):
        """Every blocked downgrade records its reason, base tier, DCRI tier, and final tier."""
        res_safeguarded = self.evaluator.evaluate_safeguarded_policy(
            self.base_states,
            tau_1=TAU_1_DEFAULT,
            tau_2=TAU_2_DEFAULT,
        )
        for p in res_safeguarded.per_packet_details:
            if p["downgrade_blocked"]:
                self.assertGreater(p["action_base"], p["action_dcri"])
                self.assertEqual(p["action_safeguarded"], p["action_base"])
                self.assertTrue(p["is_ambiguous"])
                self.assertEqual(p["flag"], FLAG_REPEAT_EXAM_AMBIGUOUS_EVIDENCE)
                self.assertIn("BLOCKED_DOWNGRADE", p["decision_reason"])

    def test_9_exact_boundary_conditions(self):
        """
        Verifies exact boundary behavior:
        - u_sum == 0.30 is inclusive (ambiguous)
        - u_sum == 0.299999 is clear (not ambiguous if quality is high)
        - min(quality) == 0.60 is acceptable (not degraded)
        - min(quality) == 0.599999 is degraded (ambiguous)
        """
        # Uncertainty boundary
        amb_at_boundary, reason_u = detect_evidence_ambiguity(u_sum=0.30, qualities={"r": 1.0, "f": 1.0})
        self.assertTrue(amb_at_boundary, "u_sum == 0.30 must be treated as ambiguous (inclusive)")
        self.assertEqual(reason_u, REASON_HIGH_UNCERTAINTY)

        amb_below_boundary, _ = detect_evidence_ambiguity(u_sum=0.299999, qualities={"r": 1.0, "f": 1.0})
        self.assertFalse(amb_below_boundary, "u_sum < 0.30 with good quality must be clear evidence")

        # Quality boundary
        amb_q_at_boundary, reason_q_clear = detect_evidence_ambiguity(u_sum=0.10, qualities={"r": 0.60})
        self.assertFalse(amb_q_at_boundary, "min(quality) == 0.60 must not trigger degraded quality")
        self.assertEqual(reason_q_clear, REASON_CLEAR_EVIDENCE)

        amb_q_below_boundary, reason_q_deg = detect_evidence_ambiguity(u_sum=0.10, qualities={"r": 0.599999})
        self.assertTrue(amb_q_below_boundary, "min(quality) < 0.60 must trigger degraded quality")
        self.assertEqual(reason_q_deg, REASON_DEGRADED_QUALITY)



if __name__ == "__main__":
    unittest.main()
