"""
FusionMedAI - Unit & Behavioral Tests for Outcome-Grounded Evaluation
Tests oracle independence, complete paired input identity, metric error defenses,
deterministic hierarchical bootstrap inference, tier derivations, and strict fail-closed artifact verification.
"""

import pytest
import shutil
import tempfile
from pathlib import Path
import json
import numpy as np

from src.fusion.outcome_evaluation.oracle_generator import OracleCohortGenerator
from src.fusion.outcome_evaluation.outcome_packet import OutcomeGroundedPacket, OutcomePacketError
from src.fusion.outcome_evaluation.outcome_metrics import (
    compute_prediction_metrics,
    compute_hierarchical_bootstrap_ci,
    assign_action_tier,
    compute_decision_loss,
    ACTION_COST_MATRIX,
    VALID_ACTION_TIERS,
)
from src.fusion.outcome_evaluation.outcome_runner import OutcomeEvaluationRunner
from src.fusion.baselines.decision_packet import ModalityRecord
from verification.fusion.outcome_evaluation.verify_outcome_artifacts import OutcomeArtifactVerifier, OutcomeVerificationError


def test_oracle_generator_independence_and_determinism():
    """Verifies that the oracle generator produces strictly bounded targets deterministically before routing."""
    gen1 = OracleCohortGenerator(seed=42)
    packets1 = gen1.generate_cohort(n_packets=50)

    gen2 = OracleCohortGenerator(seed=42)
    packets2 = gen2.generate_cohort(n_packets=50)

    assert len(packets1) == 50
    for p1, p2 in zip(packets1, packets2):
        assert p1.packet_id == p2.packet_id
        assert p1.oracle_risk == p2.oracle_risk
        assert p1.oracle_tier == p2.oracle_tier
        assert 0.0 <= p1.oracle_risk <= 1.0

        # Modality channel checks
        for m in ("retina", "foot", "clinical"):
            rec1 = p1.records[m]
            rec2 = p2.records[m]
            assert rec1.risk == rec2.risk
            assert rec1.confidence == rec2.confidence
            assert rec1.uncertainty == rec2.uncertainty
            assert rec1.quality == rec2.quality
            assert rec1.availability == rec2.availability


def test_packet_schema_strictness_and_tier_consistency():
    """Verifies that OutcomeGroundedPacket rejects invalid bounds, mismatched tiers, zero modalities, or invalid scenarios."""
    gen = OracleCohortGenerator(seed=10)
    valid_pkt = gen.generate_cohort(n_packets=1)[0]

    # Valid packet succeeds
    assert valid_pkt.packet_type == "OUTCOME_GROUNDED_DECISION_PACKET"

    # Mismatched tier for risk < 0.20
    with pytest.raises(OutcomePacketError, match="requires TIER_0_ROUTINE"):
        OutcomeGroundedPacket(
            packet_id="TEST_01",
            seed=10,
            oracle_risk=0.15,
            oracle_tier="TIER_2_ESCALATION",
            retina=valid_pkt.retina,
            foot=valid_pkt.foot,
            clinical=valid_pkt.clinical,
            scenario="CLEAN",
        )

    # Invalid scenario
    with pytest.raises(OutcomePacketError, match="Invalid scenario"):
        OutcomeGroundedPacket(
            packet_id="TEST_02",
            seed=10,
            oracle_risk=0.55,
            oracle_tier="TIER_2_ESCALATION",
            retina=valid_pkt.retina,
            foot=valid_pkt.foot,
            clinical=valid_pkt.clinical,
            scenario="UNKNOWN_SCENARIO",
        )

    # Zero available modalities rejected
    inactive_r = ModalityRecord(sample_id="S1", modality="retina", risk=0.0, calibrated_probability=(0.0, 0.0), confidence=0.0, uncertainty=0.0, quality=0.0, availability=False, reliability=0.9, model_version="v1")
    inactive_f = ModalityRecord(sample_id="S2", modality="foot", risk=0.0, calibrated_probability=(0.0, 0.0), confidence=0.0, uncertainty=0.0, quality=0.0, availability=False, reliability=0.9, model_version="v1")
    inactive_c = ModalityRecord(sample_id="S3", modality="clinical", risk=0.0, calibrated_probability=(0.0, 0.0), confidence=0.0, uncertainty=0.0, quality=0.0, availability=False, reliability=0.8, model_version="v1")

    with pytest.raises(OutcomePacketError, match="zero available modalities"):
        OutcomeGroundedPacket(
            packet_id="TEST_ZERO",
            seed=10,
            oracle_risk=0.55,
            oracle_tier="TIER_2_ESCALATION",
            retina=inactive_r,
            foot=inactive_f,
            clinical=inactive_c,
            scenario="CLEAN",
        )


def test_router_call_interception_input_identity():
    """
    Intercepts router calls to verify that B5 and B6 receive 100% identical channel inputs
    and produce exact mathematical equality on singleton availability regimes.
    """
    gen = OracleCohortGenerator(seed=88)
    packets = gen.generate_cohort(n_packets=100)
    runner = OutcomeEvaluationRunner()

    intercepted_b6_inputs = []
    intercepted_b5_inputs = []

    # Wrapper to intercept calls
    orig_route_b6 = runner.router_b6.route
    orig_route_b5 = runner.router_b5.route

    def spy_route_b6(r_input):
        intercepted_b6_inputs.append(r_input)
        return orig_route_b6(r_input)

    def spy_route_b5(r_input):
        intercepted_b5_inputs.append(r_input)
        return orig_route_b5(r_input)

    runner.router_b6.route = spy_route_b6
    runner.router_b5.route = spy_route_b5

    res = runner.evaluate_cohort(packets)

    assert len(intercepted_b6_inputs) == 100
    assert len(intercepted_b5_inputs) == 100

    for in_b6, in_b5 in zip(intercepted_b6_inputs, intercepted_b5_inputs):
        assert in_b6.available_modalities == in_b5.available_modalities
        for m in in_b6.available_modalities:
            assert in_b6.channels[m].confidence == in_b5.channels[m].confidence
            assert in_b6.channels[m].reliability == in_b5.channels[m].reliability
            assert in_b6.channels[m].uncertainty == in_b5.channels[m].uncertainty
            assert in_b6.channels[m].quality == in_b5.channels[m].quality

    # Singleton equality check
    singletons = [p for p in res["sample_packets"] if p["num_active"] == 1]
    assert len(singletons) > 0
    for p in singletons:
        assert p["r_b6"] == p["r_b5"]
        assert p["diff_mae_b6_minus_b5"] == 0.0


def test_metric_defensive_checks_and_deterministic_bootstrap():
    """Verifies that metrics fail-closed on invalid inputs and compute deterministic hierarchical bootstrap."""
    # Empty inputs raise ValueError
    with pytest.raises(ValueError, match="empty list"):
        compute_prediction_metrics([], [])

    # Mismatched lengths raise ValueError
    with pytest.raises(ValueError, match="Predictions length"):
        compute_prediction_metrics([0.5, 0.6], [0.5])

    # NaN / Inf raise ValueError
    with pytest.raises(ValueError, match="NaN or Inf"):
        compute_prediction_metrics([float("nan"), 0.6], [0.5, 0.6])

    # Action tier validation
    with pytest.raises(ValueError, match="Invalid assigned_tier"):
        compute_decision_loss("INVALID_TIER", "TIER_0_ROUTINE")

    with pytest.raises(ValueError, match="Invalid true_oracle_tier"):
        compute_decision_loss("TIER_0_ROUTINE", "INVALID_TIER")

    # Deterministic Hierarchical bootstrap test using fixed seeded PRNG
    rng = np.random.RandomState(999)
    synthetic_cohorts = [
        [float(rng.normal(-0.002, 0.005)) for _ in range(100)] for _ in range(5)
    ]
    h_res = compute_hierarchical_bootstrap_ci(synthetic_cohorts, n_bootstrap=500, seed=42)
    assert "grand_mean_diff" in h_res
    assert "ci_lower" in h_res
    assert "ci_upper" in h_res
    assert "cohort_t_pvalue" in h_res
    assert h_res["ci_lower"] <= h_res["grand_mean_diff"] <= h_res["ci_upper"]


def test_action_tier_derivation_from_scores():
    """Verifies boundary partitioning and decision cost calculations."""
    assert assign_action_tier(0.00, 0.20, 0.40) == "TIER_0_ROUTINE"
    assert assign_action_tier(0.199999, 0.20, 0.40) == "TIER_0_ROUTINE"
    assert assign_action_tier(0.20, 0.20, 0.40) == "TIER_1_ASSESSMENT"
    assert assign_action_tier(0.399999, 0.20, 0.40) == "TIER_1_ASSESSMENT"
    assert assign_action_tier(0.40, 0.20, 0.40) == "TIER_2_ESCALATION"
    assert assign_action_tier(0.95, 0.20, 0.40) == "TIER_2_ESCALATION"

    # Cost matrix tests
    assert compute_decision_loss("TIER_0_ROUTINE", "TIER_0_ROUTINE") == 0.0
    assert compute_decision_loss("TIER_0_ROUTINE", "TIER_2_ESCALATION") == 5.0
    assert compute_decision_loss("TIER_1_ASSESSMENT", "TIER_2_ESCALATION") == 2.5
    assert compute_decision_loss("TIER_2_ESCALATION", "TIER_0_ROUTINE") == 2.0


def test_quality_fidelity_and_practical_threshold_logic():
    """Verifies that quality term behavior under accurate vs false-alarm conditions is sound."""
    gen = OracleCohortGenerator(seed=301)
    packets = gen.generate_cohort(n_packets=200)
    runner = OutcomeEvaluationRunner()
    res = runner.evaluate_cohort(packets)

    fidelity_breakdown = res["fidelity_breakdown"]
    assert "ACCURATE" in fidelity_breakdown
    assert "MISLEADING_FALSE_ALARM" in fidelity_breakdown
    assert "MISLEADING_UNNOTICED" in fidelity_breakdown

    # Accurate sensing should show negative delta (B6 improves over B5)
    acc_delta = fidelity_breakdown["ACCURATE"]["delta_mae"]
    assert acc_delta < 0.0


def test_tampered_p_value_fails_oe07():
    """Verifies that tampering with the reported p-value (even below 1e-6) fails gate OE-07."""
    src_dir = Path("experiments/fusion/outcome_evaluation")
    
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for fname in ["protocol.json", "dev_results.json", "confirmatory_results.json", "summary.json", "freeze_manifest.json"]:
            shutil.copy(src_dir / fname, tmp_dir / fname)

        summary_path = tmp_dir / "summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        
        # Tamper p-value to an incorrect value (1e-8 when true is 7.95e-9)
        summary_data["primary_findings"]["H1_Primary_Outcome"]["cohort_t_pvalue"] = 1.0e-8
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        verifier = OutcomeArtifactVerifier(tmp_dir)
        with open(tmp_dir / "confirmatory_results.json", "r", encoding="utf-8") as f:
            conf = json.load(f)
        
        # Calling gate directly must fail
        verifier._gate_oe07_statistical_inference_and_decision_rule(conf, summary_data)
        assert verifier.gate_results["OE-07"]["passed"] is False


def test_tampered_ci_fails_oe07():
    """Verifies that tampering with reported CI bounds fails gate OE-07."""
    src_dir = Path("experiments/fusion/outcome_evaluation")
    
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for fname in ["protocol.json", "dev_results.json", "confirmatory_results.json", "summary.json", "freeze_manifest.json"]:
            shutil.copy(src_dir / fname, tmp_dir / fname)

        summary_path = tmp_dir / "summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        
        # Tamper CI bound
        summary_data["primary_findings"]["H1_Primary_Outcome"]["hierarchical_ci_95"] = [-0.003000, -0.001000]
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        verifier = OutcomeArtifactVerifier(tmp_dir)
        with open(tmp_dir / "confirmatory_results.json", "r", encoding="utf-8") as f:
            conf = json.load(f)
        
        verifier._gate_oe07_statistical_inference_and_decision_rule(conf, summary_data)
        assert verifier.gate_results["OE-07"]["passed"] is False


def test_incorrect_fidelity_subgroup_fails_oe05():
    """Verifies that tampering with fidelity subgroup results fails gate OE-05."""
    src_dir = Path("experiments/fusion/outcome_evaluation")
    
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for fname in ["protocol.json", "dev_results.json", "confirmatory_results.json", "summary.json", "freeze_manifest.json"]:
            shutil.copy(src_dir / fname, tmp_dir / fname)

        summary_path = tmp_dir / "summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        
        # Tamper accurate degradation delta
        summary_data["primary_findings"]["H2_Quality_Term_Isolation"]["accurate_degradation_delta_mae"] = 0.05
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        verifier = OutcomeArtifactVerifier(tmp_dir)
        with open(tmp_dir / "confirmatory_results.json", "r", encoding="utf-8") as f:
            conf = json.load(f)
        
        verifier._gate_oe05_degradation_and_fidelity_scaling(conf, summary_data)
        assert verifier.gate_results["OE-05"]["passed"] is False


def test_verify_outcome_artifacts_full_suite():
    """Runs the 8-gate artifact verification suite on sealed artifacts."""
    exp_dir = Path("experiments/fusion/outcome_evaluation")
    verifier = OutcomeArtifactVerifier(exp_dir)
    verdict = verifier.verify_all()

    assert verdict["status"] == "PASSED"
    assert verdict["passed_gates"] == 8
    assert verdict["total_gates"] == 8
