"""
FusionMedAI - Semantic Contract & Evidence Orchestration Test Suite
Comprehensive testing of target identity, probability vector validation, cross-target rejection,
actual model schema compatibility, uncertainty validation, routing invariants, active-path isolation,
deep immutability, report-level consistency, and historical benchmark decoupling.
"""

import pytest
import numpy as np
import json
import math
import types
from typing import Mapping

from src.fusion.contracts.evidence_report import (
    TargetDefinition,
    ModalityPrediction,
    RoutingResult,
    EvidenceReport,
    ResearchFusionIndex,
    SemanticContractError,
    CrossTargetAggregationError,
    validate_probability_vector,
    RETINA_TARGET,
    FOOT_TARGET,
    CLINICAL_TARGET,
    MODALITY_TARGET_REGISTRY,
)
from src.fusion.contracts.modality_output import ModalityContractValidationError
from src.fusion.contracts.retina_adapter import RetinaAdapter
from src.fusion.contracts.foot_adapter import FootAdapter
from src.fusion.contracts.clinical_adapter import ClinicalAdapter
from src.fusion.contracts.evidence_orchestrator import EvidenceOrchestrator
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)
from src.clinical.inference.schema import (
    ClinicalOutput,
    UncertaintyOutput,
    PredictiveInterval,
    FeatureAttribution,
    ShiftDetection,
    ModelProvenance,
)


# =============================================================================
# 1. Target Identity & Modality Registry Tests
# =============================================================================

def test_target_identity_preservation():
    """Target Identity: Every modality output declares a valid clinical target definition."""
    assert RETINA_TARGET.name == "diabetic_retinopathy_severity"
    assert RETINA_TARGET.type == "ordinal_multiclass"
    assert len(RETINA_TARGET.label_space) == 5
    assert isinstance(RETINA_TARGET.label_space, tuple)

    assert FOOT_TARGET.name == "wagner_wound_grade"
    assert FOOT_TARGET.type == "ordinal_multiclass"
    assert len(FOOT_TARGET.label_space) == 4
    assert isinstance(FOOT_TARGET.label_space, tuple)

    assert CLINICAL_TARGET.name == "hospital_readmission_30d"
    assert CLINICAL_TARGET.type == "binary_outcome"
    assert len(CLINICAL_TARGET.label_space) == 2
    assert isinstance(CLINICAL_TARGET.label_space, tuple)


@pytest.mark.parametrize("modality,wrong_target", [
    ("retina", FOOT_TARGET),
    ("retina", CLINICAL_TARGET),
    ("foot", RETINA_TARGET),
    ("foot", CLINICAL_TARGET),
    ("clinical", RETINA_TARGET),
    ("clinical", FOOT_TARGET),
])
def test_modality_target_mismatch_rejection(modality, wrong_target):
    """Target Mismatch: System strictly rejects all invalid modality-to-target pairings."""
    probs = [1.0 / len(wrong_target.label_space)] * len(wrong_target.label_space)
    rel = 0.929956 if modality == "retina" else (0.922266 if modality == "foot" else 0.825382)
    with pytest.raises(SemanticContractError) as exc_info:
        ModalityPrediction(
            modality=modality,
            task_id=wrong_target.name,
            target=wrong_target,
            prediction={"class": 0, "probabilities": probs},
            uncertainty={"method": "dummy", "value": 0.1},
            quality=1.0,
            available=True,
            reliability=rel,
            model_version="test_v1",
        )
    assert "Target definition mismatch" in str(exc_info.value)


# =============================================================================
# 2. Probability Vector Invariant Tests
# =============================================================================

@pytest.mark.parametrize("invalid_vector,expected_len,error_match", [
    ([0.2, 0.2, 0.2, 0.2], 5, "Unexpected probability vector length"),
    ([0.5, 0.5, float("nan"), 0.0, 0.0], 5, "must be a finite float"),
    ([0.5, 0.5, float("inf"), 0.0, 0.0], 5, "must be a finite float"),
    ([-0.1, 0.5, 0.4, 0.1, 0.1], 5, "out of valid probability bounds"),
    ([1.2, -0.1, 0.0, 0.0, -0.1], 5, "out of valid probability bounds"),
    ([0.1, 0.1, 0.1, 0.1, 0.1], 5, "Probabilities must sum to 1.0"),
    ([True, False, 0.5, 0.25, 0.25], 5, "must be a finite float"),
    (None, 5, "must be a non-null list or tuple"),
])
def test_invalid_probability_vector_rejection(invalid_vector, expected_len, error_match):
    """Shared Validator: Reject negative, NaN, infinite, incorrect length, boolean, and unnormalized probabilities."""
    with pytest.raises(SemanticContractError) as exc_info:
        validate_probability_vector(invalid_vector, expected_len)
    assert error_match in str(exc_info.value)


# =============================================================================
# 3. Model Adapter Raw Probability Validation (Semantic & Legacy)
# =============================================================================

@pytest.mark.parametrize("bad_probs", [
    [float("nan"), 0.2, 0.3, 0.4, 0.1],
    [float("inf"), 0.2, 0.3, 0.4, 0.1],
    [-0.1, 0.3, 0.4, 0.2, 0.2],
    [1.5, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],  # Malformed uniform sum=5
    [0.0, 0.0, 0.0, 0.0, 0.0],  # Zero sum
    [True, False, 0.5, 0.25, 0.25],
])
def test_retina_adapter_rejects_malformed_probabilities(bad_probs):
    """RetinaAdapter: Rejects non-finite, negative, unnormalized, boolean probabilities in semantic & legacy paths."""
    adapter = RetinaAdapter()
    raw = {"calib_probabilities": bad_probs, "mc_predictive_variance": 0.01}
    with pytest.raises(ModalityContractValidationError):
        adapter.adapt_to_semantic_prediction(raw)
    with pytest.raises(ModalityContractValidationError):
        adapter.adapt_prediction_dict(raw)


@pytest.mark.parametrize("bad_probs", [
    [float("nan"), 0.2, 0.4, 0.4],
    [float("inf"), 0.2, 0.4, 0.4],
    [-0.2, 0.4, 0.4, 0.4],
    [1.5, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0],  # Malformed uniform sum=4
    [0.0, 0.0, 0.0, 0.0],  # Zero sum
    [True, False, 0.5, 0.5],
])
def test_foot_adapter_rejects_malformed_probabilities(bad_probs):
    """FootAdapter: Rejects non-finite, negative, unnormalized, boolean probabilities in semantic & legacy paths."""
    adapter = FootAdapter()
    raw = {"calib_probabilities": bad_probs, "mc_predictive_entropy_norm": 0.3}
    with pytest.raises(ModalityContractValidationError):
        adapter.adapt_to_semantic_prediction(raw)
    with pytest.raises(ModalityContractValidationError):
        adapter.adapt_prediction_dict(raw)


@pytest.mark.parametrize("bad_prob", [
    float("nan"),
    float("inf"),
    -0.1,
    1.2,
    True,
    None,
])
def test_clinical_adapter_rejects_malformed_probability(bad_prob):
    """ClinicalAdapter: Rejects non-finite, negative, >1.0, boolean probabilities in semantic & legacy paths."""
    adapter = ClinicalAdapter()
    raw = {"calibrated_probability": bad_prob, "uncertainty": {"std_probability": 0.05}}
    with pytest.raises(ModalityContractValidationError):
        adapter.adapt_to_semantic_prediction(raw)
    with pytest.raises(ModalityContractValidationError):
        adapter.adapt_prediction_output(raw)


# =============================================================================
# 4. Modality Uncertainty Validation
# =============================================================================

@pytest.mark.parametrize("bad_var", [
    None,
    -0.05,
    float("nan"),
    float("inf"),
    True,
])
def test_retina_adapter_rejects_invalid_uncertainty(bad_var):
    """RetinaAdapter: Rejects missing, negative, non-finite, boolean variance without silent zero fallback."""
    adapter = RetinaAdapter()
    raw = {"calib_probabilities": [0.1, 0.2, 0.4, 0.2, 0.1]}
    if bad_var is not None:
        raw["mc_predictive_variance"] = bad_var
    with pytest.raises(ModalityContractValidationError):
        adapter.adapt_to_semantic_prediction(raw)
    with pytest.raises(ModalityContractValidationError):
        adapter.adapt_prediction_dict(raw)


@pytest.mark.parametrize("bad_entropy", [
    None,
    -0.1,
    1.5,
    float("nan"),
    float("inf"),
    True,
])
def test_foot_adapter_rejects_invalid_uncertainty(bad_entropy):
    """FootAdapter: Rejects missing, negative, out-of-bounds, non-finite normalized entropy."""
    adapter = FootAdapter()
    raw = {"calib_probabilities": [0.1, 0.2, 0.4, 0.3]}
    if bad_entropy is not None:
        raw["mc_predictive_entropy_norm"] = bad_entropy
    with pytest.raises(ModalityContractValidationError):
        adapter.adapt_to_semantic_prediction(raw)
    with pytest.raises(ModalityContractValidationError):
        adapter.adapt_prediction_dict(raw)


@pytest.mark.parametrize("bad_std", [
    None,
    -0.01,
    0.51,  # Bounded Bernoulli maximum standard deviation is 0.50
    1.0,
    float("nan"),
    float("inf"),
    True,
])
def test_clinical_adapter_rejects_invalid_uncertainty(bad_std):
    """ClinicalAdapter: Rejects missing, negative, impossible (>0.5), non-finite bootstrap std_probability."""
    adapter = ClinicalAdapter()
    raw = {"calibrated_probability": 0.3}
    if bad_std is not None:
        raw["uncertainty"] = {"std_probability": bad_std}
    with pytest.raises(ModalityContractValidationError):
        adapter.adapt_to_semantic_prediction(raw)
    with pytest.raises(ModalityContractValidationError):
        adapter.adapt_prediction_output(raw)


def test_modality_prediction_rejects_invalid_uncertainty_payload():
    """ModalityPrediction: Rejects missing, negative, non-finite, and out-of-bounds uncertainty values."""
    with pytest.raises(SemanticContractError):
        ModalityPrediction(
            modality="retina",
            task_id=RETINA_TARGET.name,
            target=RETINA_TARGET,
            prediction={"class": 0, "probabilities": [1.0, 0.0, 0.0, 0.0, 0.0]},
            uncertainty={"method": "mc", "value": -0.1},
            quality=1.0,
            available=True,
            reliability=FROZEN_RETINA_RELIABILITY,
            model_version="v1",
        )
    # Unavailable prediction cannot appear certain (must have value = 1.0)
    with pytest.raises(SemanticContractError):
        ModalityPrediction(
            modality="retina",
            task_id=RETINA_TARGET.name,
            target=RETINA_TARGET,
            prediction={"class": None, "probabilities": []},
            uncertainty={"method": "mc", "value": 0.0},
            quality=0.0,
            available=False,
            reliability=FROZEN_RETINA_RELIABILITY,
            model_version="v1",
        )


# =============================================================================
# 5. Production Output Compatibility (Objects and Real Interfaces)
# =============================================================================

def test_clinical_adapter_real_clinical_output_dataclass():
    """ClinicalAdapter: Accepts actual ClinicalOutput dataclass produced by ClinicalInferenceService."""
    real_output = ClinicalOutput(
        modality="clinical_tabular",
        encounter_id=12345,
        prediction=1,
        probability=0.38,
        calibrated_probability=0.32,
        calibration_method="Isotonic_Regression",
        confidence="HIGH",
        uncertainty=UncertaintyOutput(
            method="bootstrap_ensemble",
            std_probability=0.045,
            percentile_in_cohort=65.0,
            is_high_uncertainty=False,
            predictive_interval_95=PredictiveInterval(lower=0.23, upper=0.41),
            aleatoric_entropy=0.62,
        ),
        decision_tier="Tier 1",
        operating_threshold=0.20,
        feature_attributions=[
            FeatureAttribution(feature="num_medications", shap_value=0.15, rank=1, feature_value=12)
        ],
        shift_detection=ShiftDetection(
            is_degraded=False,
            missingness_ratio=0.02,
            blind_spot_warning=False,
            shift_alerts=[],
        ),
        model_provenance=ModelProvenance(
            model_name="CatBoost_HPO_Bootstrap_Ensemble",
            ensemble_size=50,
            frozen_calibrator="Isotonic_Regression",
            version="clinical_c10_v1.0",
            manifest_sha256="c10_verified_e2e",
        ),
    )

    adapter = ClinicalAdapter()
    pred = adapter.adapt_to_semantic_prediction(real_output)
    assert pred.modality == "clinical"
    assert pred.target == CLINICAL_TARGET
    assert pred.available is True
    assert pred.prediction["class"] == 0  # 0.32 < 0.5 nominal class threshold
    assert abs(sum(pred.prediction["probabilities"]) - 1.0) < 1e-5
    assert pred.uncertainty["raw_std_probability"] == 0.045
    assert pred.uncertainty["value"] == pytest.approx(0.045 / 0.5, abs=1e-5)
    assert pred.quality == pytest.approx(0.98, abs=1e-5)  # 1 - 0.02
    assert pred.prediction["quality_provenance"] == "derived_from_shift_detection"

    legacy = adapter.adapt_prediction_output(real_output)
    assert legacy.availability is True
    assert legacy.risk == pytest.approx(0.32, abs=1e-5)
    assert legacy.uncertainty == pytest.approx(0.045 / 0.5, abs=1e-5)


def test_retina_and_foot_captured_inference_output():
    """Retina & Foot Adapters: Process full captured inference dictionaries matching real frozen module signatures."""
    retina_captured = {
        "prediction": 2,
        "prediction_label": "Moderate",
        "raw_confidence": 0.72,
        "calib_confidence": 0.75,
        "raw_probabilities": [0.02, 0.08, 0.72, 0.14, 0.04],
        "calib_probabilities": [0.01, 0.09, 0.75, 0.12, 0.03],
        "calib_entropy_norm": 0.38,
        "calib_margin": 0.63,
        "mc_predictive_variance": 0.014,
        "mc_predictive_entropy_norm": 0.40,
    }
    r_pred = RetinaAdapter().adapt_to_semantic_prediction(retina_captured, quality=0.85)
    assert r_pred.prediction["class"] == 2
    assert r_pred.uncertainty["method"] == "mc_dropout_predictive_variance"
    assert r_pred.uncertainty["normalization_scale"] == 0.16
    assert r_pred.uncertainty["value"] == pytest.approx(0.014 / 0.16, abs=1e-5)
    assert r_pred.prediction["quality_provenance"] == "measured"

    foot_captured = {
        "modality": "foot",
        "prediction": 1,
        "prediction_label": "Grade 1 (Superficial)",
        "calib_probabilities": [0.08, 0.78, 0.10, 0.04],
        "calib_entropy_norm": 0.35,
        "mc_predictive_entropy_norm": 0.39,
        "mc_predictive_variance": 0.012,
    }
    f_pred = FootAdapter().adapt_to_semantic_prediction(foot_captured)
    assert f_pred.prediction["class"] == 1
    assert f_pred.uncertainty["method"] == "mc_dropout_option_b_normalized_entropy"
    assert f_pred.uncertainty["value"] == 0.39
    assert f_pred.quality == 0.5  # Neutral fallback for unmeasured
    assert f_pred.prediction["quality_provenance"] == "unmeasured_neutral_fallback"


# =============================================================================
# 6. Deep Immutability & Defensive Copies
# =============================================================================

def test_contract_deep_immutability():
    """Deep Immutability: Mutating inputs or outputs after creation is strictly prevented or isolated."""
    input_probs = [0.1, 0.7, 0.1, 0.05, 0.05]
    input_pred = {"class": 1, "probabilities": input_probs}
    input_unc = {"method": "mc", "value": 0.2}

    pred = ModalityPrediction(
        modality="retina",
        task_id=RETINA_TARGET.name,
        target=RETINA_TARGET,
        prediction=input_pred,
        uncertainty=input_unc,
        quality=0.9,
        available=True,
        reliability=FROZEN_RETINA_RELIABILITY,
        model_version="v1",
    )

    # 1. TargetDefinition label_space is a tuple
    with pytest.raises((AttributeError, TypeError)):
        RETINA_TARGET.label_space[0] = 99

    # 2. ModalityPrediction internal dicts are MappingProxyType
    with pytest.raises(TypeError):
        pred.prediction["class"] = 99

    with pytest.raises(TypeError):
        pred.uncertainty["value"] = 0.99

    # 3. Mutating outside original input does not contaminate the ModalityPrediction
    input_pred["class"] = 999
    input_probs[0] = 0.999
    assert pred.prediction["class"] == 1
    assert pred.prediction["probabilities"][0] == 0.1

    # 4. RoutingResult weights and active_modalities are immutable
    routing = RoutingResult(
        weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
        active_modalities=["retina"],
        routing_policy_version="v2",
        purpose="computational_attention_allocation",
    )
    with pytest.raises(TypeError):
        routing.weights["retina"] = 0.5
    with pytest.raises((AttributeError, TypeError)):
        routing.active_modalities[0] = "foot"

    # 5. EvidenceReport predictions is an immutable tuple
    report = EvidenceReport(
        patient_id="PT_IMMUTABLE",
        predictions=[pred],
        routing=routing,
        status="SUCCESS",
        common_target_prediction=None,
    )
    with pytest.raises((AttributeError, TypeError)):
        report.predictions[0] = None


# =============================================================================
# 7. Evidence-Report Consistency Invariants
# =============================================================================

def test_evidence_report_rejects_empty_success():
    """Report Consistency: Rejects SUCCESS report when no modality predictions are available."""
    r_unavail = RetinaAdapter().adapt_to_semantic_prediction(None)
    routing = RoutingResult(
        weights={"retina": 0.0, "foot": 0.0, "clinical": 0.0},
        active_modalities=[],
        routing_policy_version="v2",
    )
    with pytest.raises(SemanticContractError) as exc_info:
        EvidenceReport(
            patient_id="PT_ERR",
            predictions=[r_unavail],
            routing=routing,
            status="SUCCESS",
        )
    assert "status is SUCCESS but no modality predictions are available" in str(exc_info.value)


def test_evidence_report_rejects_active_empty_mismatch():
    """Report Consistency: Rejects NO_MODALITY_AVAILABLE report when active weights or predictions exist."""
    r_avail = RetinaAdapter().adapt_to_semantic_prediction({"calib_probabilities": [0.1, 0.7, 0.1, 0.05, 0.05], "mc_predictive_variance": 0.01})
    routing = RoutingResult(
        weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
        active_modalities=["retina"],
        routing_policy_version="v2",
    )
    with pytest.raises(SemanticContractError) as exc_info:
        EvidenceReport(
            patient_id="PT_ERR",
            predictions=[r_avail],
            routing=routing,
            status="NO_MODALITY_AVAILABLE",
        )
    assert "status is NO_MODALITY_AVAILABLE but active predictions exist" in str(exc_info.value)


def test_evidence_report_rejects_routing_prediction_mismatch():
    """Report Consistency: Rejects mismatch where unavailable prediction is marked active in routing."""
    r_unavail = RetinaAdapter().adapt_to_semantic_prediction(None)
    routing = RoutingResult(
        weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
        active_modalities=["retina"],  # Retina is unavailable!
        routing_policy_version="v2",
    )
    with pytest.raises(SemanticContractError) as exc_info:
        EvidenceReport(
            patient_id="PT_ERR",
            predictions=[r_unavail],
            routing=routing,
            status="SUCCESS",
        )
    assert "Available predictions [] do not match routing active modalities ['retina']" in str(exc_info.value)


def test_classification_argmax_consistency_check():
    """Classification Consistency: ModalityPrediction requires class to match argmax of probability vector."""
    with pytest.raises(SemanticContractError) as exc_info:
        ModalityPrediction(
            modality="retina",
            task_id=RETINA_TARGET.name,
            target=RETINA_TARGET,
            prediction={"class": 3, "probabilities": [0.05, 0.80, 0.05, 0.05, 0.05]},  # Argmax is 1, not 3!
            uncertainty={"method": "mc", "value": 0.1},
            quality=1.0,
            available=True,
            reliability=FROZEN_RETINA_RELIABILITY,
            model_version="v1",
        )
    assert "Classification inconsistency: predicted class is 3, but argmax of probability vector is 1" in str(exc_info.value)


# =============================================================================
# 8. Routing Invariants & Purposes
# =============================================================================

def test_routing_rejects_duplicate_active_modalities():
    """Routing Invariant: Rejects duplicate entries in active_modalities."""
    with pytest.raises(SemanticContractError) as exc_info:
        RoutingResult(
            weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
            active_modalities=["retina", "retina"],
            routing_policy_version="v2",
        )
    assert "Duplicate active modalities encountered" in str(exc_info.value)


def test_routing_rejects_unvalidated_purposes():
    """Routing Invariant: Rejects unvalidated purposes (including review_prioritization) in favor of attention allocation."""
    with pytest.raises(SemanticContractError) as exc_info:
        RoutingResult(
            weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
            active_modalities=["retina"],
            routing_policy_version="v2",
            purpose="review_prioritization",
        )
    assert "Routing purpose 'review_prioritization' is unrecognized or unsupported" in str(exc_info.value)

    # Valid purposes
    r1 = RoutingResult(
        weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
        active_modalities=["retina"],
        routing_policy_version="v2",
        purpose="computational_attention_allocation",
    )
    assert r1.purpose == "computational_attention_allocation"

    r2 = RoutingResult(
        weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
        active_modalities=["retina"],
        routing_policy_version="v2",
        purpose="experimental_benchmark",
    )
    assert r2.purpose == "experimental_benchmark"


# =============================================================================
# 9. Historical Reproducibility & Active Path Isolation
# =============================================================================

def test_historical_benchmark_rejection_of_duplicates_and_invalid_delta():
    """Historical Method: Validates delta and rejects duplicate modality predictions."""
    r_pred = RetinaAdapter().adapt_to_semantic_prediction({"calib_probabilities": [0.1, 0.7, 0.1, 0.05, 0.05], "mc_predictive_variance": 0.01})
    orchestrator = EvidenceOrchestrator()

    with pytest.raises(SemanticContractError) as exc_info:
        orchestrator.compute_historical_research_index([r_pred, r_pred])
    assert "Duplicate modality predictions encountered" in str(exc_info.value)

    with pytest.raises(SemanticContractError) as exc_info:
        orchestrator.compute_historical_research_index([r_pred], delta=-0.05)
    assert "delta penalty parameter must be a non-negative finite float" in str(exc_info.value)


def test_historical_index_frozen_c11_reproducibility():
    """Historical Method: Accurately reproduces weighted R_fusion and DCRI formulas with frozen discount."""
    r_pred = RetinaAdapter().adapt_to_semantic_prediction({"calib_probabilities": [0.0, 0.0, 1.0, 0.0, 0.0], "mc_predictive_variance": 0.016})
    f_pred = FootAdapter().adapt_to_semantic_prediction({"calib_probabilities": [0.0, 1.0, 0.0, 0.0], "mc_predictive_entropy_norm": 0.20})
    c_pred = ClinicalAdapter().adapt_to_semantic_prediction({"calibrated_probability": 0.40, "uncertainty": {"std_probability": 0.05}})

    orchestrator = EvidenceOrchestrator()
    hist = orchestrator.compute_historical_research_index([r_pred, f_pred, c_pred], delta=0.10)

    # R_retina = 0.5 (stage 2), R_foot = 1/3 = 0.333333 (grade 1), R_clinical = 0.40
    # Uncertainty sum = (0.016 / 0.16) + 0.20 + (0.05 / 0.5) = 0.10 + 0.20 + 0.10 = 0.40
    # Penalty P_U = 0.10 * 0.40 = 0.040000
    assert hist.uncertainty_penalty == pytest.approx(0.04, abs=1e-4)
    assert hist.dcri_value == pytest.approx(hist.index_value - hist.uncertainty_penalty, abs=1e-5)
    assert hist.is_clinical_probability is False
    assert hist.clinical_action_authorized is False


def test_active_pipeline_dependency_boundary():
    """Active Path Isolation: The active orchestrator module does not import historical aggregation code."""
    import inspect
    import src.fusion.contracts.evidence_orchestrator as orch_mod
    import src.fusion.contracts.evidence_report as rep_mod

    orch_source = inspect.getsource(orch_mod)
    rep_source = inspect.getsource(rep_mod)

    assert "src.fusion.dcri.aggregation" not in orch_source
    assert "src.fusion.dcri.aggregation" not in rep_source
    assert "calculate_unified_risk_probability" in orch_source  # Guard is present


def test_cross_target_rejection_at_runtime():
    """Cross-Target Guard: Direct calls to calculate_unified_risk_probability unconditionally fail closed."""
    orchestrator = EvidenceOrchestrator()
    with pytest.raises(CrossTargetAggregationError) as exc_info:
        orchestrator.calculate_unified_risk_probability(0.5, 0.3, 0.2)
    assert "CRITICAL SCIENTIFIC ERROR" in str(exc_info.value)


# =============================================================================
# 10. End-to-End Orchestration & Serialization
# =============================================================================

def test_end_to_end_orchestration_and_json_serialization():
    """End-to-End: Orchestration pipeline produces clean report and serializes strictly without NaN."""
    r_pred = RetinaAdapter().adapt_to_semantic_prediction({"calib_probabilities": [0.05, 0.80, 0.05, 0.05, 0.05], "mc_predictive_variance": 0.01})
    f_pred = FootAdapter().adapt_to_semantic_prediction({"calib_probabilities": [0.10, 0.70, 0.10, 0.10], "mc_predictive_entropy_norm": 0.25})

    orchestrator = EvidenceOrchestrator()
    report = orchestrator.orchestrate_evidence(
        patient_id="PT_E2E_001",
        predictions=[r_pred, f_pred],
        routing_purpose="computational_attention_allocation",
    )

    assert report.status == "SUCCESS"
    assert report.routing.purpose == "computational_attention_allocation"
    assert report.routing.weights["retina"] > 0.0
    assert report.routing.weights["foot"] > 0.0
    assert report.routing.weights["clinical"] == 0.0

    data = report.to_dict()
    assert "common_target_prediction" in data and data["common_target_prediction"] is None
    assert "historical_benchmark_index" not in data

    # Verify strict JSON without NaN
    json_bytes = json.dumps(data, allow_nan=False)
    parsed = json.loads(json_bytes)
    assert parsed["patient_id"] == "PT_E2E_001"
    assert len(parsed["predictions"]) == 2


# =============================================================================
# 11. Nested Deep Immutability Tests
# =============================================================================

def test_nested_metadata_deep_immutability():
    """Deep Immutability: Mutating nested dictionaries or lists post-construction is strictly prevented."""
    nested_flags = ["calibrated", "mc_dropout_enabled"]
    raw_pred = {
        "class": 0,
        "probabilities": [1.0, 0.0, 0.0, 0.0, 0.0],
        "provenance": {
            "source": "retina_unit_test",
            "flags": nested_flags,
            "pipeline_meta": {"stage": "phase_c11", "nested_list": [1, 2, 3]},
        },
    }

    pred = ModalityPrediction(
        modality="retina",
        task_id=RETINA_TARGET.name,
        target=RETINA_TARGET,
        prediction=raw_pred,
        uncertainty={"method": "mc", "value": 0.1, "details": {"passes": 25}},
        quality=0.9,
        available=True,
        reliability=FROZEN_RETINA_RELIABILITY,
        model_version="v1",
    )

    # 1. External mutation of input nested structures does not affect internal state
    nested_flags.append("external_contamination")
    raw_pred["provenance"]["source"] = "mutated_externally"
    raw_pred["provenance"]["pipeline_meta"]["stage"] = "corrupted"

    assert pred.prediction["provenance"]["source"] == "retina_unit_test"
    assert "external_contamination" not in pred.prediction["provenance"]["flags"]
    assert pred.prediction["provenance"]["pipeline_meta"]["stage"] == "phase_c11"

    # 2. Direct modification of nested MappingProxyType raises TypeError
    with pytest.raises(TypeError):
        pred.prediction["provenance"]["source"] = "direct_modification"

    with pytest.raises(TypeError):
        pred.prediction["provenance"]["pipeline_meta"]["stage"] = "direct_modification"

    # 3. Direct modification of nested tuple raises AttributeError
    with pytest.raises((AttributeError, TypeError)):
        pred.prediction["provenance"]["flags"].append("direct_append")

    # 4. Modifying un-frozen dictionary copy from to_dict() does not affect the contract
    report = EvidenceReport(
        patient_id="PT_NESTED",
        predictions=[pred],
        routing=RoutingResult(
            weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
            active_modalities=["retina"],
            routing_policy_version="v2",
            purpose="computational_attention_allocation",
        ),
        status="SUCCESS",
        common_target_prediction=None,
    )
    exported = report.to_dict()
    assert isinstance(exported["predictions"][0]["prediction"]["provenance"], dict)
    exported["predictions"][0]["prediction"]["provenance"]["source"] = "tampered_after_export"
    assert pred.prediction["provenance"]["source"] == "retina_unit_test"


# =============================================================================
# 12. Router Default Conversion & Purposed Alignment
# =============================================================================

def test_router_result_as_routing_result_default_arguments():
    """Router Conversion: Calling as_routing_result() without arguments succeeds with computational_attention_allocation."""
    from src.fusion.router.acarau_router import ACARAUv2Router
    from src.fusion.router.router_input import RouterInput, ModalityChannelInput

    r_in = RouterInput(
        retina=ModalityChannelInput("retina", 0.8, FROZEN_RETINA_RELIABILITY, 0.1, 0.9, True),
        foot=ModalityChannelInput("foot", 0.7, FROZEN_FOOT_RELIABILITY, 0.2, 0.9, True),
        clinical=ModalityChannelInput("clinical", 0.0, FROZEN_CLINICAL_RELIABILITY, 1.0, 0.0, False),
    )
    router = ACARAUv2Router()
    res = router.route(r_in)

    # Calling with zero arguments must NOT fail with purpose mismatch
    routing_contract = res.as_routing_result()
    assert isinstance(routing_contract, RoutingResult)
    assert routing_contract.purpose == "computational_attention_allocation"
    assert routing_contract.routing_policy_version == "acarau_v2.0"
    assert routing_contract.active_modalities == ("retina", "foot")
    assert abs(sum(routing_contract.weights.values()) - 1.0) < 1e-5


# =============================================================================
# 13. Adapter Semantics: Foot Entropy & Clinical Quality Fallback
# =============================================================================

def test_foot_adapter_distinguishes_mc_from_calibrated_entropy_labels():
    """FootAdapter: Labels stochastic MC entropy vs calibrated distribution entropy with unambiguous provenance."""
    adapter = FootAdapter()

    # Stochastic MC pass present
    stochastic_raw = {
        "calib_probabilities": [0.1, 0.6, 0.2, 0.1],
        "mc_predictive_entropy_norm": 0.35,
        "calib_entropy_norm": 0.45,
    }
    stochastic_pred = adapter.adapt_to_semantic_prediction(stochastic_raw)
    assert stochastic_pred.uncertainty["method"] == "mc_dropout_option_b_normalized_entropy"
    assert stochastic_pred.uncertainty["interpretation"] == "stochastic_mc_predictive_entropy"
    assert stochastic_pred.uncertainty["value"] == 0.35

    # Fallback to calibrated distribution entropy when mc_passes = 0
    deterministic_raw = {
        "calib_probabilities": [0.1, 0.6, 0.2, 0.1],
        "calib_entropy_norm": 0.45,
    }
    deterministic_pred = adapter.adapt_to_semantic_prediction(deterministic_raw)
    assert deterministic_pred.uncertainty["method"] == "calibrated_predictive_entropy"
    assert deterministic_pred.uncertainty["interpretation"] == "calibrated_distribution_entropy"
    assert deterministic_pred.uncertainty["value"] == 0.45


@pytest.mark.parametrize("invalid_missingness", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_clinical_adapter_rejects_invalid_missingness_ratio(invalid_missingness):
    """ClinicalAdapter: Rejects negative, >1.0, NaN, Inf, and boolean missingness_ratio."""
    adapter = ClinicalAdapter()
    raw = {
        "calibrated_probability": 0.30,
        "uncertainty": {"std_probability": 0.05},
        "shift_detection": {"missingness_ratio": invalid_missingness},
    }
    with pytest.raises(ModalityContractValidationError) as exc:
        adapter.adapt_to_semantic_prediction(raw)
    assert "missingness_ratio" in str(exc.value)

    with pytest.raises(ModalityContractValidationError) as exc:
        adapter.adapt_prediction_output(raw)
    assert "missingness_ratio" in str(exc.value)


def test_clinical_adapter_handles_missing_shift_detection_missingness():
    """ClinicalAdapter: Assigns explicit neutral fallback 0.5 when shift_detection is present but missingness_ratio is absent."""
    adapter = ClinicalAdapter()
    raw = {
        "calibrated_probability": 0.30,
        "uncertainty": {"std_probability": 0.05},
        "shift_detection": {"is_degraded": False},  # No missingness_ratio field
    }
    pred = adapter.adapt_to_semantic_prediction(raw)
    assert pred.quality == 0.5
    assert pred.prediction["quality_provenance"] == "unmeasured_neutral_fallback"

    legacy = adapter.adapt_prediction_output(raw)
    assert legacy.quality == 0.5


def test_retina_adapter_variance_upper_bound_enforcement():
    """RetinaAdapter: Rejects mc_predictive_variance exceeding theoretical upper bound 0.16 + 1e-6."""
    adapter = RetinaAdapter()

    # Valid variance at boundary 0.16
    valid_boundary = {
        "calib_probabilities": [0.2, 0.2, 0.2, 0.2, 0.2],
        "mc_predictive_variance": 0.16,
    }
    pred = adapter.adapt_to_semantic_prediction(valid_boundary)
    assert pred.uncertainty["value"] == pytest.approx(1.0, abs=1e-5)

    # Exceeding bound
    excessive = {
        "calib_probabilities": [0.2, 0.2, 0.2, 0.2, 0.2],
        "mc_predictive_variance": 0.165,
    }
    with pytest.raises(ModalityContractValidationError) as exc:
        adapter.adapt_to_semantic_prediction(excessive)
    assert "theoretical upper bound for 5-class simplex mean variance" in str(exc.value)

    with pytest.raises(ModalityContractValidationError) as exc:
        adapter.adapt_prediction_dict(excessive)
    assert "theoretical upper bound for 5-class simplex mean variance" in str(exc.value)


# =============================================================================
# 14. Orchestrator Input Type & Null Safety
# =============================================================================

def test_evidence_orchestrator_rejects_non_modality_prediction_instances():
    """EvidenceOrchestrator: Rejects non-ModalityPrediction items with deliberate SemanticContractError."""
    orchestrator = EvidenceOrchestrator()

    with pytest.raises(SemanticContractError) as exc:
        orchestrator.orchestrate_evidence(
            patient_id="PT_ERR",
            predictions=[{"modality": "retina", "available": True}],  # Dict instead of ModalityPrediction
        )
    assert "All predictions must be instances of ModalityPrediction" in str(exc.value)

    with pytest.raises(SemanticContractError) as exc:
        orchestrator.orchestrate_evidence(
            patient_id="PT_ERR",
            predictions="invalid_string_input",
        )
    assert "predictions must be a list or tuple of ModalityPrediction instances" in str(exc.value)


# =============================================================================
# 15. DCRI Result Invariant Strict Validator
# =============================================================================

def test_dcri_result_strict_validator_passes_valid_result():
    """DCRIResult: validate_strict_invariants() verifies valid results and fail-closed empty results."""
    from src.fusion.dcri.dcri_result import DCRIResult

    valid_res = DCRIResult(
        packet_id="PKT_VAL_001",
        r_fusion=0.50,
        u_sum=0.40,
        u_mean=0.20,
        uncertainty_penalty=0.08,
        dcri=0.42,
        delta=0.20,
        modality_weights={"retina": 0.6, "foot": 0.4, "clinical": 0.0},
        modality_risks={"retina": 0.5, "foot": 0.5, "clinical": 0.0},
        weighted_risk_contributions={"retina": 0.3, "foot": 0.2, "clinical": 0.0},
        modality_uncertainties={"retina": 0.1, "foot": 0.3, "clinical": 0.0},
        uncertainty_penalty_contributions={"retina": 0.02, "foot": 0.06, "clinical": 0.0},
        active_modalities=["retina", "foot"],
        num_active=2,
        status="SUCCESS",
    )
    valid_res.validate_strict_invariants()

    empty_res = DCRIResult.no_modality_available(delta=0.20)
    empty_res.validate_strict_invariants()


def test_dcri_result_strict_validator_catches_decompositions():
    """DCRIResult: validate_strict_invariants() catches factorization, sum, and status discrepancies."""
    from src.fusion.dcri.dcri_result import DCRIResult, DCRIContractError

    # Invalid status
    with pytest.raises(DCRIContractError) as exc:
        DCRIResult(
            packet_id="PKT_VAL_BAD_STATUS",
            r_fusion=0.5,
            u_sum=0.4,
            u_mean=0.2,
            uncertainty_penalty=0.08,
            dcri=0.42,
            delta=0.20,
            modality_weights={"retina": 0.6, "foot": 0.4, "clinical": 0.0},
            modality_risks={"retina": 0.5, "foot": 0.5, "clinical": 0.0},
            weighted_risk_contributions={"retina": 0.3, "foot": 0.2, "clinical": 0.0},
            modality_uncertainties={"retina": 0.1, "foot": 0.3, "clinical": 0.0},
            uncertainty_penalty_contributions={"retina": 0.02, "foot": 0.06, "clinical": 0.0},
            active_modalities=["retina", "foot"],
            num_active=2,
            status="UNAUTHORIZED_STATUS",
        )
    assert "status" in str(exc.value)

    # Mismatched weighted contribution K_i != w_i * r_i is rejected immediately upon construction
    with pytest.raises(DCRIContractError) as exc:
        DCRIResult(
            packet_id="PKT_VAL_BAD_K",
            r_fusion=0.50,
            u_sum=0.40,
            u_mean=0.20,
            uncertainty_penalty=0.08,
            dcri=0.42,
            delta=0.20,
            modality_weights={"retina": 0.6, "foot": 0.4, "clinical": 0.0},
            modality_risks={"retina": 0.5, "foot": 0.5, "clinical": 0.0},
            weighted_risk_contributions={"retina": 0.25, "foot": 0.25, "clinical": 0.0},  # Sum is 0.50, but K_retina should be 0.30!
            modality_uncertainties={"retina": 0.1, "foot": 0.3, "clinical": 0.0},
            uncertainty_penalty_contributions={"retina": 0.02, "foot": 0.06, "clinical": 0.0},
            active_modalities=["retina", "foot"],
            num_active=2,
            status="SUCCESS",
        )
    assert "Weighted contribution K_retina" in str(exc.value)


# =============================================================================
# 16. Pinned Frozen C11 Historical Regression Fixtures & Static AST Isolation
# =============================================================================

def test_pinned_frozen_c11_regression_fixture_values():
    """Historical Regression: Pins exact analytical reference numbers for router weights, index, penalty, and DCRI."""
    # Synthetic canonical packet matching exact derivation:
    # Retina: z_R = 1.0*1.0 + 1.0*0.929956 - 1.0*0.10 + 1.0*0.50 = 2.329956
    # Foot:   z_F = 1.0*1.0 + 1.0*0.922266 - 1.0*0.20 + 1.0*0.50 = 2.222266
    # Clin:   z_C = 1.0*0.20 + 1.0*0.825382 - 1.0*0.10 + 1.0*0.50 = 1.425382
    r_pred = RetinaAdapter().adapt_to_semantic_prediction(
        {"calib_probabilities": [0.0, 0.0, 1.0, 0.0, 0.0], "mc_predictive_variance": 0.016}
    )
    f_pred = FootAdapter().adapt_to_semantic_prediction(
        {"calib_probabilities": [0.0, 1.0, 0.0, 0.0], "mc_predictive_entropy_norm": 0.20}
    )
    c_pred = ClinicalAdapter().adapt_to_semantic_prediction(
        {"calibrated_probability": 0.40, "uncertainty": {"std_probability": 0.05}}
    )

    orchestrator = EvidenceOrchestrator()
    hist = orchestrator.compute_historical_research_index([r_pred, f_pred, c_pred], delta=0.10)

    # EXACT pinned reference numbers
    assert hist.weights["retina"] == pytest.approx(0.434288, abs=1e-5)
    assert hist.weights["foot"] == pytest.approx(0.389950, abs=1e-5)
    assert hist.weights["clinical"] == pytest.approx(0.175762, abs=1e-5)
    assert hist.index_value == pytest.approx(0.417432, abs=1e-5)
    assert hist.uncertainty_penalty == pytest.approx(0.040000, abs=1e-5)
    assert hist.dcri_value == pytest.approx(0.377432, abs=1e-5)


def test_ast_and_runtime_dependency_isolation():
    """Static AST & Runtime Isolation: Verifies orchestrator AST has no import targeting dcri.aggregation and active orchestration does not load it."""
    import ast
    import sys
    from pathlib import Path

    orch_path = Path("src/fusion/contracts/evidence_orchestrator.py")
    tree = ast.parse(orch_path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "src.fusion.dcri.aggregation" not in alias.name
                assert "aggregation" not in alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "src.fusion.dcri.aggregation" not in mod
            assert not (mod == "src.fusion.dcri" and any(alias.name == "aggregation" for alias in node.names))

    # Runtime sys.modules check during active evidence orchestration
    saved_agg = sys.modules.pop("src.fusion.dcri.aggregation", None)
    try:
        r_pred = RetinaAdapter().adapt_to_semantic_prediction(
            {"calib_probabilities": [0.1, 0.7, 0.1, 0.05, 0.05], "mc_predictive_variance": 0.02}
        )
        orchestrator = EvidenceOrchestrator()
        report = orchestrator.orchestrate_evidence(
            patient_id="PT_RUNTIME_ISOLATION",
            predictions=[r_pred],
            routing_purpose="computational_attention_allocation",
        )
        assert report.status == "SUCCESS"
        assert "src.fusion.dcri.aggregation" not in sys.modules
    finally:
        if saved_agg is not None:
            sys.modules["src.fusion.dcri.aggregation"] = saved_agg


# =============================================================================
# 17. Service-Interface Integration Tests
# =============================================================================

def test_clinical_adapter_service_interface_duck_typing():
    """Service Interface: ClinicalAdapter integrates with ClinicalInferenceService predict_encounter and predict_single."""
    from src.clinical.inference.schema import (
        ClinicalOutput,
        UncertaintyOutput,
        PredictiveInterval,
        ShiftDetection,
        ModelProvenance,
    )

    def _build_clinical_output(encounter_data):
        return ClinicalOutput(
            modality="clinical_tabular",
            encounter_id=encounter_data.get("encounter_id", 999),
            prediction=0,
            probability=0.25,
            calibrated_probability=0.22,
            calibration_method="isotonic_regression",
            confidence="HIGH",
            uncertainty=UncertaintyOutput(
                method="bootstrap_ensemble",
                std_probability=0.03,
                percentile_in_cohort=50.0,
                is_high_uncertainty=False,
                predictive_interval_95=PredictiveInterval(lower=0.16, upper=0.28),
                aleatoric_entropy=0.52,
            ),
            decision_tier="Tier 0",
            operating_threshold=0.20,
            feature_attributions=[],
            shift_detection=ShiftDetection(
                is_degraded=False,
                missingness_ratio=0.05,
                blind_spot_warning=False,
                shift_alerts=[],
            ),
            model_provenance=ModelProvenance(
                model_name="CatBoost_HPO_Bootstrap_Ensemble",
                ensemble_size=50,
                frozen_calibrator="Isotonic_Regression",
                version="clinical_c10_v1.0",
                manifest_sha256="c10_verified_e2e",
            ),
        )

    class StandardClinicalService:
        """Matches ClinicalInferenceService.predict_encounter(raw_input) signature."""
        def predict_encounter(self, raw_input):
            return _build_clinical_output(raw_input)

    class LegacySingleClinicalService:
        """Matches legacy predict_single(encounter_data) signature."""
        def predict_single(self, encounter_data):
            return _build_clinical_output(encounter_data)

    # 1. Test standard predict_encounter interface and semantic adaptation of ClinicalOutput dataclass
    std_service = StandardClinicalService()
    adapter = ClinicalAdapter(clinical_service=std_service)
    output = adapter.adapt_encounter({"encounter_id": 1001})

    assert output.availability is True
    assert output.risk == pytest.approx(0.22, abs=1e-5)
    assert output.uncertainty == pytest.approx(0.03 / 0.5, abs=1e-5)
    assert output.quality == pytest.approx(0.95, abs=1e-5)

    clinical_dataclass_out = std_service.predict_encounter({"encounter_id": 1001})
    sem_pred = adapter.adapt_to_semantic_prediction(clinical_dataclass_out)
    assert sem_pred.available is True
    assert sem_pred.prediction["probabilities"] == (pytest.approx(0.78), pytest.approx(0.22))
    assert sem_pred.prediction["quality_provenance"] == "derived_from_shift_detection"
    assert sem_pred.quality == pytest.approx(0.95, abs=1e-5)

    # 2. Test fallback predict_single interface
    legacy_adapter = ClinicalAdapter(clinical_service=LegacySingleClinicalService())
    legacy_out = legacy_adapter.adapt_encounter({"encounter_id": 1002})
    assert legacy_out.availability is True
    assert legacy_out.risk == pytest.approx(0.22, abs=1e-5)


# =============================================================================
# 18. Universal Deep Freeze & RoutingResult Canonical Modality Tests
# =============================================================================

def test_modality_prediction_rejects_mutable_and_unsupported_payload_objects():
    """Deep Freeze Guard: ModalityPrediction rejects numpy arrays, custom mutable objects, and non-string keys."""
    class CustomMutablePayload:
        def __init__(self):
            self.state = [1, 2, 3]

    # 1. Rejects custom mutable object inside prediction dict
    with pytest.raises(SemanticContractError) as exc:
        ModalityPrediction(
            modality="retina",
            task_id=RETINA_TARGET.name,
            target=RETINA_TARGET,
            prediction={
                "class": 0,
                "probabilities": [1.0, 0.0, 0.0, 0.0, 0.0],
                "mutable_obj": CustomMutablePayload(),
            },
            uncertainty={"method": "mc", "value": 0.1},
            quality=0.9,
            available=True,
            reliability=FROZEN_RETINA_RELIABILITY,
            model_version="v1",
        )
    assert "Unsupported or mutable payload object" in str(exc.value)

    # 2. Rejects mutable numpy ndarray inside uncertainty dict
    with pytest.raises(SemanticContractError) as exc:
        ModalityPrediction(
            modality="retina",
            task_id=RETINA_TARGET.name,
            target=RETINA_TARGET,
            prediction={"class": 0, "probabilities": [1.0, 0.0, 0.0, 0.0, 0.0]},
            uncertainty={"method": "mc", "value": 0.1, "raw_array": np.array([0.1, 0.2])},
            quality=0.9,
            available=True,
            reliability=FROZEN_RETINA_RELIABILITY,
            model_version="v1",
        )
    assert "ndarray" in str(exc.value)

    # 3. Rejects non-string dictionary keys
    with pytest.raises(SemanticContractError) as exc:
        ModalityPrediction(
            modality="retina",
            task_id=RETINA_TARGET.name,
            target=RETINA_TARGET,
            prediction={"class": 0, "probabilities": [1.0, 0.0, 0.0, 0.0, 0.0], "bad_map": {1: "one"}},
            uncertainty={"method": "mc", "value": 0.1},
            quality=0.9,
            available=True,
            reliability=FROZEN_RETINA_RELIABILITY,
            model_version="v1",
        )
    assert "must be strings" in str(exc.value)

    # 4. Rejects unordered set and frozenset objects for deterministic JSON serialization
    for unordered_obj in ({"tag_a", "tag_b"}, frozenset({"tag_a", "tag_b"})):
        with pytest.raises(SemanticContractError) as exc:
            ModalityPrediction(
                modality="retina",
                task_id=RETINA_TARGET.name,
                target=RETINA_TARGET,
                prediction={"class": 0, "probabilities": [1.0, 0.0, 0.0, 0.0, 0.0], "tags": unordered_obj},
                uncertainty={"method": "mc", "value": 0.1},
                quality=0.9,
                available=True,
                reliability=FROZEN_RETINA_RELIABILITY,
                model_version="v1",
            )
        assert "ordered sequences (list/tuple)" in str(exc.value)


def test_routing_result_rejects_unknown_and_non_string_active_modalities():
    """RoutingResult Guard: Rejects unknown active_modalities even when canonical weights sum to 1.0."""
    # Unknown modality in active_modalities while retina weight is 1.0
    with pytest.raises(SemanticContractError) as exc:
        RoutingResult(
            weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
            active_modalities=["retina", "unknown"],
            routing_policy_version="v2",
            purpose="computational_attention_allocation",
        )
    assert "Unknown active modalities" in str(exc.value)

    # Non-string element in active_modalities
    with pytest.raises(SemanticContractError) as exc:
        RoutingResult(
            weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
            active_modalities=["retina", 123],  # type: ignore
            routing_policy_version="v2",
            purpose="computational_attention_allocation",
        )
    assert "must be a string" in str(exc.value)


# =============================================================================
# 19. Adapter Boundary & Fallback Tests (Retina Tolerance & Foot Legacy Dict)
# =============================================================================

def test_retina_adapter_variance_float_tolerance_boundaries():
    """RetinaAdapter: Accepts variance at 0.16 + 0.5e-6 (clipping to 1.0) and rejects 0.16 + 2e-6."""
    adapter = RetinaAdapter()

    # Inside float tolerance: 0.16 + 0.5e-6 -> accepted and clipped to 1.0
    inside_tol = {
        "calib_probabilities": [0.2, 0.2, 0.2, 0.2, 0.2],
        "mc_predictive_variance": 0.16 + 0.5e-6,
    }
    sem_pred = adapter.adapt_to_semantic_prediction(inside_tol)
    assert sem_pred.uncertainty["value"] == 1.0
    leg_pred = adapter.adapt_prediction_dict(inside_tol)
    assert leg_pred.uncertainty == 1.0

    # Outside float tolerance: 0.16 + 2.0e-6 -> rejected
    outside_tol = {
        "calib_probabilities": [0.2, 0.2, 0.2, 0.2, 0.2],
        "mc_predictive_variance": 0.16 + 2.0e-6,
    }
    with pytest.raises(ModalityContractValidationError):
        adapter.adapt_to_semantic_prediction(outside_tol)
    with pytest.raises(ModalityContractValidationError):
        adapter.adapt_prediction_dict(outside_tol)


def test_foot_adapter_legacy_adapt_prediction_dict_calib_entropy_fallback():
    """FootAdapter: Legacy adapt_prediction_dict falls back cleanly when only calib_entropy_norm is provided."""
    adapter = FootAdapter()
    raw = {
        "calib_probabilities": [0.1, 0.6, 0.2, 0.1],
        "calib_entropy_norm": 0.42,
    }
    out = adapter.adapt_prediction_dict(raw)
    assert out.availability is True
    assert out.uncertainty == pytest.approx(0.42, abs=1e-6)


# =============================================================================
# 20. Historical Research Index Legacy Projected Scalar Validation
# =============================================================================

def test_historical_research_index_rejects_missing_legacy_projected_scalar():
    """Historical Index Guard: Rejects active ModalityPrediction with legacy_projected_scalar=None."""
    r_pred_no_legacy = ModalityPrediction(
        modality="retina",
        task_id=RETINA_TARGET.name,
        target=RETINA_TARGET,
        prediction={"class": 2, "probabilities": [0.0, 0.0, 1.0, 0.0, 0.0]},
        uncertainty={"method": "mc", "value": 0.1},
        quality=0.9,
        available=True,
        reliability=FROZEN_RETINA_RELIABILITY,
        model_version="v1",
        legacy_projected_scalar=None,
    )
    orchestrator = EvidenceOrchestrator()
    with pytest.raises(SemanticContractError) as exc:
        orchestrator.compute_historical_research_index([r_pred_no_legacy])
    assert "legacy_projected_scalar" in str(exc.value)


# =============================================================================
# 21. DCRIResult & RouterResult Deep Immutability & Cross-Field Validation
# =============================================================================

def test_dcri_result_empty_state_channel_zeroing_and_immutability():
    """DCRIResult Guard: Rejects non-zero channel maps in NO_MODALITY_AVAILABLE, inactive non-zero risk/uncertainty in strict check, and enforces immutability including |=."""
    from src.fusion.dcri.dcri_result import DCRIResult, DCRIContractError

    # 1. Empty state with non-zero modality_risks is rejected
    with pytest.raises(DCRIContractError) as exc:
        DCRIResult(
            packet_id="PKT_EMPTY_BAD_RISK",
            r_fusion=0.0,
            u_sum=0.0,
            u_mean=0.0,
            uncertainty_penalty=0.0,
            dcri=0.0,
            delta=0.20,
            modality_weights={"retina": 0.0, "foot": 0.0, "clinical": 0.0},
            modality_risks={"retina": 0.5, "foot": 0.0, "clinical": 0.0},
            weighted_risk_contributions={"retina": 0.0, "foot": 0.0, "clinical": 0.0},
            modality_uncertainties={"retina": 0.0, "foot": 0.0, "clinical": 0.0},
            uncertainty_penalty_contributions={"retina": 0.0, "foot": 0.0, "clinical": 0.0},
            active_modalities=[],
            num_active=0,
            status="NO_MODALITY_AVAILABLE",
        )
    assert "must be 0.0 in NO_MODALITY_AVAILABLE" in str(exc.value)

    # 2. DCRIResult constructor (__post_init__) immediately rejects non-zero risk or uncertainty on an inactive modality
    with pytest.raises(DCRIContractError) as exc:
        DCRIResult(
            packet_id="PKT_INACTIVE_RISK",
            r_fusion=0.50,
            u_sum=0.10,
            u_mean=0.10,
            uncertainty_penalty=0.02,
            dcri=0.48,
            delta=0.20,
            modality_weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
            modality_risks={"retina": 0.50, "foot": 0.40, "clinical": 0.0},  # foot is inactive but risk=0.40
            weighted_risk_contributions={"retina": 0.50, "foot": 0.0, "clinical": 0.0},
            modality_uncertainties={"retina": 0.10, "foot": 0.0, "clinical": 0.0},
            uncertainty_penalty_contributions={"retina": 0.02, "foot": 0.0, "clinical": 0.0},
            active_modalities=["retina"],
            num_active=1,
            status="SUCCESS",
        )
    assert "Inactive modality 'foot' has non-zero risk" in str(exc.value)

    with pytest.raises(DCRIContractError) as exc:
        DCRIResult(
            packet_id="PKT_INACTIVE_UNC",
            r_fusion=0.50,
            u_sum=0.10,
            u_mean=0.10,
            uncertainty_penalty=0.02,
            dcri=0.48,
            delta=0.20,
            modality_weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
            modality_risks={"retina": 0.50, "foot": 0.0, "clinical": 0.0},
            weighted_risk_contributions={"retina": 0.50, "foot": 0.0, "clinical": 0.0},
            modality_uncertainties={"retina": 0.10, "foot": 0.25, "clinical": 0.0},  # foot is inactive but uncertainty=0.25
            uncertainty_penalty_contributions={"retina": 0.02, "foot": 0.0, "clinical": 0.0},
            active_modalities=["retina"],
            num_active=1,
            status="SUCCESS",
        )
    assert "Inactive modality 'foot' has non-zero uncertainty" in str(exc.value)

    # 3. Container immutability (_FrozenDict including |= and _FrozenList)
    valid_empty = DCRIResult.no_modality_available(delta=0.20)
    orig_weights = dict(valid_empty.modality_weights)
    with pytest.raises(TypeError):
        valid_empty.modality_weights["retina"] = 1.0  # type: ignore
    with pytest.raises(TypeError):
        w_ref = valid_empty.modality_weights
        w_ref |= {"unexpected": 123.0}  # type: ignore
    assert dict(valid_empty.modality_weights) == orig_weights
    assert "unexpected" not in valid_empty.modality_weights
    assert json.loads(json.dumps(valid_empty.modality_weights)) == orig_weights

    with pytest.raises(TypeError):
        valid_empty.modality_risks["retina"] = 0.5  # type: ignore
    with pytest.raises(TypeError):
        valid_empty.active_modalities.append("retina")


def test_router_result_cross_field_validation_and_immutability():
    """RouterResult Guard: Enforces canonical active_modalities, availability agreement, empty-state zeroing, and immutability including |=."""
    from src.fusion.router.router_result import RouterResult, RouterContractError

    # 1. Unknown modality in active_modalities
    with pytest.raises(RouterContractError) as exc:
        RouterResult(
            weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
            logits={"retina": 1.0, "foot": 0.5, "clinical": 0.2},
            masked_logits={"retina": 1.0, "foot": float("-inf"), "clinical": float("-inf")},
            availability={"retina": True, "foot": False, "clinical": False},
            coefficients={"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "eta": 1.0},
            active_modalities=["unknown"],
            num_active=1,
            normalization_sum=1.0,
            routing_entropy=0.0,
            dominant_modality="retina",
            status="SUCCESS",
        )
    assert "Unknown or non-string modality" in str(exc.value)

    # 2. Disagreement between active_modalities and availability map
    with pytest.raises(RouterContractError) as exc:
        RouterResult(
            weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
            logits={"retina": 1.0, "foot": 0.5, "clinical": 0.2},
            masked_logits={"retina": 1.0, "foot": float("-inf"), "clinical": float("-inf")},
            availability={"retina": False, "foot": True, "clinical": False},
            coefficients={"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "eta": 1.0},
            active_modalities=["retina"],
            num_active=1,
            normalization_sum=1.0,
            routing_entropy=0.0,
            dominant_modality="retina",
            status="SUCCESS",
        )
    assert "disagrees with availability map" in str(exc.value)

    # 3. Non-zero normalization_sum or routing_entropy in NO_MODALITY_AVAILABLE
    with pytest.raises(RouterContractError) as exc:
        RouterResult(
            weights={"retina": 0.0, "foot": 0.0, "clinical": 0.0},
            logits={"retina": float("-inf"), "foot": float("-inf"), "clinical": float("-inf")},
            masked_logits={"retina": float("-inf"), "foot": float("-inf"), "clinical": float("-inf")},
            availability={"retina": False, "foot": False, "clinical": False},
            coefficients={"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "eta": 1.0},
            active_modalities=[],
            num_active=0,
            normalization_sum=1.0,  # Should be 0.0
            routing_entropy=0.0,
            dominant_modality=None,
            status="NO_MODALITY_AVAILABLE",
        )
    assert "normalization_sum" in str(exc.value)

    # 4. Container immutability including in-place union |=
    empty_router = RouterResult.no_modality_available(
        coefficients={"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "eta": 1.0},
    )
    orig_router_weights = dict(empty_router.weights)
    with pytest.raises(TypeError):
        empty_router.weights["retina"] = 1.0  # type: ignore
    with pytest.raises(TypeError):
        rw_ref = empty_router.weights
        rw_ref |= {"unexpected": 123.0}  # type: ignore
    assert dict(empty_router.weights) == orig_router_weights
    assert "unexpected" not in empty_router.weights
    assert json.loads(json.dumps(empty_router.weights)) == orig_router_weights

    with pytest.raises(TypeError):
        empty_router.availability["retina"] = True  # type: ignore
    with pytest.raises(TypeError):
        empty_router.active_modalities.append("retina")


# =============================================================================
# 22. Successful-State DCRI Channel Maps, ResearchFusionIndex, Router Diagnostics & Direct Construction
# =============================================================================

def test_dcri_result_successful_state_map_keys_and_values_validation():
    """DCRIResult Guard: Rejects missing keys, extra keys, and boolean/non-finite values in all 5 channel maps in SUCCESS state."""
    from src.fusion.dcri.dcri_result import DCRIResult, DCRIContractError

    base_kwargs = dict(
        packet_id="PKT_SUCCESS_MAP_CHECK",
        r_fusion=0.50,
        u_sum=0.10,
        u_mean=0.10,
        uncertainty_penalty=0.02,
        dcri=0.48,
        delta=0.20,
        modality_weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
        modality_risks={"retina": 0.50, "foot": 0.0, "clinical": 0.0},
        weighted_risk_contributions={"retina": 0.50, "foot": 0.0, "clinical": 0.0},
        modality_uncertainties={"retina": 0.10, "foot": 0.0, "clinical": 0.0},
        uncertainty_penalty_contributions={"retina": 0.02, "foot": 0.0, "clinical": 0.0},
        active_modalities=["retina"],
        num_active=1,
        status="SUCCESS",
    )

    for map_field in (
        "modality_weights",
        "modality_risks",
        "weighted_risk_contributions",
        "modality_uncertainties",
        "uncertainty_penalty_contributions",
    ):
        # 1. Missing key ('clinical' omitted)
        kw_missing = dict(base_kwargs)
        kw_missing[map_field] = {"retina": base_kwargs[map_field]["retina"], "foot": 0.0}
        with pytest.raises(DCRIContractError) as exc:
            DCRIResult(**kw_missing)
        assert f"{map_field} must be a mapping containing exactly keys" in str(exc.value)

        # 2. Extra unexpected key
        kw_extra = dict(base_kwargs)
        kw_extra[map_field] = {**base_kwargs[map_field], "extra": 0.0}
        with pytest.raises(DCRIContractError) as exc:
            DCRIResult(**kw_extra)
        assert f"{map_field} must be a mapping containing exactly keys" in str(exc.value)

        # 3. Boolean value rejected
        kw_bool = dict(base_kwargs)
        kw_bool[map_field] = {**base_kwargs[map_field], "foot": False}
        with pytest.raises(DCRIContractError) as exc:
            DCRIResult(**kw_bool)
        assert "must be a finite float (excluding bool)" in str(exc.value)


def test_research_fusion_index_direct_construction_boundary_guards():
    """ResearchFusionIndex Guard: Direct construction cannot set clinical authorization flags or invalid scalars/weights."""
    from src.fusion.contracts.evidence_report import ResearchFusionIndex

    valid_weights = {"retina": 0.5, "foot": 0.5, "clinical": 0.0}

    # 1. Cannot set is_clinical_probability=True
    with pytest.raises(SemanticContractError) as exc:
        ResearchFusionIndex(
            index_value=0.40,
            dcri_value=0.35,
            weights=valid_weights,
            uncertainty_penalty=0.05,
            is_clinical_probability=True,
        )
    assert "is_clinical_probability must be strictly False" in str(exc.value)

    # 2. Cannot set clinical_action_authorized=True
    with pytest.raises(SemanticContractError) as exc:
        ResearchFusionIndex(
            index_value=0.40,
            dcri_value=0.35,
            weights=valid_weights,
            uncertainty_penalty=0.05,
            clinical_action_authorized=True,
        )
    assert "clinical_action_authorized must be strictly False" in str(exc.value)

    # 3. Rejects boolean or non-finite scalar values
    with pytest.raises(SemanticContractError):
        ResearchFusionIndex(
            index_value=True,  # type: ignore
            dcri_value=0.35,
            weights=valid_weights,
            uncertainty_penalty=0.05,
        )

    # 4. Rejects inconsistent dcri_value != index_value - uncertainty_penalty
    with pytest.raises(SemanticContractError) as exc:
        ResearchFusionIndex(
            index_value=0.40,
            dcri_value=0.10,
            weights=valid_weights,
            uncertainty_penalty=0.05,
        )
    assert "must equal index_value - uncertainty_penalty" in str(exc.value)

    # 5. Rejects invalid weight keys or sum
    with pytest.raises(SemanticContractError):
        ResearchFusionIndex(
            index_value=0.40,
            dcri_value=0.35,
            weights={"retina": 0.5, "foot": 0.2, "clinical": 0.0},
            uncertainty_penalty=0.05,
        )


def test_router_result_diagnostic_fields_strict_validation():
    """RouterResult Guard: Validates strict booleans in availability, logit/masked_logit relationships, entropy bounds/formula, and dominant_modality."""
    from src.fusion.router.router_result import RouterResult, RouterContractError

    valid_kwargs = dict(
        weights={"retina": 0.6, "foot": 0.4, "clinical": 0.0},
        logits={"retina": 1.4, "foot": 1.0, "clinical": 0.5},
        masked_logits={"retina": 1.4, "foot": 1.0, "clinical": float("-inf")},
        availability={"retina": True, "foot": True, "clinical": False},
        coefficients={"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "eta": 1.0},
        active_modalities=["retina", "foot"],
        num_active=2,
        normalization_sum=1.0,
        routing_entropy=0.673012,
        dominant_modality="retina",
        status="SUCCESS",
    )
    # Baseline valid construction succeeds
    res = RouterResult(**valid_kwargs)
    assert res.dominant_modality == "retina"

    # 1. String "false" in availability map is rejected (not coerced via bool("false") -> True)
    bad_avail = {**valid_kwargs, "availability": {"retina": True, "foot": True, "clinical": "false"}}
    with pytest.raises(RouterContractError) as exc:
        RouterResult(**bad_avail)
    assert "must be a strict boolean" in str(exc.value)

    # 2. Missing key in coefficients
    bad_coeff = {**valid_kwargs, "coefficients": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}}
    with pytest.raises(RouterContractError) as exc:
        RouterResult(**bad_coeff)
    assert "coefficients must contain exactly keys" in str(exc.value)

    # 3. Active modality masked_logit != raw logit
    bad_active_mask = {**valid_kwargs, "masked_logits": {"retina": 0.9, "foot": 1.0, "clinical": float("-inf")}}
    with pytest.raises(RouterContractError) as exc:
        RouterResult(**bad_active_mask)
    assert "must have masked_logits == logits" in str(exc.value)

    # 4. Inactive modality masked_logit != -inf
    bad_inactive_mask = {**valid_kwargs, "masked_logits": {"retina": 1.4, "foot": 1.0, "clinical": -1e9}}
    with pytest.raises(RouterContractError) as exc:
        RouterResult(**bad_inactive_mask)
    assert "must have masked_logits == -inf" in str(exc.value)

    # 5a. Routing entropy exceeding ln(num_active)
    bad_entropy_bound = {**valid_kwargs, "routing_entropy": math.log(2) + 0.1}
    with pytest.raises(RouterContractError) as exc:
        RouterResult(**bad_entropy_bound)
    assert "exceeds maximum" in str(exc.value)

    # 5b. Routing entropy within [0, ln(2)] bounds (0.20) but inconsistent with weights (0.6, 0.4) -> H(w) = 0.673012
    bad_entropy_formula = {**valid_kwargs, "routing_entropy": 0.20}
    with pytest.raises(RouterContractError) as exc:
        RouterResult(**bad_entropy_formula)
    assert "does not match Shannon entropy of weights" in str(exc.value)

    # 6. Dominant modality does not match maximum weight active modality
    bad_dom = {**valid_kwargs, "dominant_modality": "foot"}
    with pytest.raises(RouterContractError) as exc:
        RouterResult(**bad_dom)
    assert "dominant_modality" in str(exc.value)

    # 7. Boolean num_active is rejected
    bad_num_active = {
        **valid_kwargs,
        "weights": {"retina": 1.0, "foot": 0.0, "clinical": 0.0},
        "masked_logits": {"retina": 1.4, "foot": float("-inf"), "clinical": float("-inf")},
        "availability": {"retina": True, "foot": False, "clinical": False},
        "active_modalities": ["retina"],
        "num_active": True,
        "routing_entropy": 0.0,
    }
    with pytest.raises(RouterContractError) as exc:
        RouterResult(**bad_num_active)
    assert "num_active must be an integer (excluding bool)" in str(exc.value)


def test_routing_result_and_evidence_report_direct_construction_guards():
    """Direct Construction Guard: RoutingResult and EvidenceReport raise SemanticContractError on malformed inputs."""
    r_pred = RetinaAdapter().adapt_to_semantic_prediction(
        {"calib_probabilities": [0.1, 0.7, 0.1, 0.05, 0.05], "mc_predictive_variance": 0.02}
    )
    valid_routing = RoutingResult(
        weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
        active_modalities=["retina"],
        routing_policy_version="acarau_v2.0",
        purpose="computational_attention_allocation",
        entropy=0.0,
    )

    # 1. Empty routing_policy_version in RoutingResult
    with pytest.raises(SemanticContractError) as exc:
        RoutingResult(
            weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
            active_modalities=["retina"],
            routing_policy_version="   ",
        )
    assert "routing_policy_version" in str(exc.value)

    # 2a. Invalid entropy in RoutingResult (non-zero entropy for 1 active modality)
    with pytest.raises(SemanticContractError) as exc:
        RoutingResult(
            weights={"retina": 1.0, "foot": 0.0, "clinical": 0.0},
            active_modalities=["retina"],
            routing_policy_version="acarau_v2.0",
            entropy=0.5,
        )
    assert "Routing entropy" in str(exc.value)

    # 2b. In-bounds entropy in RoutingResult that does not match Shannon entropy of weights (0.6, 0.4)
    with pytest.raises(SemanticContractError) as exc:
        RoutingResult(
            weights={"retina": 0.6, "foot": 0.4, "clinical": 0.0},
            active_modalities=["retina", "foot"],
            routing_policy_version="acarau_v2.0",
            entropy=0.25,
        )
    assert "does not match Shannon entropy of weights" in str(exc.value)

    # 3. Empty patient_id in EvidenceReport
    with pytest.raises(SemanticContractError) as exc:
        EvidenceReport(
            patient_id="",
            predictions=[r_pred],
            routing=valid_routing,
            status="SUCCESS",
        )
    assert "patient_id" in str(exc.value)

    # 4. Non-RoutingResult routing in EvidenceReport
    with pytest.raises(SemanticContractError) as exc:
        EvidenceReport(
            patient_id="PT_DIRECT_001",
            predictions=[r_pred],
            routing={"weights": {"retina": 1.0}},  # type: ignore
            status="SUCCESS",
        )
    assert "routing must be a RoutingResult instance" in str(exc.value)

    # 5. Non-ModalityPrediction item in EvidenceReport.predictions raises SemanticContractError (not AttributeError)
    with pytest.raises(SemanticContractError) as exc:
        EvidenceReport(
            patient_id="PT_DIRECT_002",
            predictions=[{"modality": "retina", "available": True}],
            routing=valid_routing,
            status="SUCCESS",
        )
    assert "All items in EvidenceReport.predictions must be ModalityPrediction instances" in str(exc.value)



