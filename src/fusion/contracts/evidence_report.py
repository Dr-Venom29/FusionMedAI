"""
FusionMedAI - Heterogeneous Clinical Evidence Orchestration Contracts
Defines explicit target definitions, modality predictions, routing results,
and evidence reports that preserve native clinical semantics without false aggregation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from collections.abc import Mapping
import math
import types
import numpy as np


class SemanticContractError(ValueError):
    """Raised when a semantic contract invariant is violated."""
    pass


class CrossTargetAggregationError(TypeError):
    """Raised when unsupported mathematical operations attempt to aggregate incompatible clinical targets."""
    pass


@dataclass(frozen=True)
class TargetDefinition:
    """
    Formal clinical target definition for a diagnostic model.
    """
    name: str                           # e.g., "diabetic_retinopathy_severity"
    type: str                           # "ordinal_multiclass" or "binary_outcome"
    horizon: str                        # "current_assessment" or "30_days"
    label_space: Union[List[Union[int, str]], Tuple[Union[int, str], ...]]  # e.g., (0, 1, 2, 3, 4)
    clinical_description: str

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise SemanticContractError("Target definition requires a non-empty name string.")
        if self.type not in ("ordinal_multiclass", "binary_outcome", "multiclass"):
            raise SemanticContractError(f"Target type '{self.type}' is unrecognized.")
        if not self.label_space or not isinstance(self.label_space, (list, tuple)):
            raise SemanticContractError("Target definition requires a non-empty label_space list or tuple.")
        object.__setattr__(self, "label_space", tuple(self.label_space))


# Pre-defined target definitions for the three native models
RETINA_TARGET = TargetDefinition(
    name="diabetic_retinopathy_severity",
    type="ordinal_multiclass",
    horizon="current_assessment",
    label_space=[0, 1, 2, 3, 4],
    clinical_description="Diabetic Retinopathy ordinal stage (0: None, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative)",
)

FOOT_TARGET = TargetDefinition(
    name="wagner_wound_grade",
    type="ordinal_multiclass",
    horizon="current_assessment",
    label_space=[0, 1, 2, 3],
    clinical_description="Wagner diabetic foot ulcer grade (0: Pre-ulcerative, 1: Superficial, 2: Deep to tendon/bone, 3: Abscess/osteomyelitis)",
)

CLINICAL_TARGET = TargetDefinition(
    name="hospital_readmission_30d",
    type="binary_outcome",
    horizon="30_days",
    label_space=[0, 1],
    clinical_description="All-cause inpatient hospital readmission within 30 days of discharge (0: No, 1: Yes)",
)


def validate_probability_vector(probs: Optional[Union[List[float], Tuple[float, ...]]], expected_length: int) -> None:
    """
    Validates that a probability vector has exact expected length, contains only
    finite numeric floats in [0, 1] (rejecting bools, NaN, inf), and sums to 1.0 within tolerance.
    """
    if probs is None or not isinstance(probs, (list, tuple)):
        raise SemanticContractError("Probability vector must be a non-null list or tuple.")
    if len(probs) != expected_length:
        raise SemanticContractError(
            f"Unexpected probability vector length: expected {expected_length}, got {len(probs)}."
        )
    for idx, p in enumerate(probs):
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p):
            raise SemanticContractError(
                f"Probability at index {idx} must be a finite float, got {p!r}."
            )
        if p < 0.0 or p > 1.0:
            raise SemanticContractError(
                f"Probability at index {idx} ({p}) out of valid probability bounds [0.0, 1.0]."
            )
    prob_sum = float(sum(probs))
    if not math.isfinite(prob_sum) or abs(prob_sum - 1.0) > 1e-4:
        raise SemanticContractError(
            f"Probabilities must sum to 1.0 ± 1e-4, got sum = {prob_sum:.8f}."
        )


def _deep_freeze(obj: Any) -> Any:
    """
    Recursively converts mutable mappings and sequences into immutable equivalents,
    and rejects unsupported, unordered, or mutable non-JSON-compatible objects:
    - dict / Mapping -> types.MappingProxyType with string keys and recursively frozen values
    - list / tuple -> tuple with recursively frozen elements
    - primitive immutable types (int, float, str, bool, None) -> validated & returned as-is
    - any other object (e.g., set, frozenset, np.ndarray, custom mutable objects) -> raises SemanticContractError
    """
    if isinstance(obj, (dict, Mapping)):
        frozen_dict = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise SemanticContractError(
                    f"Mapping keys in contract payload must be strings, got {type(k).__name__}."
                )
            frozen_dict[k] = _deep_freeze(v)
        return types.MappingProxyType(frozen_dict)
    elif isinstance(obj, (list, tuple)):
        return tuple(_deep_freeze(item) for item in obj)
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, int):
        return int(obj)
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            raise SemanticContractError(f"Non-finite float {obj!r} is not permitted in contract payload.")
        return float(obj)
    elif isinstance(obj, (str, type(None))):
        return obj
    elif isinstance(obj, np.generic):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            val = float(obj)
            if not math.isfinite(val):
                raise SemanticContractError(f"Non-finite float {val!r} is not permitted in contract payload.")
            return val
        raise SemanticContractError(
            f"Unsupported numpy scalar type '{type(obj).__name__}' in immutable contract."
        )
    else:
        raise SemanticContractError(
            f"Unsupported or mutable payload object of type '{type(obj).__name__}' in immutable contract; "
            f"only JSON-compatible scalar values, mappings, and ordered sequences (list/tuple) are permitted."
        )


def _deep_unfreeze(obj: Any) -> Any:
    """
    Recursively converts frozen mappings and tuples back into standard JSON-serializable dicts and lists.
    """
    if isinstance(obj, (dict, Mapping)):
        return {k: _deep_unfreeze(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_deep_unfreeze(item) for item in obj]
    else:
        return obj


# Authoritative modality-to-target registry
MODALITY_TARGET_REGISTRY: Dict[str, TargetDefinition] = {
    "retina": RETINA_TARGET,
    "foot": FOOT_TARGET,
    "clinical": CLINICAL_TARGET,
}


@dataclass(frozen=True)
class ModalityPrediction:
    """
    Validated Modality Prediction payload preserving native task semantics.
    Enforces genuine recursive deep immutability via MappingProxyType and defensive copies.
    """
    modality: str                       # "retina", "foot", "clinical"
    task_id: str                        # "diabetic_retinopathy_severity", etc.
    target: TargetDefinition
    prediction: Any                     # Native class probabilities, predicted class, calibration details
    uncertainty: Any                    # method, value, interpretation
    quality: float                      # Signal quality in [0.0, 1.0]
    available: bool                     # Availability indicator
    reliability: float                  # Validation-anchored reliability in [0.0, 1.0]
    model_version: str                  # Frozen model checkpoint identifier
    legacy_projected_scalar: Optional[float] = None  # Historical projection (explicitly non-clinical)

    def __post_init__(self):
        if self.modality not in MODALITY_TARGET_REGISTRY:
            raise SemanticContractError(
                f"Modality '{self.modality}' unrecognized. Must be one of {list(MODALITY_TARGET_REGISTRY.keys())}."
            )
        expected_target = MODALITY_TARGET_REGISTRY[self.modality]
        if self.target != expected_target or self.task_id != expected_target.name:
            raise SemanticContractError(
                f"Target definition mismatch: modality '{self.modality}' cannot be paired with "
                f"task '{self.task_id}' / target '{self.target.name}'. Expected target '{expected_target.name}'."
            )
        if not isinstance(self.available, bool):
            raise SemanticContractError("available must be a boolean.")
        if not isinstance(self.quality, (int, float)) or isinstance(self.quality, bool) or not math.isfinite(self.quality) or not (0.0 <= self.quality <= 1.0):
            raise SemanticContractError(f"quality must be a finite float in [0.0, 1.0], got {self.quality}")
        if not isinstance(self.reliability, (int, float)) or isinstance(self.reliability, bool) or not math.isfinite(self.reliability) or not (0.0 <= self.reliability <= 1.0):
            raise SemanticContractError(f"reliability must be a finite float in [0.0, 1.0], got {self.reliability}")
        if self.legacy_projected_scalar is not None:
            if (
                not isinstance(self.legacy_projected_scalar, (int, float))
                or isinstance(self.legacy_projected_scalar, bool)
                or not math.isfinite(self.legacy_projected_scalar)
                or not (0.0 <= self.legacy_projected_scalar <= 1.0)
            ):
                raise SemanticContractError(
                    f"legacy_projected_scalar must be None or a finite float in [0.0, 1.0], got {self.legacy_projected_scalar!r}"
                )

        # Uncertainty validation:
        if not isinstance(self.uncertainty, (dict, Mapping)):
            raise SemanticContractError("uncertainty payload must be a non-null dictionary or mapping.")
        if "method" not in self.uncertainty or not isinstance(self.uncertainty["method"], str) or not self.uncertainty["method"].strip():
            raise SemanticContractError("uncertainty payload must contain a non-empty 'method' string.")
        if "value" not in self.uncertainty:
            raise SemanticContractError("uncertainty payload must contain a numeric 'value' key.")

        u_val = self.uncertainty["value"]
        if not isinstance(u_val, (int, float)) or isinstance(u_val, bool) or not math.isfinite(u_val):
            raise SemanticContractError(f"uncertainty 'value' must be a finite float, got {u_val!r}.")
        if u_val < 0.0 or u_val > 1.0:
            raise SemanticContractError(f"uncertainty 'value' ({u_val}) out of valid bounds [0.0, 1.0].")

        # Prediction and availability validation:
        if not isinstance(self.prediction, (dict, Mapping)):
            raise SemanticContractError("prediction payload must be a non-null dictionary or mapping.")

        if not self.available:
            if abs(u_val - 1.0) > 1e-6:
                raise SemanticContractError(f"Unavailable prediction must have maximal uncertainty 1.0, got {u_val}.")
            if self.prediction.get("probabilities") not in (None, [], ()):
                raise SemanticContractError("Unavailable prediction must have empty probabilities.")
        else:
            probs = self.prediction.get("probabilities")
            validate_probability_vector(probs, len(self.target.label_space))

            # Classification consistency check:
            pred_class = self.prediction.get("class")
            if pred_class is None or not isinstance(pred_class, int) or isinstance(pred_class, bool):
                raise SemanticContractError(f"Available prediction must have an integer 'class', got {pred_class!r}.")
            expected_class = int(np.argmax(probs))
            if pred_class != expected_class:
                raise SemanticContractError(
                    f"Classification inconsistency: predicted class is {pred_class}, "
                    f"but argmax of probability vector is {expected_class}."
                )

        # Recursive defensive copy & deep immutability
        object.__setattr__(self, "prediction", _deep_freeze(self.prediction))
        object.__setattr__(self, "uncertainty", _deep_freeze(self.uncertainty))

    def get_calibrated_probability(self) -> List[float]:
        """Returns the native calibrated probability distribution."""
        if not self.available:
            return []
        return list(self.prediction.get("probabilities", []))

    def get_uncertainty_value(self) -> float:
        """Returns the scalar uncertainty estimate."""
        if not self.available:
            return 1.0
        val = self.uncertainty.get("value")
        if val is None or not math.isfinite(val):
            raise SemanticContractError("Invalid uncertainty value encountered.")
        return float(val)


@dataclass(frozen=True)
class RoutingResult:
    """
    Routing authority allocation result.
    Weights represent attention / review prioritization, NOT contributions to shared clinical risk.
    """
    weights: Any
    active_modalities: Any
    routing_policy_version: str
    purpose: str = "computational_attention_allocation"
    entropy: Optional[float] = None

    def __post_init__(self):
        if not isinstance(self.routing_policy_version, str) or not self.routing_policy_version.strip():
            raise SemanticContractError("routing_policy_version must be a non-empty string.")

        valid_purposes = (
            "computational_attention_allocation",
            "experimental_benchmark",
        )
        if self.purpose not in valid_purposes:
            raise SemanticContractError(
                f"Routing purpose '{self.purpose}' is unrecognized or unsupported. "
                f"Must be one of {valid_purposes}."
            )
        if not isinstance(self.active_modalities, (list, tuple)):
            raise SemanticContractError(
                f"active_modalities must be a list or tuple of modality strings, got {type(self.active_modalities).__name__}."
            )
        active = list(self.active_modalities)
        for m in active:
            if not isinstance(m, str):
                raise SemanticContractError(
                    f"Each active modality entry must be a string, got {type(m).__name__}."
                )
        allowed = {"retina", "foot", "clinical"}
        unknown = set(active) - allowed
        if unknown:
            raise SemanticContractError(
                f"Unknown active modalities: {sorted(unknown)}"
            )
        if len(active) != len(set(active)):
            raise SemanticContractError(f"Duplicate active modalities encountered: {active}")

        if not isinstance(self.weights, (dict, Mapping)):
            raise SemanticContractError("Routing weights must be a dictionary or mapping.")
        if set(self.weights.keys()) != allowed:
            raise SemanticContractError(f"Routing weights must contain exactly {allowed}, got {set(self.weights.keys())}")
        for mod, w in self.weights.items():
            if not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w):
                raise SemanticContractError(f"Routing weight for '{mod}' must be finite float, got {w}.")
            if not (0.0 - 1e-7 <= w <= 1.0 + 1e-7):
                raise SemanticContractError(f"Routing weight for '{mod}' ({w}) out of bounds [0.0, 1.0].")
            if mod not in active and abs(w) > 1e-7:
                raise SemanticContractError(f"Inactive modality '{mod}' cannot have non-zero weight {w}.")
        if active:
            w_sum = sum(self.weights.get(m, 0.0) for m in active)
            if not math.isfinite(w_sum) or abs(w_sum - 1.0) > 1e-4:
                raise SemanticContractError(f"Active routing weights must sum to 1.0, got {w_sum:.6f}")
        else:
            w_sum = sum(self.weights.values())
            if abs(w_sum) > 1e-7:
                raise SemanticContractError(f"All routing weights must be 0.0 when no modalities are active, got sum = {w_sum}")

        # Compute expected Shannon entropy H(w) = -sum_{i in active, w_i > 0} w_i * ln(w_i)
        expected_entropy = 0.0
        for m in active:
            w_val = float(self.weights[m])
            if w_val > 1e-12:
                expected_entropy -= w_val * math.log(w_val)

        if self.entropy is None:
            object.__setattr__(self, "entropy", round(expected_entropy, 6))
        else:
            if not isinstance(self.entropy, (int, float)) or isinstance(self.entropy, bool) or not math.isfinite(self.entropy):
                raise SemanticContractError(f"Routing entropy must be a finite float, got {self.entropy!r}.")
            if self.entropy < -1e-7:
                raise SemanticContractError(f"Routing entropy must be non-negative, got {self.entropy}.")
            max_entropy = math.log(len(active)) if len(active) > 1 else 0.0
            if self.entropy > max_entropy + 1e-5:
                raise SemanticContractError(
                    f"Routing entropy ({self.entropy:.6f}) exceeds theoretical maximum ({max_entropy:.6f}) for {len(active)} active modalities."
                )
            if abs(float(self.entropy) - expected_entropy) > 1e-4:
                raise SemanticContractError(
                    f"Routing entropy ({self.entropy:.6f}) does not match Shannon entropy of weights ({expected_entropy:.6f})."
                )
            object.__setattr__(self, "entropy", float(self.entropy))

        object.__setattr__(self, "weights", types.MappingProxyType({str(k): float(v) for k, v in self.weights.items()}))
        object.__setattr__(self, "active_modalities", tuple(active))


@dataclass(frozen=True)
class ResearchFusionIndex:
    """
    Historical controlled research fusion index.
    STRICTLY NON-CLINICAL: For benchmark reproduction only.
    """
    index_value: float
    dcri_value: float
    weights: Any
    uncertainty_penalty: float
    is_clinical_probability: bool = False
    clinical_action_authorized: bool = False
    disclaimer: str = (
        "EXPERIMENTAL BENCHMARK ONLY: This score represents a weighted average of heterogeneous "
        "ordinal and binary risk projections. It does not measure unified patient risk, a composite "
        "outcome probability, or validated clinical triage urgency."
    )

    def __post_init__(self):
        if not isinstance(self.is_clinical_probability, bool) or self.is_clinical_probability is not False:
            raise SemanticContractError(
                "UNAUTHORIZED SEMANTIC OVERRIDE: ResearchFusionIndex.is_clinical_probability must be strictly False."
            )
        if not isinstance(self.clinical_action_authorized, bool) or self.clinical_action_authorized is not False:
            raise SemanticContractError(
                "UNAUTHORIZED SEMANTIC OVERRIDE: ResearchFusionIndex.clinical_action_authorized must be strictly False."
            )
        if not isinstance(self.disclaimer, str) or not self.disclaimer.strip():
            raise SemanticContractError("ResearchFusionIndex.disclaimer must be a non-empty string.")

        for field_name, val in (
            ("index_value", self.index_value),
            ("dcri_value", self.dcri_value),
            ("uncertainty_penalty", self.uncertainty_penalty),
        ):
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
                raise SemanticContractError(f"ResearchFusionIndex.{field_name} must be a finite float, got {val!r}.")

        if self.index_value < -1e-7 or self.index_value > 1.0 + 1e-7:
            raise SemanticContractError(f"ResearchFusionIndex.index_value ({self.index_value}) must be in [0.0, 1.0].")
        if self.uncertainty_penalty < -1e-7:
            raise SemanticContractError(
                f"ResearchFusionIndex.uncertainty_penalty ({self.uncertainty_penalty}) must be non-negative."
            )
        if abs((self.index_value - self.uncertainty_penalty) - self.dcri_value) > 1e-4:
            raise SemanticContractError(
                f"ResearchFusionIndex.dcri_value ({self.dcri_value}) must equal index_value - uncertainty_penalty "
                f"({self.index_value - self.uncertainty_penalty:.6f})."
            )

        allowed = {"retina", "foot", "clinical"}
        if not isinstance(self.weights, (dict, Mapping)) or set(self.weights.keys()) != allowed:
            raise SemanticContractError(
                f"ResearchFusionIndex.weights must be a mapping with exact keys {sorted(allowed)}."
            )
        for mod, w in self.weights.items():
            if not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w):
                raise SemanticContractError(f"ResearchFusionIndex weight for '{mod}' must be a finite float, got {w!r}.")
            if w < -1e-7 or w > 1.0 + 1e-7:
                raise SemanticContractError(f"ResearchFusionIndex weight for '{mod}' ({w}) out of bounds [0.0, 1.0].")

        w_sum = sum(float(v) for v in self.weights.values())
        if abs(w_sum) <= 1e-7:
            if abs(self.index_value) > 1e-7 or abs(self.dcri_value) > 1e-7 or abs(self.uncertainty_penalty) > 1e-7:
                raise SemanticContractError(
                    "Zero-weight ResearchFusionIndex must have zero index_value, dcri_value, and uncertainty_penalty."
                )
        elif abs(w_sum - 1.0) > 1e-4:
            raise SemanticContractError(
                f"ResearchFusionIndex weights must sum to 1.0 (active) or 0.0 (empty), got {w_sum:.6f}."
            )

        object.__setattr__(self, "weights", types.MappingProxyType({str(k): float(v) for k, v in self.weights.items()}))


@dataclass(frozen=True)
class EvidenceReport:
    """
    Primary patient-evidence report for multimodal clinical evidence orchestration.
    Enforces report-level consistency across status, predictions, and routing.
    """
    patient_id: str
    predictions: Any
    routing: RoutingResult
    status: str
    common_target_prediction: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not isinstance(self.patient_id, str) or not self.patient_id.strip():
            raise SemanticContractError("patient_id must be a non-empty string.")
        if self.status not in ("SUCCESS", "NO_MODALITY_AVAILABLE"):
            raise SemanticContractError(f"Status '{self.status}' must be 'SUCCESS' or 'NO_MODALITY_AVAILABLE'.")
        # STRICT RESEARCH BOUNDARY: Reject ANY non-null common_target_prediction until a real paired model exists
        if self.common_target_prediction is not None:
            raise SemanticContractError(
                "UNAUTHORIZED COMMON TARGET PREDICTION: FusionMedAI operates on unpaired cohorts. "
                "No validated common-target model exists. common_target_prediction must be None."
            )
        if not isinstance(self.routing, RoutingResult):
            raise SemanticContractError(
                f"routing must be a RoutingResult instance, got {type(self.routing).__name__}."
            )
        if not isinstance(self.predictions, (list, tuple)):
            raise SemanticContractError(
                f"predictions must be a list or tuple of ModalityPrediction instances, got {type(self.predictions).__name__}."
            )

        preds = list(self.predictions)
        for p in preds:
            if not isinstance(p, ModalityPrediction):
                raise SemanticContractError(
                    f"All items in EvidenceReport.predictions must be ModalityPrediction instances, got {type(p).__name__}."
                )

        mod_names = [p.modality for p in preds]
        if len(mod_names) != len(set(mod_names)):
            raise SemanticContractError(f"Duplicate modality predictions encountered in EvidenceReport: {mod_names}")
        for m in mod_names:
            if m not in MODALITY_TARGET_REGISTRY:
                raise SemanticContractError(f"Unrecognized modality '{m}' in EvidenceReport.")

        avail_mods = [p.modality for p in preds if p.available]
        routing_active = list(self.routing.active_modalities)

        # Consistency check 1: available predictions must match routing active modalities
        if set(avail_mods) != set(routing_active):
            raise SemanticContractError(
                f"Consistency violation: Available predictions {avail_mods} do not match "
                f"routing active modalities {routing_active}."
            )

        # Consistency check 2: status agreement
        if self.status == "NO_MODALITY_AVAILABLE":
            if avail_mods:
                raise SemanticContractError(
                    f"Report status is NO_MODALITY_AVAILABLE but active predictions exist: {avail_mods}"
                )
            if routing_active:
                raise SemanticContractError(
                    f"Report status is NO_MODALITY_AVAILABLE but routing has active modalities: {routing_active}"
                )
            for m, w in self.routing.weights.items():
                if abs(w) > 1e-7:
                    raise SemanticContractError(
                        f"Report status is NO_MODALITY_AVAILABLE but routing weight for '{m}' is non-zero: {w}"
                    )
        elif self.status == "SUCCESS":
            if not avail_mods:
                raise SemanticContractError(
                    "Report status is SUCCESS but no modality predictions are available."
                )

        object.__setattr__(self, "predictions", tuple(preds))

    def get_prediction_for_modality(self, modality: str) -> Optional[ModalityPrediction]:
        """Finds prediction for a given modality."""
        for p in self.predictions:
            if p.modality == modality:
                return p
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the evidence report to a standard JSON-serializable dictionary copy."""
        return {
            "patient_id": self.patient_id,
            "status": self.status,
            "predictions": [
                {
                    "modality": p.modality,
                    "task_id": p.task_id,
                    "target": {
                        "name": p.target.name,
                        "type": p.target.type,
                        "horizon": p.target.horizon,
                        "label_space": list(p.target.label_space),
                        "clinical_description": p.target.clinical_description,
                    },
                    "prediction": _deep_unfreeze(p.prediction),
                    "uncertainty": _deep_unfreeze(p.uncertainty),
                    "quality": p.quality,
                    "available": p.available,
                    "reliability": p.reliability,
                    "model_version": p.model_version,
                }
                for p in self.predictions
            ],
            "routing": {
                "weights": _deep_unfreeze(self.routing.weights),
                "active_modalities": list(self.routing.active_modalities),
                "routing_policy_version": self.routing.routing_policy_version,
                "purpose": self.routing.purpose,
                "entropy": self.routing.entropy,
            },
            "common_target_prediction": None,
        }

