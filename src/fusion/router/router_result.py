"""
FusionMedAI - Phase C11.4: Router Diagnostics & Output Contract
Defines research-grade immutable diagnostics container RouterResult for ACARA-U dynamic routing.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Mapping
import math


class RouterContractError(ValueError):
    """Raised when RouterResult invariants or contracts are violated."""
    pass


class _FrozenDict(dict):
    """Immutable dict subclass preserving dict equality and JSON serializability while blocking mutation."""

    def _immutable_error(self, *args, **kwargs):
        raise TypeError("RouterResult mapping container is immutable and cannot be modified.")

    __setitem__ = _immutable_error
    __delitem__ = _immutable_error
    pop = _immutable_error
    popitem = _immutable_error
    clear = _immutable_error
    update = _immutable_error
    setdefault = _immutable_error
    __ior__ = _immutable_error


class _FrozenList(list):
    """Immutable list subclass preserving list equality and isinstance(x, list) while blocking mutation."""

    def _immutable_error(self, *args, **kwargs):
        raise TypeError("RouterResult.active_modalities is immutable and cannot be modified.")

    __setitem__ = _immutable_error
    __delitem__ = _immutable_error
    append = _immutable_error
    extend = _immutable_error
    insert = _immutable_error
    pop = _immutable_error
    remove = _immutable_error
    clear = _immutable_error
    sort = _immutable_error
    reverse = _immutable_error
    __iadd__ = _immutable_error
    __imul__ = _immutable_error


@dataclass(frozen=True)
class RouterResult:
    """
    Immutable research-grade result container emitted by ACARA-U v2 dynamic router.
    All nested mapping and list containers are defensively copied and frozen on construction.
    
    Attributes:
        weights: Normalized modality weights w_i (sum to 1.0 across active modalities; 0.0 for unavailable).
        logits: Raw pre-masking logit scores z_i = alpha*C_i + beta*R_i - gamma*U_i + eta*Q_i.
        masked_logits: Post-masking logits z_tilde_i (equal to z_i if active, -inf if unavailable).
        availability: Map of modality availability flags A_i.
        coefficients: Router hyperparameters Theta = (alpha, beta, gamma, eta) applied.
        active_modalities: Immutable list of available modality identifiers (|A| in 0..3).
        num_active: Number of active modalities.
        normalization_sum: Sum of weights (1.0 for non-empty, 0.0 for zero-modality).
        routing_entropy: Modality weight dispersion entropy H(w) = -sum w_i * ln(w_i).
        dominant_modality: Modality with highest assigned routing weight (None if zero-modality).
        status: Routing status ('SUCCESS' or 'NO_MODALITY_AVAILABLE').
        router_version: Frozen router version string.
    """
    weights: Dict[str, float]
    logits: Dict[str, float]
    masked_logits: Dict[str, float]
    availability: Dict[str, bool]
    coefficients: Dict[str, float]
    active_modalities: List[str]
    num_active: int
    normalization_sum: float
    routing_entropy: float
    dominant_modality: Optional[str]
    status: str = "SUCCESS"
    router_version: str = "acarau_v2.0"

    def __post_init__(self) -> None:
        if self.status not in ("SUCCESS", "NO_MODALITY_AVAILABLE"):
            raise RouterContractError(f"RouterResult status must be 'SUCCESS' or 'NO_MODALITY_AVAILABLE', got '{self.status}'.")
        if not isinstance(self.router_version, str) or not self.router_version.strip():
            raise RouterContractError("router_version must be a non-empty string.")
        if not isinstance(self.num_active, int) or isinstance(self.num_active, bool):
            raise RouterContractError(f"num_active must be an integer (excluding bool), got {self.num_active!r}.")

        canonical_modalities = {"retina", "foot", "clinical"}
        canonical_coeff_keys = {"alpha", "beta", "gamma", "eta"}

        # 1. Validate active_modalities list, uniqueness, and canonical domain
        if not isinstance(self.active_modalities, (list, tuple)):
            raise RouterContractError("active_modalities must be a list or tuple.")
        active_list = list(self.active_modalities)
        for m in active_list:
            if not isinstance(m, str) or m not in canonical_modalities:
                raise RouterContractError(f"Unknown or non-string modality '{m}' in active_modalities; must be in {sorted(canonical_modalities)}.")
        if len(active_list) != len(set(active_list)):
            raise RouterContractError(f"Duplicate entries in active_modalities: {active_list}.")

        if self.num_active != len(active_list):
            raise RouterContractError(
                f"num_active ({self.num_active}) does not match len(active_modalities) ({len(active_list)})."
            )

        # 2. Validate channel map keys, strict boolean availability, and coefficients
        if not isinstance(self.weights, (dict, Mapping)) or set(self.weights.keys()) != canonical_modalities:
            raise RouterContractError(f"weights must contain exactly keys {sorted(canonical_modalities)}.")
        if not isinstance(self.logits, (dict, Mapping)) or set(self.logits.keys()) != canonical_modalities:
            raise RouterContractError(f"logits must contain exactly keys {sorted(canonical_modalities)}.")
        if not isinstance(self.masked_logits, (dict, Mapping)) or set(self.masked_logits.keys()) != canonical_modalities:
            raise RouterContractError(f"masked_logits must contain exactly keys {sorted(canonical_modalities)}.")
        if not isinstance(self.availability, (dict, Mapping)) or set(self.availability.keys()) != canonical_modalities:
            raise RouterContractError(f"availability must contain exactly keys {sorted(canonical_modalities)}.")
        for mod, is_av in self.availability.items():
            if not isinstance(is_av, bool):
                raise RouterContractError(f"availability['{mod}'] must be a strict boolean, got {is_av!r}.")

        if not isinstance(self.coefficients, (dict, Mapping)) or set(self.coefficients.keys()) != canonical_coeff_keys:
            raise RouterContractError(f"coefficients must contain exactly keys {sorted(canonical_coeff_keys)}.")
        for c_key, c_val in self.coefficients.items():
            if not isinstance(c_val, (int, float)) or isinstance(c_val, bool) or not math.isfinite(c_val):
                raise RouterContractError(f"coefficients['{c_key}'] must be a finite float, got {c_val!r}.")

        for mod in canonical_modalities:
            raw_z = self.logits[mod]
            mask_z = self.masked_logits[mod]
            if not isinstance(raw_z, (int, float)) or isinstance(raw_z, bool) or math.isnan(raw_z):
                raise RouterContractError(f"logits['{mod}'] must be a non-NaN numeric value, got {raw_z!r}.")
            if not isinstance(mask_z, (int, float)) or isinstance(mask_z, bool) or math.isnan(mask_z):
                raise RouterContractError(f"masked_logits['{mod}'] must be a non-NaN numeric value, got {mask_z!r}.")

        avail_true_set = {m for m, is_av in self.availability.items() if is_av is True}
        if avail_true_set != set(active_list):
            raise RouterContractError(
                f"active_modalities {active_list} disagrees with availability map {dict(self.availability)}."
            )

        # 3. Check state-specific invariants
        if self.status == "NO_MODALITY_AVAILABLE":
            if self.num_active != 0 or len(active_list) != 0:
                raise RouterContractError("NO_MODALITY_AVAILABLE status must have 0 active modalities.")
            for mod, w in self.weights.items():
                if not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w) or abs(w) > 1e-7:
                    raise RouterContractError(f"Weight for '{mod}' must be 0.0 in NO_MODALITY_AVAILABLE state, got {w}.")
            for mod, mask_z in self.masked_logits.items():
                if mask_z != float("-inf"):
                    raise RouterContractError(
                        f"masked_logits['{mod}'] must be -inf in NO_MODALITY_AVAILABLE state, got {mask_z}."
                    )
            if not isinstance(self.normalization_sum, (int, float)) or isinstance(self.normalization_sum, bool) or not math.isfinite(self.normalization_sum) or abs(self.normalization_sum) > 1e-7:
                raise RouterContractError(f"normalization_sum must be 0.0 in NO_MODALITY_AVAILABLE state, got {self.normalization_sum}.")
            if not isinstance(self.routing_entropy, (int, float)) or isinstance(self.routing_entropy, bool) or not math.isfinite(self.routing_entropy) or abs(self.routing_entropy) > 1e-7:
                raise RouterContractError(f"routing_entropy must be 0.0 in NO_MODALITY_AVAILABLE state, got {self.routing_entropy}.")
            if self.dominant_modality is not None:
                raise RouterContractError("Dominant modality must be None when no modalities are available.")
        else:
            if self.num_active == 0:
                raise RouterContractError("SUCCESS status requires at least 1 active modality.")
            for mod, w in self.weights.items():
                if not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w):
                    raise RouterContractError(f"Modality '{mod}' weight must be finite float, got {w}.")
                if w < 0.0 - 1e-7 or w > 1.0 + 1e-7:
                    raise RouterContractError(f"Modality '{mod}' weight {w:.8f} out of bounds [0.0, 1.0].")
                if not self.availability[mod] and abs(w) > 1e-7:
                    raise RouterContractError(f"Unavailable modality '{mod}' assigned non-zero weight {w:.8f}.")

                # Logit and masked_logit relationship checks
                raw_z = self.logits[mod]
                mask_z = self.masked_logits[mod]
                if not math.isfinite(raw_z):
                    raise RouterContractError(f"Active-state raw logit logits['{mod}'] must be finite, got {raw_z}.")
                if self.availability[mod]:
                    if not math.isfinite(mask_z) or abs(raw_z - mask_z) > 1e-7:
                        raise RouterContractError(
                            f"Active modality '{mod}' must have masked_logits == logits ({raw_z}), got {mask_z}."
                        )
                else:
                    if mask_z != float("-inf"):
                        raise RouterContractError(
                            f"Unavailable modality '{mod}' must have masked_logits == -inf, got {mask_z}."
                        )

            # Normalization invariant: verify actual sum
            actual_sum = float(sum(self.weights.values()))
            if abs(actual_sum - 1.0) > 1e-6:
                raise RouterContractError(f"Actual weights must sum to 1.0, got {actual_sum:.8f}.")
            if not isinstance(self.normalization_sum, (int, float)) or isinstance(self.normalization_sum, bool) or not math.isfinite(self.normalization_sum) or abs(self.normalization_sum - 1.0) > 1e-6:
                raise RouterContractError(f"Weights must sum to 1.0, got {self.normalization_sum}.")

            # Routing entropy finiteness, bounds, and exact Shannon formula agreement
            if not isinstance(self.routing_entropy, (int, float)) or isinstance(self.routing_entropy, bool) or not math.isfinite(self.routing_entropy):
                raise RouterContractError(f"routing_entropy must be a finite float, got {self.routing_entropy!r}.")
            if self.routing_entropy < -1e-7:
                raise RouterContractError(f"routing_entropy must be non-negative, got {self.routing_entropy}.")
            max_entropy = math.log(self.num_active) if self.num_active > 1 else 0.0
            if self.routing_entropy > max_entropy + 1e-5:
                raise RouterContractError(
                    f"routing_entropy ({self.routing_entropy:.6f}) exceeds maximum ({max_entropy:.6f}) for {self.num_active} active modalities."
                )
            expected_entropy = 0.0
            for m in active_list:
                w_val = float(self.weights[m])
                if w_val > 1e-12:
                    expected_entropy -= w_val * math.log(w_val)
            if abs(float(self.routing_entropy) - expected_entropy) > 1e-5:
                raise RouterContractError(
                    f"routing_entropy ({self.routing_entropy:.6f}) does not match Shannon entropy of weights ({expected_entropy:.6f})."
                )

            # Dominant modality consistency
            if not isinstance(self.dominant_modality, str) or self.dominant_modality not in active_list:
                raise RouterContractError(
                    f"dominant_modality must be an active modality in {active_list}, got {self.dominant_modality!r}."
                )
            max_active_weight = max(self.weights[m] for m in active_list)
            if abs(self.weights[self.dominant_modality] - max_active_weight) > 1e-7:
                raise RouterContractError(
                    f"dominant_modality '{self.dominant_modality}' weight ({self.weights[self.dominant_modality]:.6f}) "
                    f"does not match maximum active weight ({max_active_weight:.6f})."
                )

        # 4. Defensively copy and freeze all nested containers
        object.__setattr__(self, "weights", _FrozenDict({str(k): float(v) for k, v in self.weights.items()}))
        object.__setattr__(self, "logits", _FrozenDict({str(k): float(v) for k, v in self.logits.items()}))
        object.__setattr__(self, "masked_logits", _FrozenDict({str(k): float(v) for k, v in self.masked_logits.items()}))
        object.__setattr__(self, "availability", _FrozenDict({str(k): bool(v) for k, v in self.availability.items()}))
        object.__setattr__(self, "coefficients", _FrozenDict({str(k): float(v) for k, v in self.coefficients.items()}))
        object.__setattr__(self, "active_modalities", _FrozenList(active_list))

    @classmethod
    def no_modality_available(
        cls,
        coefficients: Dict[str, float],
        router_version: str = "acarau_v2.0",
    ) -> "RouterResult":
        """Constructs a graceful failure result when all modalities are missing."""
        return cls(
            weights={"retina": 0.0, "foot": 0.0, "clinical": 0.0},
            logits={"retina": float("-inf"), "foot": float("-inf"), "clinical": float("-inf")},
            masked_logits={"retina": float("-inf"), "foot": float("-inf"), "clinical": float("-inf")},
            availability={"retina": False, "foot": False, "clinical": False},
            coefficients=coefficients,
            active_modalities=[],
            num_active=0,
            normalization_sum=0.0,
            routing_entropy=0.0,
            dominant_modality=None,
            status="NO_MODALITY_AVAILABLE",
            router_version=router_version,
        )

    def as_routing_result(self, purpose: str = "computational_attention_allocation"):
        """Converts to semantic RoutingResult dataclass for EvidenceReport."""
        from src.fusion.contracts.evidence_report import RoutingResult
        return RoutingResult(
            weights={m: round(float(w), 6) for m, w in self.weights.items()},
            active_modalities=list(self.active_modalities),
            routing_policy_version=self.router_version,
            purpose=purpose,
            entropy=round(float(self.routing_entropy), 6),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result into a JSON-compatible dictionary."""
        return {
            "weights": {m: round(float(w), 6) for m, w in self.weights.items()},
            "logits": {m: round(float(z), 6) if math.isfinite(z) else str(z) for m, z in self.logits.items()},
            "masked_logits": {m: round(float(z), 6) if math.isfinite(z) else str(z) for m, z in self.masked_logits.items()},
            "availability": dict(self.availability),
            "coefficients": dict(self.coefficients),
            "active_modalities": list(self.active_modalities),
            "num_active": self.num_active,
            "normalization_sum": round(float(self.normalization_sum), 6),
            "routing_entropy": round(float(self.routing_entropy), 6),
            "dominant_modality": self.dominant_modality,
            "status": self.status,
            "router_version": self.router_version,
        }
