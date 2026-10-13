"""
FusionMedAI - Phase C11.6: DCRI Result Contract & Schema (Historical Research Benchmark)
Defines the immutable output container for historical C11 DCRI risk aggregation and uncertainty discounting.
Enforces all structural, boundedness, conservation, active-channel factorization, and inactive-modality
zeroing invariants across all five channel maps directly in __post_init__, and provides
`validate_strict_invariants()` for explicit on-demand re-verification.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Mapping
import math


class DCRIContractError(ValueError):
    """Raised when DCRI result invariants or contracts are violated."""
    pass


class _FrozenDict(dict):
    """Immutable dict subclass preserving dict equality and JSON serializability while blocking mutation."""

    def _immutable_error(self, *args, **kwargs):
        raise TypeError("DCRIResult mapping container is immutable and cannot be modified.")

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
        raise TypeError("DCRIResult.active_modalities is immutable and cannot be modified.")

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
class DCRIResult:
    """
    Immutable result object produced by historical C11 DCRI risk aggregation.
    All nested mapping and list containers are defensively copied and frozen on construction.
    Validates structural, boundedness, partition-of-unity, sum-conservation, active-channel
    pairwise factorization, and inactive-channel zeroing across all five channel maps
    (modality_weights, modality_risks, weighted_risk_contributions, modality_uncertainties,
    and uncertainty_penalty_contributions) on construction in `__post_init__`;
    `validate_strict_invariants()` provides an explicit on-demand re-verification method.
    
    Attributes:
        packet_id: Identifier of the evaluated decision packet.
        r_fusion: Weighted fused risk index in [0.0, 1.0].
        u_sum: Sum of predictive uncertainties across active modalities: sum_{i in A} U_i.
        u_mean: Mean predictive uncertainty across active modalities: (1 / |A|) sum_{i in A} U_i.
        uncertainty_penalty: Uncertainty discount P_U(delta) = delta * u_sum.
        dcri: Derived Decision-Critical Risk Index DCRI = r_fusion - P_U(delta) in [-delta*|A|, 1.0].
        delta: Uncertainty penalty scaling hyperparameter delta >= 0.0 (provisional convenience default 0.20; not locked).
        modality_weights: Modality authority weights w_i summing to 1.0 over active modalities.
        modality_risks: Modality scalar risk projections r_i in [0.0, 1.0].
        weighted_risk_contributions: Decomposition K_i = w_i * r_i summing to r_fusion.
        modality_uncertainties: Predictive uncertainty U_i in [0.0, 1.0].
        uncertainty_penalty_contributions: Decomposition p_i = delta * U_i summing to uncertainty_penalty.
        active_modalities: Immutable list of available modality identifiers.
        num_active: Count of available modalities |A|.
        status: Evaluation status ('SUCCESS' or 'NO_MODALITY_AVAILABLE').
    """
    packet_id: str
    r_fusion: float
    u_sum: float
    u_mean: float
    uncertainty_penalty: float
    dcri: float
    delta: float
    modality_weights: Dict[str, float]
    modality_risks: Dict[str, float]
    weighted_risk_contributions: Dict[str, float]
    modality_uncertainties: Dict[str, float]
    uncertainty_penalty_contributions: Dict[str, float]
    active_modalities: List[str]
    num_active: int
    status: str = "SUCCESS"

    def _freeze_containers(self, active_list: List[str]) -> None:
        object.__setattr__(self, "modality_weights", _FrozenDict({str(k): float(v) for k, v in self.modality_weights.items()}))
        object.__setattr__(self, "modality_risks", _FrozenDict({str(k): float(v) for k, v in self.modality_risks.items()}))
        object.__setattr__(self, "weighted_risk_contributions", _FrozenDict({str(k): float(v) for k, v in self.weighted_risk_contributions.items()}))
        object.__setattr__(self, "modality_uncertainties", _FrozenDict({str(k): float(v) for k, v in self.modality_uncertainties.items()}))
        object.__setattr__(self, "uncertainty_penalty_contributions", _FrozenDict({str(k): float(v) for k, v in self.uncertainty_penalty_contributions.items()}))
        object.__setattr__(self, "active_modalities", _FrozenList(active_list))

    def __post_init__(self) -> None:
        # 0. Status and Packet ID Validation
        if not isinstance(self.packet_id, str) or not self.packet_id.strip():
            raise DCRIContractError("Field 'packet_id' must be a non-empty string.")
        if self.status not in ("SUCCESS", "NO_MODALITY_AVAILABLE"):
            raise DCRIContractError(
                f"Field 'status' must be 'SUCCESS' or 'NO_MODALITY_AVAILABLE', got {self.status!r}."
            )
        if not isinstance(self.num_active, int) or isinstance(self.num_active, bool):
            raise DCRIContractError(f"Field 'num_active' must be an integer, got {self.num_active!r}.")

        # 1. Delta Validation (must be valid finite float >= 0 across all states including NO_MODALITY_AVAILABLE)
        if not isinstance(self.delta, (int, float)) or isinstance(self.delta, bool) or math.isnan(self.delta) or math.isinf(self.delta):
            raise DCRIContractError(f"Field 'delta' must be finite float, got {self.delta}.")
        if self.delta < 0.0:
            raise DCRIContractError(f"Field 'delta' must be non-negative, got {self.delta}.")

        all_canonical_modalities = ("retina", "foot", "clinical")
        if not isinstance(self.active_modalities, (list, tuple)):
            raise DCRIContractError("active_modalities must be a list or tuple.")
        active_list = list(self.active_modalities)
        if len(active_list) != len(set(active_list)):
            raise DCRIContractError(f"Duplicate modalities in active_modalities: {active_list}.")
        for m in active_list:
            if not isinstance(m, str) or m not in all_canonical_modalities:
                raise DCRIContractError(f"Unknown modality '{m}' in active_modalities.")

        # Validate exact keys and finite non-boolean numeric values across all 5 channel maps in all states
        for map_name, ch_map in [
            ("modality_weights", self.modality_weights),
            ("modality_risks", self.modality_risks),
            ("weighted_risk_contributions", self.weighted_risk_contributions),
            ("modality_uncertainties", self.modality_uncertainties),
            ("uncertainty_penalty_contributions", self.uncertainty_penalty_contributions),
        ]:
            if not isinstance(ch_map, (dict, Mapping)) or set(ch_map.keys()) != set(all_canonical_modalities):
                raise DCRIContractError(
                    f"{map_name} must be a mapping containing exactly keys {all_canonical_modalities}."
                )
            for m, v in ch_map.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v) or math.isinf(v):
                    raise DCRIContractError(
                        f"{map_name}['{m}'] must be a finite float (excluding bool), got {v!r}."
                    )

        # 2. Zero-Modality Rejection Handling
        if self.status == "NO_MODALITY_AVAILABLE":
            if self.num_active != 0:
                raise DCRIContractError("NO_MODALITY_AVAILABLE status must have num_active == 0.")
            if len(active_list) != 0:
                raise DCRIContractError("NO_MODALITY_AVAILABLE status must have empty active_modalities.")
            for name, val in [
                ("r_fusion", self.r_fusion),
                ("u_sum", self.u_sum),
                ("u_mean", self.u_mean),
                ("uncertainty_penalty", self.uncertainty_penalty),
                ("dcri", self.dcri),
            ]:
                if not isinstance(val, (int, float)) or isinstance(val, bool) or math.isnan(val) or math.isinf(val) or abs(val) > 1e-7:
                    raise DCRIContractError(f"{name} must be 0.0 in NO_MODALITY_AVAILABLE status, got {val}.")

            for map_name, ch_map in [
                ("modality_weights", self.modality_weights),
                ("modality_risks", self.modality_risks),
                ("weighted_risk_contributions", self.weighted_risk_contributions),
                ("modality_uncertainties", self.modality_uncertainties),
                ("uncertainty_penalty_contributions", self.uncertainty_penalty_contributions),
            ]:
                for m, v in ch_map.items():
                    if abs(v) > 1e-7:
                        raise DCRIContractError(
                            f"{map_name}['{m}'] must be 0.0 in NO_MODALITY_AVAILABLE status, got {v}."
                        )

            self._freeze_containers(active_list)
            return

        # 3. Modality Count Alignment
        if self.num_active != len(active_list):
            raise DCRIContractError(
                f"num_active ({self.num_active}) does not match len(active_modalities) ({len(active_list)})."
            )
        if self.num_active == 0:
            raise DCRIContractError("SUCCESS status requires at least 1 active modality.")

        # 4. Finite Scalar Checks for active states
        for name, val in [
            ("r_fusion", self.r_fusion),
            ("u_sum", self.u_sum),
            ("u_mean", self.u_mean),
            ("uncertainty_penalty", self.uncertainty_penalty),
            ("dcri", self.dcri),
        ]:
            if not isinstance(val, (int, float)) or isinstance(val, bool) or math.isnan(val) or math.isinf(val):
                raise DCRIContractError(f"Field '{name}' must be finite float, got {val}.")

        if self.r_fusion < -1e-7 or self.r_fusion > 1.0 + 1e-7:
            raise DCRIContractError(f"r_fusion = {self.r_fusion:.6f} exceeds valid bounds [0.0, 1.0].")

        # 5. Modality Channel Validation (Bounds, Inactive Zeroing, and Active Pairwise Factorization)
        active_set = set(active_list)

        for m in all_canonical_modalities:
            w = self.modality_weights[m]
            r = self.modality_risks[m]
            u = self.modality_uncertainties[m]
            k = self.weighted_risk_contributions[m]
            p = self.uncertainty_penalty_contributions[m]

            # Boundedness checks
            if w < -1e-7 or w > 1.0 + 1e-7:
                raise DCRIContractError(f"Modality '{m}' weight ({w:.6f}) out of [0.0, 1.0].")
            if r < -1e-7 or r > 1.0 + 1e-7:
                raise DCRIContractError(f"Modality '{m}' risk ({r:.6f}) out of [0.0, 1.0].")
            if u < -1e-7 or u > 1.0 + 1e-7:
                raise DCRIContractError(f"Modality '{m}' uncertainty ({u:.6f}) out of [0.0, 1.0].")

            if m not in active_set:
                # Inactive Modality Invariant: w_i = 0, r_i = 0, U_i = 0, K_i = 0, p_i = 0
                if abs(w) > 1e-7:
                    raise DCRIContractError(f"Inactive modality '{m}' has non-zero weight {w:.6f}.")
                if abs(r) > 1e-7:
                    raise DCRIContractError(f"Inactive modality '{m}' has non-zero risk {r:.6f}.")
                if abs(u) > 1e-7:
                    raise DCRIContractError(f"Inactive modality '{m}' has non-zero uncertainty {u:.6f}.")
                if abs(k) > 1e-7:
                    raise DCRIContractError(f"Inactive modality '{m}' has non-zero risk contribution {k:.6f}.")
                if abs(p) > 1e-7:
                    raise DCRIContractError(f"Inactive modality '{m}' has non-zero penalty contribution {p:.6f}.")
            else:
                # Pairwise factorization for active modality: K_i == w_i * r_i and p_i == delta * U_i
                if abs(k - (w * r)) > 1e-5:
                    raise DCRIContractError(
                        f"Weighted contribution K_{m} = {k:.6f} does not equal w_{m} * r_{m} = {w * r:.6f}."
                    )
                if abs(p - (self.delta * u)) > 1e-5:
                    raise DCRIContractError(
                        f"Penalty contribution p_{m} = {p:.6f} does not equal delta * U_{m} = {self.delta * u:.6f}."
                    )

        # 6. Partition of Unity: sum_{i in A} w_i == 1.0
        active_weight_sum = sum(self.modality_weights[m] for m in active_list)
        if abs(active_weight_sum - 1.0) > 1e-5:
            raise DCRIContractError(
                f"Active modality weights sum ({active_weight_sum:.6f}) does not equal 1.0."
            )

        # 7. Conservation Invariant: sum K_i == r_fusion
        active_k_sum = sum(self.weighted_risk_contributions[m] for m in active_list)
        if abs(active_k_sum - self.r_fusion) > 1e-5:
            raise DCRIContractError(
                f"Weighted contribution sum ({active_k_sum:.6f}) does not match r_fusion ({self.r_fusion:.6f})."
            )

        # 8. Uncertainty Sum, Mean, and Penalty Factorization
        calc_u_sum = sum(self.modality_uncertainties[m] for m in active_list)
        if abs(calc_u_sum - self.u_sum) > 1e-5:
            raise DCRIContractError(
                f"u_sum = {self.u_sum:.6f} does not match sum of active uncertainties = {calc_u_sum:.6f}."
            )
        calc_u_mean = calc_u_sum / self.num_active
        if abs(calc_u_mean - self.u_mean) > 1e-5:
            raise DCRIContractError(
                f"u_mean = {self.u_mean:.6f} does not match u_sum / num_active = {calc_u_mean:.6f}."
            )
        if abs((self.delta * self.u_sum) - self.uncertainty_penalty) > 1e-5:
            raise DCRIContractError(
                f"uncertainty_penalty = {self.uncertainty_penalty:.6f} does not equal delta * u_sum = {self.delta * self.u_sum:.6f}."
            )

        active_p_sum = sum(self.uncertainty_penalty_contributions[m] for m in active_list)
        if abs(active_p_sum - self.uncertainty_penalty) > 1e-5:
            raise DCRIContractError(
                f"Penalty contribution sum ({active_p_sum:.6f}) does not match uncertainty_penalty ({self.uncertainty_penalty:.6f})."
            )

        # 9. DCRI Equation Invariant: DCRI == r_fusion - uncertainty_penalty
        expected_dcri = self.r_fusion - self.uncertainty_penalty
        if abs(self.dcri - expected_dcri) > 1e-5:
            raise DCRIContractError(
                f"DCRI ({self.dcri:.6f}) does not match r_fusion - penalty ({expected_dcri:.6f})."
            )

        # 10. DCRI Mathematical Bounds: [-delta * |A|, 1.0] (Explicitly unclamped)
        min_bound = - (self.delta * self.num_active) - 1e-6
        max_bound = 1.0 + 1e-6
        if self.dcri < min_bound or self.dcri > max_bound:
            raise DCRIContractError(
                f"DCRI = {self.dcri:.6f} out of theoretical bounds [{min_bound:.4f}, {max_bound:.4f}]."
            )

        self._freeze_containers(active_list)

    def validate_strict_invariants(self) -> None:
        """
        Performs exhaustive mathematical and relational invariant verification on this DCRIResult.
        Verifies:
        - Exact pairwise product factorization: K_i == w_i * r_i
        - Exact pairwise penalty factorization: p_i == delta * U_i
        - Composite uncertainty sum and mean alignment: u_sum == sum U_i, u_mean == u_sum / |A|
        - Exact penalty calculation: uncertainty_penalty == delta * u_sum
        - Modality uniqueness and canonical naming across all channels
        - Inactive channel zeroing across ALL maps (weights, risks, uncertainties, and contributions)
        - Complete zero-state verification across all scalars and maps when NO_MODALITY_AVAILABLE
        """
        canonical_modalities = ("retina", "foot", "clinical")

        # 1. Active modalities uniqueness and domain
        if len(self.active_modalities) != len(set(self.active_modalities)):
            raise DCRIContractError(
                f"Duplicate modalities in active_modalities: {self.active_modalities}."
            )
        for m in self.active_modalities:
            if m not in canonical_modalities:
                raise DCRIContractError(f"Unknown modality '{m}' in active_modalities.")

        if self.status == "NO_MODALITY_AVAILABLE":
            if self.num_active != 0 or len(self.active_modalities) != 0:
                raise DCRIContractError("NO_MODALITY_AVAILABLE must have 0 active modalities.")
            if abs(self.r_fusion) > 1e-7 or abs(self.dcri) > 1e-7 or abs(self.uncertainty_penalty) > 1e-7:
                raise DCRIContractError("r_fusion, dcri, and uncertainty_penalty must be 0.0 in NO_MODALITY_AVAILABLE.")
            if abs(self.u_sum) > 1e-7 or abs(self.u_mean) > 1e-7:
                raise DCRIContractError("u_sum and u_mean must be 0.0 in NO_MODALITY_AVAILABLE.")
            for map_name, ch_map in [
                ("modality_weights", self.modality_weights),
                ("modality_risks", self.modality_risks),
                ("weighted_risk_contributions", self.weighted_risk_contributions),
                ("modality_uncertainties", self.modality_uncertainties),
                ("uncertainty_penalty_contributions", self.uncertainty_penalty_contributions),
            ]:
                for m, v in ch_map.items():
                    if abs(v) > 1e-7:
                        raise DCRIContractError(
                            f"{map_name}['{m}'] must be 0.0 in NO_MODALITY_AVAILABLE, got {v:.6f}."
                        )
            return

        active_set = set(self.active_modalities)

        # 2. Pairwise factorizations for active modalities and strict zeroing for inactive modalities
        for m in canonical_modalities:
            w_i = self.modality_weights.get(m, 0.0)
            r_i = self.modality_risks.get(m, 0.0)
            u_i = self.modality_uncertainties.get(m, 0.0)
            k_i = self.weighted_risk_contributions.get(m, 0.0)
            p_i = self.uncertainty_penalty_contributions.get(m, 0.0)

            if m not in active_set:
                if abs(w_i) > 1e-7 or abs(r_i) > 1e-7 or abs(u_i) > 1e-7 or abs(k_i) > 1e-7 or abs(p_i) > 1e-7:
                    raise DCRIContractError(
                        f"Inactive modality '{m}' must have all zero channel entries (w={w_i}, r={r_i}, u={u_i}, k={k_i}, p={p_i})."
                    )
            else:
                expected_k = w_i * r_i
                if abs(k_i - expected_k) > 1e-5:
                    raise DCRIContractError(
                        f"Weighted contribution K_{m} = {k_i:.6f} does not equal w_{m} * r_{m} = {expected_k:.6f}."
                    )

                expected_p = self.delta * u_i
                if abs(p_i - expected_p) > 1e-5:
                    raise DCRIContractError(
                        f"Penalty contribution p_{m} = {p_i:.6f} does not equal delta * U_{m} = {expected_p:.6f}."
                    )

        # 3. Composite uncertainty sum and mean alignment
        actual_u_sum = sum(self.modality_uncertainties.get(m, 0.0) for m in self.active_modalities)
        if abs(self.u_sum - actual_u_sum) > 1e-5:
            raise DCRIContractError(
                f"u_sum ({self.u_sum:.6f}) does not match sum of active uncertainties ({actual_u_sum:.6f})."
            )

        expected_u_mean = actual_u_sum / self.num_active if self.num_active > 0 else 0.0
        if abs(self.u_mean - expected_u_mean) > 1e-5:
            raise DCRIContractError(
                f"u_mean ({self.u_mean:.6f}) does not match u_sum / |A| ({expected_u_mean:.6f})."
            )

        expected_penalty = self.delta * self.u_sum
        if abs(self.uncertainty_penalty - expected_penalty) > 1e-5:
            raise DCRIContractError(
                f"uncertainty_penalty ({self.uncertainty_penalty:.6f}) does not match delta * u_sum ({expected_penalty:.6f})."
            )

    @classmethod
    def no_modality_available(cls, packet_id: str = "EMPTY_PACKET", delta: float = 0.2) -> "DCRIResult":
        """Factory method for zero-modality safe rejection."""
        return cls(
            packet_id=packet_id,
            r_fusion=0.0,
            u_sum=0.0,
            u_mean=0.0,
            uncertainty_penalty=0.0,
            dcri=0.0,
            delta=float(delta),
            modality_weights={"retina": 0.0, "foot": 0.0, "clinical": 0.0},
            modality_risks={"retina": 0.0, "foot": 0.0, "clinical": 0.0},
            weighted_risk_contributions={"retina": 0.0, "foot": 0.0, "clinical": 0.0},
            modality_uncertainties={"retina": 0.0, "foot": 0.0, "clinical": 0.0},
            uncertainty_penalty_contributions={"retina": 0.0, "foot": 0.0, "clinical": 0.0},
            active_modalities=[],
            num_active=0,
            status="NO_MODALITY_AVAILABLE",
        )

    def as_research_benchmark_index(self):
        """Converts to ResearchFusionIndex with explicit non-clinical research disclaimer."""
        from src.fusion.contracts.evidence_report import ResearchFusionIndex
        return ResearchFusionIndex(
            index_value=round(float(self.r_fusion), 6),
            dcri_value=round(float(self.dcri), 6),
            weights={m: round(float(w), 6) for m, w in self.modality_weights.items()},
            uncertainty_penalty=round(float(self.uncertainty_penalty), 6),
            is_clinical_probability=False,
            clinical_action_authorized=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes DCRIResult to canonical dictionary representation."""
        return {
            "packet_id": self.packet_id,
            "r_fusion": round(float(self.r_fusion), 6),
            "u_sum": round(float(self.u_sum), 6),
            "u_mean": round(float(self.u_mean), 6),
            "uncertainty_penalty": round(float(self.uncertainty_penalty), 6),
            "dcri": round(float(self.dcri), 6),
            "delta": round(float(self.delta), 6),
            "modality_weights": {m: round(float(w), 6) for m, w in self.modality_weights.items()},
            "modality_risks": {m: round(float(r), 6) for m, r in self.modality_risks.items()},
            "weighted_risk_contributions": {m: round(float(k), 6) for m, k in self.weighted_risk_contributions.items()},
            "modality_uncertainties": {m: round(float(u), 6) for m, u in self.modality_uncertainties.items()},
            "uncertainty_penalty_contributions": {m: round(float(p), 6) for m, p in self.uncertainty_penalty_contributions.items()},
            "active_modalities": list(self.active_modalities),
            "num_active": int(self.num_active),
            "status": self.status,
            "interpretation_disclaimer": "EXPERIMENTAL_BENCHMARK_ONLY: Does not represent unified clinical risk probability.",
        }
