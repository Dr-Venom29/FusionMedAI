"""
FusionMedAI - Phase C11.6: DCRI Result Contract & Schema
Defines the immutable output container for DCRI risk aggregation and uncertainty discounting.
Strictly self-validating across all mathematical invariants and modality channels.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import math


class DCRIContractError(ValueError):
    """Raised when DCRI result invariants or contracts are violated."""
    pass


@dataclass(frozen=True)
class DCRIResult:
    """
    Immutable, self-validating result object produced by DCRI risk aggregation.
    
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
        active_modalities: List of available modality identifiers.
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

    def __post_init__(self) -> None:
        # 1. Delta Validation (must be valid finite float >= 0 across all states including NO_MODALITY_AVAILABLE)
        if not isinstance(self.delta, (int, float)) or math.isnan(self.delta) or math.isinf(self.delta):
            raise DCRIContractError(f"Field 'delta' must be finite float, got {self.delta}.")
        if self.delta < 0.0:
            raise DCRIContractError(f"Field 'delta' must be non-negative, got {self.delta}.")

        # 2. Zero-Modality Rejection Handling
        if self.status == "NO_MODALITY_AVAILABLE":
            if self.num_active != 0:
                raise DCRIContractError("NO_MODALITY_AVAILABLE status must have num_active == 0.")
            if len(self.active_modalities) != 0:
                raise DCRIContractError("NO_MODALITY_AVAILABLE status must have empty active_modalities.")
            if self.r_fusion != 0.0 or self.dcri != 0.0:
                raise DCRIContractError("r_fusion and dcri must be 0.0 in NO_MODALITY_AVAILABLE status.")
            return

        # 3. Modality Count Alignment
        if self.num_active != len(self.active_modalities):
            raise DCRIContractError(
                f"num_active ({self.num_active}) does not match len(active_modalities) ({len(self.active_modalities)})."
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
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise DCRIContractError(f"Field '{name}' must be finite float, got {val}.")

        if self.r_fusion < -1e-7 or self.r_fusion > 1.0 + 1e-7:
            raise DCRIContractError(f"r_fusion = {self.r_fusion:.6f} exceeds valid bounds [0.0, 1.0].")

        # 5. Modality Channel Validation (Weights, Risks, Uncertainties, Inactive Zeroing)
        all_canonical_modalities = ("retina", "foot", "clinical")
        active_set = set(self.active_modalities)

        for m in all_canonical_modalities:
            w = self.modality_weights.get(m, 0.0)
            r = self.modality_risks.get(m, 0.0)
            u = self.modality_uncertainties.get(m, 0.0)
            k = self.weighted_risk_contributions.get(m, 0.0)
            p = self.uncertainty_penalty_contributions.get(m, 0.0)

            # Finite checks
            for var_name, var_val in [("weight", w), ("risk", r), ("uncertainty", u), ("k_contrib", k), ("p_contrib", p)]:
                if not isinstance(var_val, (int, float)) or math.isnan(var_val) or math.isinf(var_val):
                    raise DCRIContractError(f"Modality '{m}' {var_name} must be finite float, got {var_val}.")

            # Boundedness checks
            if w < -1e-7 or w > 1.0 + 1e-7:
                raise DCRIContractError(f"Modality '{m}' weight ({w:.6f}) out of [0.0, 1.0].")
            if r < -1e-7 or r > 1.0 + 1e-7:
                raise DCRIContractError(f"Modality '{m}' risk ({r:.6f}) out of [0.0, 1.0].")
            if u < -1e-7 or u > 1.0 + 1e-7:
                raise DCRIContractError(f"Modality '{m}' uncertainty ({u:.6f}) out of [0.0, 1.0].")

            if m not in active_set:
                # Inactive Modality Invariant: w_i = 0, K_i = 0, p_i = 0
                if abs(w) > 1e-7:
                    raise DCRIContractError(f"Inactive modality '{m}' has non-zero weight {w:.6f}.")
                if abs(k) > 1e-7:
                    raise DCRIContractError(f"Inactive modality '{m}' has non-zero risk contribution {k:.6f}.")
                if abs(p) > 1e-7:
                    raise DCRIContractError(f"Inactive modality '{m}' has non-zero penalty contribution {p:.6f}.")

        # 6. Partition of Unity: sum_{i in A} w_i == 1.0
        active_weight_sum = sum(self.modality_weights.get(m, 0.0) for m in self.active_modalities)
        if abs(active_weight_sum - 1.0) > 1e-5:
            raise DCRIContractError(
                f"Active modality weights sum ({active_weight_sum:.6f}) does not equal 1.0."
            )

        # 7. Conservation Invariant: sum K_i == r_fusion
        active_k_sum = sum(self.weighted_risk_contributions.get(m, 0.0) for m in self.active_modalities)
        if abs(active_k_sum - self.r_fusion) > 1e-5:
            raise DCRIContractError(
                f"Weighted contribution sum ({active_k_sum:.6f}) does not match r_fusion ({self.r_fusion:.6f})."
            )

        # 8. Conservation Invariant: sum p_i == uncertainty_penalty
        active_p_sum = sum(self.uncertainty_penalty_contributions.get(m, 0.0) for m in self.active_modalities)
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
        }
