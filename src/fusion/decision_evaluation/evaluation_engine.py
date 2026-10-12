"""
FusionMedAI - Phase C11.17: Final Decision-Level Evaluation Engine
Evaluates complete frozen ACARA-U policy and comparator baselines B1–B6
across baseline comparisons, availability regimes, input degradation,
cross-modality conflict, and calibration conditions.
"""

from typing import Dict, Any, List, Optional, Tuple, Sequence
import math
import numpy as np

from src.fusion.baselines.decision_packet import ControlledDecisionPacket, FusionResult, ModalityRecord
from src.fusion.baselines.fusion_runner import FusionRunner
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router.router_input import RouterInput, ModalityChannelInput
from src.fusion.dcri.dcri_engine import DCRIEngine
from src.fusion.decision_evaluation.evaluation_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    DELTA_FROZEN,
    TAU_1_DEFAULT,
    TAU_2_DEFAULT,
    ALL_POLICIES,
    POLICY_B1,
    POLICY_B2,
    POLICY_B3,
    POLICY_B4,
    POLICY_B5,
    POLICY_B6,
    ALL_REGIMES,
    REGIME_EMPTY,
    REGIME_RFC,
    REGIME_MODALITY_MAP,
    MODALITIES,
    ALL_DEGRADATIONS,
    DEGRADATION_D0,
    DEGRADATION_D1,
    DEGRADATION_D2,
    DEGRADATION_D3,
    TIER_0_ROUTINE,
    TIER_1_ASSESSMENT,
    TIER_2_ESCALATION,
    TIER_NAMES,
    TIER_KEYS,
)
from src.fusion.decision_evaluation.evaluation_metrics import (
    calc_stats,
    compute_conflict_metrics,
    compute_dcri,
    assign_decision_tier,
)
from src.fusion.calibration.calibration_condition import (
    create_uncalibrated_decision_packet,
    create_calibrated_decision_packet,
)
from src.fusion.degradation.degradation_engine import (
    apply_scenario_degradation,
    get_live_degraded_modality_quality,
    apply_degradation_to_packet,
)
from src.fusion.degradation.degradation_spec import (
    SCENARIO_ALL_MODALITIES,
    RETINA_OPERATORS,
    FOOT_OPERATORS,
    CLINICAL_OPERATORS,
)


def validate_record(
    record: Dict[str, Any],
    delta: float = DELTA_FROZEN,
    tau_1: float = TAU_1_DEFAULT,
    tau_2: float = TAU_2_DEFAULT,
) -> None:
    """
    Validates mathematical and structural invariants of an evaluated decision record.
    Raises ValueError if any invariant is violated.
    """
    pkt_id = record.get("packet_id", "UNKNOWN")

    # Reject if is_finite is False
    if not record.get("is_finite", False):
        raise ValueError(f"Packet {pkt_id}: is_finite is False (error: {record.get('error')})")

    status = record.get("status")
    num_active = record.get("num_active", 0)
    active_modalities = record.get("active_modalities", [])
    weights = record.get("weights", {})

    if status == "NO_MODALITY_AVAILABLE":
        if num_active != 0 or len(active_modalities) != 0:
            raise ValueError(f"Packet {pkt_id}: NO_MODALITY_AVAILABLE but num_active={num_active}, active={active_modalities}")
        for m in MODALITIES:
            if m not in weights:
                raise ValueError(f"Packet {pkt_id}: missing weight key {m} under NO_MODALITY_AVAILABLE")
            w_raw = weights[m]
            if w_raw is None or not math.isfinite(float(w_raw)) or abs(float(w_raw)) > 1e-9:
                raise ValueError(f"Packet {pkt_id}: non-zero or non-finite weight {m}={w_raw} under NO_MODALITY_AVAILABLE")
        for fld in ["r_fusion", "dcri", "uncertainty_penalty"]:
            val = record.get(fld)
            if val is None or not math.isfinite(float(val)) or abs(float(val)) > 1e-9:
                raise ValueError(f"Packet {pkt_id}: non-zero or non-finite {fld}={val} under empty regime")
        if record.get("decision_tier") != TIER_0_ROUTINE:
            raise ValueError(f"Packet {pkt_id}: non-tier 0 under empty regime: {record.get('decision_tier')}")
        return

    if status != "SUCCESS":
        raise ValueError(f"Packet {pkt_id}: status is {status}, expected SUCCESS")

    # Invariants on active modalities
    if num_active != len(active_modalities):
        raise ValueError(f"Packet {pkt_id}: num_active={num_active} != len(active_modalities)={len(active_modalities)}")
    if len(set(active_modalities)) != len(active_modalities):
        raise ValueError(f"Packet {pkt_id}: duplicate modalities in active_modalities: {active_modalities}")
    for m in active_modalities:
        if m not in MODALITIES:
            raise ValueError(f"Packet {pkt_id}: invalid modality in active_modalities: {m}")

    # All expected modality weight keys present and finite
    for m in MODALITIES:
        if m not in weights:
            raise ValueError(f"Packet {pkt_id}: missing weight key {m}")
        w_val = weights[m]
        if w_val is None or not math.isfinite(float(w_val)):
            raise ValueError(f"Packet {pkt_id}: weight {m} is not finite ({w_val})")

    # Validate inputs: risks, uncertainties, qualities, reliabilities, confidences
    for dict_name in ["input_risks", "input_uncertainties", "input_qualities", "input_reliabilities", "input_confidences"]:
        inp_dict = record.get(dict_name, {})
        for m in MODALITIES:
            if m not in inp_dict:
                raise ValueError(f"Packet {pkt_id}: missing {m} in {dict_name}")
            val = inp_dict[m]
            if val is None or not math.isfinite(float(val)):
                raise ValueError(f"Packet {pkt_id}: non-finite value in {dict_name}[{m}]: {val}")
            f_val = float(val)
            if f_val < -1e-6 or f_val > 1.0 + 1e-6:
                raise ValueError(f"Packet {pkt_id}: value out of bounds in {dict_name}[{m}]: {f_val}")

    if not math.isfinite(delta) or delta < 0.0 or delta > 1.0:
        raise ValueError(f"Invalid delta parameter: {delta} (must be finite in [0.0, 1.0])")

    # Numeric finite and range checks
    for field, min_val, max_val in [
        ("r_fusion", 0.0, 1.0),
        ("uncertainty_penalty", 0.0, 3.0 * delta),
        ("u_sum", 0.0, 3.0),
        ("routing_entropy", 0.0, math.log(3.0) + 1e-6),
        ("css", 0.0, 1.0),
        ("delta_max", 0.0, 1.0),
        ("sigma_w", 0.0, 1.0),
    ]:
        val = record.get(field)
        if val is None or not math.isfinite(float(val)):
            raise ValueError(f"Packet {pkt_id}: field {field} is not finite ({val})")
        f_val = float(val)
        if f_val < min_val - 1e-6 or f_val > max_val + 1e-6:
            raise ValueError(f"Packet {pkt_id}: field {field}={f_val} outside expected range [{min_val}, {max_val}]")

    # dcri domain check: [-delta * M, 1.0]
    dcri_val = record.get("dcri")
    if dcri_val is None or not math.isfinite(float(dcri_val)):
        raise ValueError(f"Packet {pkt_id}: dcri is not finite ({dcri_val})")
    f_dcri = float(dcri_val)
    if f_dcri < -(delta * 3.0) - 1e-6 or f_dcri > 1.0 + 1e-6:
        raise ValueError(f"Packet {pkt_id}: dcri={f_dcri} outside theoretical range [{-(delta*3.0)}, 1.0]")

    # Weights simplex check
    w_sum = sum(float(weights[m]) for m in MODALITIES)
    if abs(w_sum - 1.0) > 1e-5:
        raise ValueError(f"Packet {pkt_id}: weight sum {w_sum} != 1.0")
    for m, w in weights.items():
        w_f = float(w)
        if w_f < -1e-6 or w_f > 1.0 + 1e-6:
            raise ValueError(f"Packet {pkt_id}: weight {m}={w_f} outside [0, 1]")
        if m not in active_modalities and w_f > 1e-6:
            raise ValueError(f"Packet {pkt_id}: inactive modality {m} has non-zero weight {w_f}")

    # Independently recompute u_sum from active input uncertainties
    input_uncs = record.get("input_uncertainties", {})
    expected_u_sum = sum(float(input_uncs[m]) for m in active_modalities)
    actual_u_sum = float(record["u_sum"])
    if abs(actual_u_sum - expected_u_sum) > 1e-6:
        raise ValueError(f"Packet {pkt_id}: u_sum ({actual_u_sum}) != sum of active uncertainties ({expected_u_sum})")

    # Independently recompute r_fusion from active weighted risks
    input_risks = record.get("input_risks", {})
    expected_r_fusion = sum(float(weights[m]) * float(input_risks[m]) for m in active_modalities)
    actual_r_fusion = float(record["r_fusion"])
    if abs(actual_r_fusion - expected_r_fusion) > 1e-6:
        raise ValueError(f"Packet {pkt_id}: r_fusion ({actual_r_fusion}) != weighted active risks ({expected_r_fusion})")

    # DCRI identity check: DCRI = R_fusion - P_U
    r_f = actual_r_fusion
    p_u = float(record["uncertainty_penalty"])
    dcri = float(record["dcri"])
    if abs(dcri - (r_f - p_u)) > 1e-5:
        raise ValueError(f"Packet {pkt_id}: DCRI ({dcri}) != R_fusion ({r_f}) - P_U ({p_u})")

    # P_U identity check: P_U = delta * u_sum
    if abs(p_u - (delta * actual_u_sum)) > 1e-5:
        raise ValueError(f"Packet {pkt_id}: P_U ({p_u}) != delta * u_sum ({delta * actual_u_sum})")

    # Tier check matches assign_decision_tier
    expected_tier = assign_decision_tier(dcri, tau_1=tau_1, tau_2=tau_2)
    tier = record.get("decision_tier")
    if tier != expected_tier:
        raise ValueError(f"Packet {pkt_id}: decision tier {tier} != expected tier {expected_tier} for dcri={dcri}")

    # Conflict metrics validation: css == delta_max, conflict_category matches CSS threshold
    css = float(record.get("css", 0.0))
    delta_max = float(record.get("delta_max", 0.0))
    if abs(css - delta_max) > 1e-6:
        raise ValueError(f"Packet {pkt_id}: CSS ({css}) != delta_max ({delta_max})")

    cat = record.get("conflict_category")
    if num_active <= 0:
        expected_cat = "NO_MODALITY_AVAILABLE"
    elif num_active == 1:
        expected_cat = "NOT_APPLICABLE"
    elif delta_max < 0.20:
        expected_cat = "LOW"
    elif delta_max < 0.35:
        expected_cat = "MODERATE"
    else:
        expected_cat = "HIGH"

    if cat != expected_cat:
        raise ValueError(f"Packet {pkt_id}: conflict_category '{cat}' != expected '{expected_cat}' for delta_max={delta_max}")


class DecisionEvaluationEngine:
    """
    Core execution engine for Phase C11.17 decision-level evaluation.
    Evaluates packets deterministically under frozen ACARA-U and baseline policies.
    """

    def __init__(self, delta: float = DELTA_FROZEN, tau_1: float = TAU_1_DEFAULT, tau_2: float = TAU_2_DEFAULT):
        if not (0.0 <= delta <= 1.0):
            raise ValueError(f"Invalid delta: {delta}. Must be in [0.0, 1.0]")
        if not (0.0 < tau_1 < tau_2 < 1.0):
            raise ValueError(f"Invalid thresholds: tau_1={tau_1}, tau_2={tau_2}. Must satisfy 0 < tau_1 < tau_2 < 1")

        self.delta = delta
        self.tau_1 = tau_1
        self.tau_2 = tau_2

        # Frozen ACARA-U Router
        self.coefficients = RouterCoefficients(
            alpha=ALPHA_REF,
            beta=BETA_REF,
            gamma=GAMMA_REF,
            eta=ETA_REF,
        )
        self.router = ACARAUv2Router(
            coefficients=self.coefficients,
            router_version="acarau_c11_17_frozen",
        )
        self.dcri_engine = DCRIEngine(router=self.router)
        self.baseline_runner = FusionRunner()
        self.packet_exceptions: int = 0

    def evaluate_packet_b6(
        self,
        packet: ControlledDecisionPacket,
        regime_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates a single packet under full ACARA-U (B6) policy with optional regime override.
        Guarantees strict fail-closed behavior for EMPTY regime.
        """
        try:
            natural_modalities = list(packet.available_modalities)
            is_forced_regime = regime_override is not None
            active_modalities = natural_modalities
            if is_forced_regime:
                active_modalities = list(REGIME_MODALITY_MAP.get(regime_override, ()))

            num_active = len(active_modalities)

            # Fail-closed handling for empty modality set
            if num_active == 0:
                rec = {
                    "packet_id": packet.packet_id,
                    "policy_id": POLICY_B6,
                    "regime": regime_override if regime_override else REGIME_EMPTY,
                    "natural_modalities": natural_modalities,
                    "evaluated_modalities": [],
                    "active_modalities": [],
                    "num_active": 0,
                    "input_risks": {m: float(getattr(packet, m).risk) for m in MODALITIES},
                    "input_uncertainties": {m: float(getattr(packet, m).uncertainty) for m in MODALITIES},
                    "input_qualities": {m: float(getattr(packet, m).quality) for m in MODALITIES},
                    "input_reliabilities": {m: float(getattr(packet, m).reliability) for m in MODALITIES},
                    "input_confidences": {m: float(getattr(packet, m).confidence) for m in MODALITIES},
                    "weights": {"retina": 0.0, "foot": 0.0, "clinical": 0.0},
                    "raw_logits": {"retina": 0.0, "foot": 0.0, "clinical": 0.0},
                    "masked_logits": {"retina": None, "foot": None, "clinical": None},
                    "r_fusion": 0.0,
                    "u_sum": 0.0,
                    "uncertainty_penalty": 0.0,
                    "dcri": 0.0,
                    "routing_entropy": 0.0,
                    "dominant_modality": None,
                    "decision_tier": TIER_0_ROUTINE,
                    "decision_tier_name": TIER_KEYS[TIER_0_ROUTINE],
                    "status": "NO_MODALITY_AVAILABLE",
                    "conflict_metrics": compute_conflict_metrics([], {}, {}, 0.0),
                    "css": 0.0,
                    "delta_max": 0.0,
                    "sigma_w": 0.0,
                    "conflict_category": "NO_MODALITY_AVAILABLE",
                    "is_finite": True,
                    "error": None,
                }
                validate_record(rec, delta=self.delta, tau_1=self.tau_1, tau_2=self.tau_2)
                return rec

            # Build channel inputs
            channels = {}
            risks = {}
            for m in MODALITIES:
                rec = getattr(packet, m)
                is_avail = (m in active_modalities)
                risks[m] = float(rec.risk)
                channels[m] = ModalityChannelInput(
                    modality=m,
                    confidence=float(rec.confidence),
                    reliability=float(rec.reliability),
                    uncertainty=float(rec.uncertainty),
                    quality=float(rec.quality) if is_avail else 0.0,
                    availability=is_avail,
                )

            router_input = RouterInput(
                retina=channels["retina"],
                foot=channels["foot"],
                clinical=channels["clinical"],
            )

            # Dynamic routing
            router_result = self.router.route(router_input)
            weights = dict(router_result.weights)

            # Fused risk = inner product of active weights and risks
            r_fusion = float(sum(weights[m] * risks[m] for m in active_modalities))

            # Canonical unweighted uncertainty penalty: P_U = delta * sum_{i in A} U_i
            dcri, p_u = compute_dcri(
                r_fusion=r_fusion,
                uncertainties={m: float(getattr(packet, m).uncertainty) for m in active_modalities},
                active_modalities=active_modalities,
                delta=self.delta,
            )
            u_sum = float(sum(float(getattr(packet, m).uncertainty) for m in active_modalities))

            # Hypothetical decision tier
            decision_tier = assign_decision_tier(dcri, tau_1=self.tau_1, tau_2=self.tau_2)

            # Conflict metrics
            conflict = compute_conflict_metrics(
                active_modalities=active_modalities,
                risks=risks,
                weights=weights,
                r_fusion=r_fusion,
            )

            # Dominant modality
            dominant_m = max(active_modalities, key=lambda m: weights[m]) if active_modalities else None

            # Finiteness check
            is_finite = (
                math.isfinite(r_fusion)
                and math.isfinite(dcri)
                and math.isfinite(p_u)
                and all(math.isfinite(w) for w in weights.values())
            )

            masked_logits = {}
            for m in MODALITIES:
                raw_val = router_result.masked_logits.get(m)
                if raw_val is not None and math.isfinite(float(raw_val)):
                    masked_logits[m] = float(raw_val)
                else:
                    masked_logits[m] = None

            res = {
                "packet_id": packet.packet_id,
                "policy_id": POLICY_B6,
                "regime": regime_override if regime_override else (REGIME_RFC if num_active == 3 else f"ACT_{num_active}"),
                "natural_modalities": natural_modalities,
                "evaluated_modalities": active_modalities,
                "active_modalities": active_modalities,
                "num_active": num_active,
                "input_risks": {m: float(getattr(packet, m).risk) for m in MODALITIES},
                "input_uncertainties": {m: float(getattr(packet, m).uncertainty) for m in MODALITIES},
                "input_qualities": {m: float(getattr(packet, m).quality) for m in MODALITIES},
                "input_reliabilities": {m: float(getattr(packet, m).reliability) for m in MODALITIES},
                "input_confidences": {m: float(getattr(packet, m).confidence) for m in MODALITIES},
                "weights": {m: float(weights.get(m, 0.0)) for m in MODALITIES},
                "raw_logits": {m: float(router_result.logits.get(m, 0.0)) for m in MODALITIES},
                "masked_logits": masked_logits,
                "r_fusion": float(r_fusion),
                "u_sum": float(u_sum),
                "uncertainty_penalty": float(p_u),
                "dcri": float(dcri),
                "routing_entropy": float(router_result.routing_entropy),
                "dominant_modality": dominant_m,
                "decision_tier": decision_tier,
                "decision_tier_name": TIER_KEYS[decision_tier],
                "status": "SUCCESS",
                "conflict_metrics": conflict,
                "css": float(conflict["css"]),
                "delta_max": float(conflict["delta_max"]),
                "sigma_w": float(conflict["sigma_w"]),
                "conflict_category": conflict["conflict_category"],
                "is_finite": is_finite,
                "error": None,
            }
            validate_record(res, delta=self.delta, tau_1=self.tau_1, tau_2=self.tau_2)
            return res
        except Exception as exc:
            self.packet_exceptions += 1
            return {
                "packet_id": packet.packet_id,
                "policy_id": POLICY_B6,
                "regime": regime_override if regime_override else "ERROR",
                "natural_modalities": list(packet.available_modalities),
                "evaluated_modalities": [],
                "active_modalities": [],
                "num_active": 0,
                "weights": {"retina": 0.0, "foot": 0.0, "clinical": 0.0},
                "raw_logits": {},
                "masked_logits": {},
                "r_fusion": None,
                "u_sum": None,
                "uncertainty_penalty": None,
                "dcri": None,
                "routing_entropy": None,
                "dominant_modality": None,
                "decision_tier": None,
                "decision_tier_name": "error",
                "status": "EXCEPTION",
                "conflict_metrics": {},
                "css": None,
                "delta_max": None,
                "sigma_w": None,
                "conflict_category": "ERROR",
                "is_finite": False,
                "error": str(exc),
            }

    def evaluate_packet_policy(
        self,
        packet: ControlledDecisionPacket,
        policy_id: str,
        regime_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates a single packet under any policy B1 through B6.
        For B6, uses evaluate_packet_b6.
        For B1–B5, uses baseline FusionRunner with canonical DCRI penalty reconciliation.
        """
        if policy_id == POLICY_B6:
            return self.evaluate_packet_b6(packet, regime_override=regime_override)

        # Baseline execution
        try:
            # Mask packet if regime_override is provided
            eval_packet = packet
            if regime_override is not None:
                target_mods = set(REGIME_MODALITY_MAP.get(regime_override, ()))
                # If target is EMPTY, fail closed immediately
                if len(target_mods) == 0:
                    rec = {
                        "packet_id": packet.packet_id,
                        "policy_id": policy_id,
                        "regime": regime_override,
                        "natural_modalities": list(packet.available_modalities),
                        "evaluated_modalities": [],
                        "active_modalities": [],
                        "num_active": 0,
                        "input_risks": {m: float(getattr(packet, m).risk) for m in MODALITIES},
                        "input_uncertainties": {m: float(getattr(packet, m).uncertainty) for m in MODALITIES},
                        "input_qualities": {m: float(getattr(packet, m).quality) for m in MODALITIES},
                        "input_reliabilities": {m: float(getattr(packet, m).reliability) for m in MODALITIES},
                        "input_confidences": {m: float(getattr(packet, m).confidence) for m in MODALITIES},
                        "weights": {"retina": 0.0, "foot": 0.0, "clinical": 0.0},
                        "raw_logits": {},
                        "masked_logits": {"retina": None, "foot": None, "clinical": None},
                        "r_fusion": 0.0,
                        "u_sum": 0.0,
                        "uncertainty_penalty": 0.0,
                        "dcri": 0.0,
                        "routing_entropy": 0.0,
                        "dominant_modality": None,
                        "decision_tier": TIER_0_ROUTINE,
                        "decision_tier_name": TIER_KEYS[TIER_0_ROUTINE],
                        "status": "NO_MODALITY_AVAILABLE",
                        "conflict_metrics": compute_conflict_metrics([], {}, {}, 0.0),
                        "css": 0.0,
                        "delta_max": 0.0,
                        "sigma_w": 0.0,
                        "conflict_category": "NO_MODALITY_AVAILABLE",
                        "is_finite": True,
                        "error": None,
                    }
                    validate_record(rec, delta=self.delta, tau_1=self.tau_1, tau_2=self.tau_2)
                    return rec

                # Construct masked packet
                r_avail = ("retina" in target_mods)
                f_avail = ("foot" in target_mods)
                c_avail = ("clinical" in target_mods)

                r_rec = ModalityRecord(
                    sample_id=packet.retina.sample_id,
                    modality="retina",
                    risk=packet.retina.risk,
                    calibrated_probability=packet.retina.calibrated_probability,
                    confidence=packet.retina.confidence,
                    uncertainty=packet.retina.uncertainty,
                    quality=packet.retina.quality if r_avail else 0.0,
                    availability=r_avail,
                    reliability=packet.retina.reliability,
                    model_version=packet.retina.model_version,
                )
                f_rec = ModalityRecord(
                    sample_id=packet.foot.sample_id,
                    modality="foot",
                    risk=packet.foot.risk,
                    calibrated_probability=packet.foot.calibrated_probability,
                    confidence=packet.foot.confidence,
                    uncertainty=packet.foot.uncertainty,
                    quality=packet.foot.quality if f_avail else 0.0,
                    availability=f_avail,
                    reliability=packet.foot.reliability,
                    model_version=packet.foot.model_version,
                )
                c_rec = ModalityRecord(
                    sample_id=packet.clinical.sample_id,
                    modality="clinical",
                    risk=packet.clinical.risk,
                    calibrated_probability=packet.clinical.calibrated_probability,
                    confidence=packet.clinical.confidence,
                    uncertainty=packet.clinical.uncertainty,
                    quality=packet.clinical.quality if c_avail else 0.0,
                    availability=c_avail,
                    reliability=packet.clinical.reliability,
                    model_version=packet.clinical.model_version,
                )
                eval_packet = ControlledDecisionPacket(
                    packet_id=packet.packet_id,
                    retina=r_rec,
                    foot=f_rec,
                    clinical=c_rec,
                    seed=packet.seed,
                    packet_type=packet.packet_type,
                )

            fusion_res: FusionResult = self.baseline_runner.evaluate_packet(eval_packet, policy_id)
            active_modalities = list(eval_packet.available_modalities)
            weights = dict(fusion_res.weights)
            r_fusion = float(fusion_res.r_fusion)

            # Compute canonical DCRI
            dcri, p_u = compute_dcri(
                r_fusion=r_fusion,
                uncertainties={m: float(getattr(eval_packet, m).uncertainty) for m in active_modalities},
                active_modalities=active_modalities,
                delta=self.delta,
            )
            u_sum = float(sum(float(getattr(eval_packet, m).uncertainty) for m in active_modalities))

            decision_tier = assign_decision_tier(dcri, tau_1=self.tau_1, tau_2=self.tau_2)

            risks = {m: float(getattr(eval_packet, m).risk) for m in MODALITIES}
            conflict = compute_conflict_metrics(
                active_modalities=active_modalities,
                risks=risks,
                weights=weights,
                r_fusion=r_fusion,
            )

            is_finite = (
                math.isfinite(r_fusion)
                and math.isfinite(dcri)
                and math.isfinite(p_u)
                and all(math.isfinite(w) for w in weights.values())
            )

            res = {
                "packet_id": packet.packet_id,
                "policy_id": policy_id,
                "regime": regime_override if regime_override else (REGIME_RFC if len(active_modalities) == 3 else f"ACT_{len(active_modalities)}"),
                "natural_modalities": list(packet.available_modalities),
                "evaluated_modalities": active_modalities,
                "active_modalities": active_modalities,
                "num_active": len(active_modalities),
                "input_risks": {m: float(getattr(eval_packet, m).risk) for m in MODALITIES},
                "input_uncertainties": {m: float(getattr(eval_packet, m).uncertainty) for m in MODALITIES},
                "input_qualities": {m: float(getattr(eval_packet, m).quality) for m in MODALITIES},
                "input_reliabilities": {m: float(getattr(eval_packet, m).reliability) for m in MODALITIES},
                "input_confidences": {m: float(getattr(eval_packet, m).confidence) for m in MODALITIES},
                "weights": {m: float(weights.get(m, 0.0)) for m in MODALITIES},
                "raw_logits": {},
                "masked_logits": {m: None for m in MODALITIES},
                "r_fusion": float(r_fusion),
                "u_sum": float(u_sum),
                "uncertainty_penalty": float(p_u),
                "dcri": float(dcri),
                "routing_entropy": float(fusion_res.routing_entropy),
                "dominant_modality": fusion_res.dominant_modality,
                "decision_tier": decision_tier,
                "decision_tier_name": TIER_KEYS[decision_tier],
                "status": "SUCCESS",
                "conflict_metrics": conflict,
                "css": float(conflict["css"]),
                "delta_max": float(conflict["delta_max"]),
                "sigma_w": float(conflict["sigma_w"]),
                "conflict_category": conflict["conflict_category"],
                "is_finite": is_finite,
                "error": None,
            }
            validate_record(res, delta=self.delta, tau_1=self.tau_1, tau_2=self.tau_2)
            return res
        except Exception as exc:
            self.packet_exceptions += 1
            return {
                "packet_id": packet.packet_id,
                "policy_id": policy_id,
                "regime": regime_override if regime_override else "ERROR",
                "natural_modalities": list(packet.available_modalities),
                "evaluated_modalities": [],
                "active_modalities": [],
                "num_active": 0,
                "weights": {"retina": 0.0, "foot": 0.0, "clinical": 0.0},
                "raw_logits": {},
                "masked_logits": {m: None for m in MODALITIES},
                "r_fusion": None,
                "u_sum": None,
                "uncertainty_penalty": None,
                "dcri": None,
                "routing_entropy": None,
                "dominant_modality": None,
                "decision_tier": None,
                "decision_tier_name": "error",
                "status": "EXCEPTION",
                "conflict_metrics": {},
                "css": None,
                "delta_max": None,
                "sigma_w": None,
                "conflict_category": "ERROR",
                "is_finite": False,
                "error": str(exc),
            }

    def evaluate_cohort(
        self,
        cohort: Sequence[ControlledDecisionPacket],
        policy_id: str,
        regime_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluates an entire cohort of packets under a designated policy and optional regime."""
        records = [
            self.evaluate_packet_policy(p, policy_id=policy_id, regime_override=regime_override)
            for p in cohort
        ]
        # Validate cohort execution integrity
        exceptions = [r for r in records if r.get("status") == "EXCEPTION"]
        if exceptions:
            first_err = exceptions[0].get("error", "Unknown error")
            raise RuntimeError(
                f"Cohort evaluation under policy {policy_id} produced {len(exceptions)} exception records! First error: {first_err}"
            )
        return records
