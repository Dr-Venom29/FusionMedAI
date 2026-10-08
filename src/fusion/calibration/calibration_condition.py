"""
src/fusion/calibration/calibration_condition.py
Phase C11.11: Modality Calibration Conditions & Probability Transformations

Defines:
1. The 6 experimental conditions (B0–B5).
2. Frozen calibration parameters across constituent modalities:
   - Retina: Temperature Scaling (T = 1.6218)
   - Foot: Vector Scaling (W = [1.041, 1.043, 0.8711, 1.1301], b = [0.0292, 0.0625, 0.0630, -0.1547])
   - Clinical: Platt / Logit Scaling parameters selected in C7 workflow (a = 0.983834, b = -0.002721)
3. Forward and inverse transformations between raw and calibrated distributions.
4. Risk projection recomputations:
   - Retina: r_R = sum (k/4) * p_k
   - Foot: r_F = sum (k/3) * p_k
   - Clinical: r_C = p_1
5. Dual confidence propagation modes:
   - Primary: Risk-only transformation preserving frozen confidence contracts
   - Secondary: Full confidence propagation where C_i = max(p_i)
"""

from typing import Dict, Any, List, Tuple, Sequence, Optional, Union
from dataclasses import dataclass
import numpy as np

from src.fusion.contracts.modality_output import (
    project_retina_risk,
    project_foot_risk,
    project_clinical_risk,
)
from src.fusion.baselines.decision_packet import (
    ModalityRecord,
    ControlledDecisionPacket,
    DecisionPacketError,
)

# Experimental Condition Identifiers
COND_B0_UNCAL_UNIFORM = "B0_uncalibrated_uniform"
COND_B1_UNCAL_RELIABILITY = "B1_uncalibrated_reliability"
COND_B2_UNCAL_ACARAU = "B2_uncalibrated_acarau"
COND_B3_CAL_UNIFORM = "B3_calibrated_uniform"
COND_B4_CAL_RELIABILITY = "B4_calibrated_reliability"
COND_B5_CAL_ACARAU = "B5_calibrated_acarau"

CALIBRATION_CONDITIONS: Tuple[str, ...] = (
    COND_B0_UNCAL_UNIFORM,
    COND_B1_UNCAL_RELIABILITY,
    COND_B2_UNCAL_ACARAU,
    COND_B3_CAL_UNIFORM,
    COND_B4_CAL_RELIABILITY,
    COND_B5_CAL_ACARAU,
)

# Frozen Modality Calibration Parameters
FROZEN_RETINA_TEMPERATURE: float = 1.6218

FROZEN_FOOT_VECTOR_WEIGHTS: Tuple[float, ...] = (1.0410, 1.0430, 0.8711, 1.1301)
FROZEN_FOOT_VECTOR_BIAS: Tuple[float, ...] = (0.0292, 0.0625, 0.0630, -0.1547)

# Clinical Platt Scaling parameters from C7 benchmark (validation fit: slope=0.983834, intercept=-0.002721)
# Provides smooth, exact parametric invertibility between logit margins and calibrated probabilities.
CLINICAL_PLATT_SLOPE: float = 0.9838337648553361
CLINICAL_PLATT_INTERCEPT: float = -0.0027214330440547435



@dataclass(frozen=True)
class ModalityCalibrationState:
    """Represents a modality's raw vs calibrated probability and risk state."""
    modality: str
    p_raw: Tuple[float, ...]
    p_cal: Tuple[float, ...]
    risk_raw: float
    risk_cal: float
    confidence_raw: float
    confidence_cal: float
    delta_risk: float
    delta_confidence: float


def uncalibrate_retina_probability(
    cal_prob: Sequence[float],
    temperature: float = FROZEN_RETINA_TEMPERATURE,
) -> Tuple[float, ...]:
    """
    Inverts Temperature Scaling for 5-class Retinopathy probabilities:
        z_k = T * log(p_k_cal)
        p_raw = softmax(z) = (p_cal ^ T) / sum(p_cal ^ T)
    """
    p = np.clip(np.asarray(cal_prob, dtype=np.float64), 1e-12, 1.0)
    # p_raw = p^T / sum(p^T)
    p_pow = np.power(p, temperature)
    sum_pow = np.sum(p_pow)
    if sum_pow <= 0:
        p_raw = np.ones_like(p) / len(p)
    else:
        p_raw = p_pow / sum_pow
    return tuple(float(x) for x in p_raw)


def calibrate_retina_probability(
    raw_prob: Sequence[float],
    temperature: float = FROZEN_RETINA_TEMPERATURE,
) -> Tuple[float, ...]:
    """
    Applies Temperature Scaling for 5-class Retinopathy probabilities:
        p_cal = softmax(log(p_raw) / T) = (p_raw ^ (1/T)) / sum(p_raw ^ (1/T))
    """
    p = np.clip(np.asarray(raw_prob, dtype=np.float64), 1e-12, 1.0)
    p_pow = np.power(p, 1.0 / temperature)
    sum_pow = np.sum(p_pow)
    if sum_pow <= 0:
        p_cal = np.ones_like(p) / len(p)
    else:
        p_cal = p_pow / sum_pow
    return tuple(float(x) for x in p_cal)


def uncalibrate_foot_probability(
    cal_prob: Sequence[float],
    weights: Sequence[float] = FROZEN_FOOT_VECTOR_WEIGHTS,
    bias: Sequence[float] = FROZEN_FOOT_VECTOR_BIAS,
) -> Tuple[float, ...]:
    """
    Inverts Vector Scaling for 4-class Wagner Diabetic Foot Ulcer probabilities:
        z_cal = log(p_cal)
        z_raw = (z_cal - b) / W
        p_raw = softmax(z_raw)
    """
    p = np.clip(np.asarray(cal_prob, dtype=np.float64), 1e-12, 1.0)
    w = np.asarray(weights, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    z_cal = np.log(p)
    z_raw = (z_cal - b) / w
    # Stable softmax
    z_raw_shifted = z_raw - np.max(z_raw)
    exp_z = np.exp(z_raw_shifted)
    p_raw = exp_z / np.sum(exp_z)
    return tuple(float(x) for x in p_raw)


def calibrate_foot_probability(
    raw_prob: Sequence[float],
    weights: Sequence[float] = FROZEN_FOOT_VECTOR_WEIGHTS,
    bias: Sequence[float] = FROZEN_FOOT_VECTOR_BIAS,
) -> Tuple[float, ...]:
    """
    Applies Vector Scaling for 4-class Wagner Diabetic Foot Ulcer probabilities:
        z_raw = log(p_raw)
        z_cal = W * z_raw + b
        p_cal = softmax(z_cal)
    """
    p = np.clip(np.asarray(raw_prob, dtype=np.float64), 1e-12, 1.0)
    w = np.asarray(weights, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    z_raw = np.log(p)
    z_cal = w * z_raw + b
    z_cal_shifted = z_cal - np.max(z_cal)
    exp_z = np.exp(z_cal_shifted)
    p_cal = exp_z / np.sum(exp_z)
    return tuple(float(x) for x in p_cal)


def uncalibrate_clinical_probability(
    cal_prob: Sequence[float],
    slope: float = CLINICAL_PLATT_SLOPE,
    intercept: float = CLINICAL_PLATT_INTERCEPT,
) -> Tuple[float, ...]:
    """
    Inverts Calibration for binary clinical readmission probabilities:
        z_cal = log(p1_cal / (1 - p1_cal))
        z_raw = (z_cal - intercept) / slope
        p1_raw = 1 / (1 + exp(-z_raw))
    """
    p1_cal = cal_prob[1] if len(cal_prob) == 2 else cal_prob[0]
    p1_clipped = np.clip(p1_cal, 1e-12, 1.0 - 1e-12)
    z_cal = np.log(p1_clipped / (1.0 - p1_clipped))
    z_raw = (z_cal - intercept) / max(slope, 1e-6)
    p1_raw = float(1.0 / (1.0 + np.exp(-z_raw)))
    p1_raw = float(np.clip(p1_raw, 0.0, 1.0))
    p0_raw = float(1.0 - p1_raw)
    return (p0_raw, p1_raw)


def calibrate_clinical_probability(
    raw_prob: Sequence[float],
    slope: float = CLINICAL_PLATT_SLOPE,
    intercept: float = CLINICAL_PLATT_INTERCEPT,
) -> Tuple[float, ...]:
    """
    Applies Calibration for binary clinical readmission probabilities:
        z_raw = log(p1_raw / (1 - p1_raw))
        z_cal = slope * z_raw + intercept
        p1_cal = 1 / (1 + exp(-z_cal))
    """
    p1_raw = raw_prob[1] if len(raw_prob) == 2 else raw_prob[0]
    p1_clipped = np.clip(p1_raw, 1e-12, 1.0 - 1e-12)
    z_raw = np.log(p1_clipped / (1.0 - p1_clipped))
    z_cal = slope * z_raw + intercept
    p1_cal = float(1.0 / (1.0 + np.exp(-z_cal)))
    p1_cal = float(np.clip(p1_cal, 0.0, 1.0))
    p0_cal = float(1.0 - p1_cal)
    return (p0_cal, p1_cal)


def extract_modality_calibration_state(
    record: ModalityRecord,
    propagate_confidence: bool = False,
) -> ModalityCalibrationState:
    """
    Extracts paired raw and calibrated states from a ModalityRecord.
    """
    modality = record.modality
    p_cal = tuple(float(x) for x in record.calibrated_probability)
    risk_cal = float(record.risk)

    if modality == "retina":
        p_raw = uncalibrate_retina_probability(p_cal)
        risk_raw = project_retina_risk(p_raw)
        c_raw = float(np.max(p_raw)) if propagate_confidence else float(record.confidence)
        c_cal = float(np.max(p_cal)) if propagate_confidence else float(record.confidence)
    elif modality == "foot":
        p_raw = uncalibrate_foot_probability(p_cal)
        risk_raw = project_foot_risk(p_raw)
        c_raw = float(np.max(p_raw)) if propagate_confidence else float(record.confidence)
        c_cal = float(np.max(p_cal)) if propagate_confidence else float(record.confidence)
    elif modality == "clinical":
        p_raw = uncalibrate_clinical_probability(p_cal)
        risk_raw = project_clinical_risk(p_raw)
        c_raw = float(np.max(p_raw)) if propagate_confidence else float(record.confidence)
        c_cal = float(np.max(p_cal)) if propagate_confidence else float(record.confidence)
    else:
        raise DecisionPacketError(f"Unknown modality: {modality}")

    return ModalityCalibrationState(
        modality=modality,
        p_raw=p_raw,
        p_cal=p_cal,
        risk_raw=risk_raw,
        risk_cal=risk_cal,
        confidence_raw=c_raw,
        confidence_cal=c_cal,
        delta_risk=risk_cal - risk_raw,
        delta_confidence=c_cal - c_raw,
    )


def create_uncalibrated_modality_record(
    record: ModalityRecord,
    propagate_confidence: bool = False,
) -> ModalityRecord:
    """
    Constructs a ModalityRecord with uncalibrated probabilities, recomputed risk,
    and appropriate confidence while preserving quality, uncertainty, reliability, and availability.
    """
    state = extract_modality_calibration_state(record, propagate_confidence=propagate_confidence)
    return ModalityRecord(
        sample_id=record.sample_id,
        modality=record.modality,
        risk=state.risk_raw,
        calibrated_probability=state.p_raw,
        confidence=state.confidence_raw,
        uncertainty=record.uncertainty,
        quality=record.quality,
        availability=record.availability,
        reliability=record.reliability,
        model_version=record.model_version + "_uncalibrated",
    )


def create_calibrated_modality_record(
    record: ModalityRecord,
    propagate_confidence: bool = False,
) -> ModalityRecord:
    """
    Constructs a ModalityRecord with calibrated probabilities, recomputed risk,
    and appropriate confidence while preserving quality, uncertainty, reliability, and availability.
    """
    state = extract_modality_calibration_state(record, propagate_confidence=propagate_confidence)
    return ModalityRecord(
        sample_id=record.sample_id,
        modality=record.modality,
        risk=state.risk_cal,
        calibrated_probability=state.p_cal,
        confidence=state.confidence_cal,
        uncertainty=record.uncertainty,
        quality=record.quality,
        availability=record.availability,
        reliability=record.reliability,
        model_version=record.model_version,
    )


def create_uncalibrated_decision_packet(
    packet: ControlledDecisionPacket,
    propagate_confidence: bool = False,
) -> ControlledDecisionPacket:
    """Transforms all constituent modalities in a decision packet into their uncalibrated counterparts."""
    return ControlledDecisionPacket(
        packet_id=packet.packet_id,
        retina=create_uncalibrated_modality_record(packet.retina, propagate_confidence=propagate_confidence),
        foot=create_uncalibrated_modality_record(packet.foot, propagate_confidence=propagate_confidence),
        clinical=create_uncalibrated_modality_record(packet.clinical, propagate_confidence=propagate_confidence),
        seed=packet.seed,
        packet_type="CONTROLLED_DECISION_PACKET",
    )


def create_calibrated_decision_packet(
    packet: ControlledDecisionPacket,
    propagate_confidence: bool = False,
) -> ControlledDecisionPacket:
    """Transforms all constituent modalities in a decision packet into their calibrated counterparts."""
    return ControlledDecisionPacket(
        packet_id=packet.packet_id,
        retina=create_calibrated_modality_record(packet.retina, propagate_confidence=propagate_confidence),
        foot=create_calibrated_modality_record(packet.foot, propagate_confidence=propagate_confidence),
        clinical=create_calibrated_modality_record(packet.clinical, propagate_confidence=propagate_confidence),
        seed=packet.seed,
        packet_type="CONTROLLED_DECISION_PACKET",
    )
