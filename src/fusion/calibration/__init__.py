"""
src/fusion/calibration/__init__.py
Phase C11.11: Modality Calibration Impact on Decision-Level Fusion
"""

from src.fusion.calibration.calibration_condition import (
    CALIBRATION_CONDITIONS,
    COND_B0_UNCAL_UNIFORM,
    COND_B1_UNCAL_RELIABILITY,
    COND_B2_UNCAL_ACARAU,
    COND_B3_CAL_UNIFORM,
    COND_B4_CAL_RELIABILITY,
    COND_B5_CAL_ACARAU,
    FROZEN_RETINA_TEMPERATURE,
    FROZEN_FOOT_VECTOR_WEIGHTS,
    FROZEN_FOOT_VECTOR_BIAS,
    ModalityCalibrationState,
    uncalibrate_retina_probability,
    uncalibrate_foot_probability,
    uncalibrate_clinical_probability,
    create_calibrated_modality_record,
    create_uncalibrated_modality_record,
    create_calibrated_decision_packet,
    create_uncalibrated_decision_packet,
)

from src.fusion.calibration.calibration_result import (
    ModalityCalibrationProfile,
    PacketCalibrationResult,
    CohortCalibrationSummary,
    PairedCalibrationDelta,
)

from src.fusion.calibration.calibration_metrics import (
    compute_modality_ece,
    compute_modality_brier,
    compute_modality_nll,
    compute_routing_entropy,
    compute_conflict_index,
)

from src.fusion.calibration.calibration_runner import (
    CalibrationExperimentRunner,
)

from src.fusion.calibration.paired_calibration_analysis import (
    compute_paired_bootstrap_cis,
    evaluate_calibration_hypotheses,
)

__all__ = [
    "CALIBRATION_CONDITIONS",
    "COND_B0_UNCAL_UNIFORM",
    "COND_B1_UNCAL_RELIABILITY",
    "COND_B2_UNCAL_ACARAU",
    "COND_B3_CAL_UNIFORM",
    "COND_B4_CAL_RELIABILITY",
    "COND_B5_CAL_ACARAU",
    "FROZEN_RETINA_TEMPERATURE",
    "FROZEN_FOOT_VECTOR_WEIGHTS",
    "FROZEN_FOOT_VECTOR_BIAS",
    "ModalityCalibrationState",
    "uncalibrate_retina_probability",
    "uncalibrate_foot_probability",
    "uncalibrate_clinical_probability",
    "create_calibrated_modality_record",
    "create_uncalibrated_modality_record",
    "create_calibrated_decision_packet",
    "create_uncalibrated_decision_packet",
    "ModalityCalibrationProfile",
    "PacketCalibrationResult",
    "CohortCalibrationSummary",
    "PairedCalibrationDelta",
    "compute_modality_ece",
    "compute_modality_brier",
    "compute_modality_nll",
    "compute_routing_entropy",
    "compute_conflict_index",
    "CalibrationExperimentRunner",
    "compute_paired_bootstrap_cis",
    "evaluate_calibration_hypotheses",
]
