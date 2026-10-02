"""
FusionMedAI - Phase C11.5: Multimodal Baseline Ladder Package (B1–B6)
Exposes baseline classes, decision packet containers, and comparative evaluation runners.
"""

from src.fusion.baselines.decision_packet import (
    ModalityRecord,
    ControlledDecisionPacket,
    FusionResult,
    DecisionPacketError,
)
from src.fusion.baselines.b1_reliability_selected import ReliabilitySelectedBaseline
from src.fusion.baselines.b2_uniform import UniformAverageBaseline
from src.fusion.baselines.b3_confidence import ConfidenceFusionBaseline
from src.fusion.baselines.b4_confidence_reliability import ConfidenceReliabilityBaseline
from src.fusion.baselines.b5_confidence_reliability_uncertainty import ConfidenceReliabilityUncertaintyBaseline
from src.fusion.baselines.b6_acarau import FullACARAUBaseline
from src.fusion.baselines.fusion_runner import FusionRunner

__all__ = [
    "ModalityRecord",
    "ControlledDecisionPacket",
    "FusionResult",
    "DecisionPacketError",
    "ReliabilitySelectedBaseline",
    "UniformAverageBaseline",
    "ConfidenceFusionBaseline",
    "ConfidenceReliabilityBaseline",
    "ConfidenceReliabilityUncertaintyBaseline",
    "FullACARAUBaseline",
    "FusionRunner",
]
