"""
FusionMedAI - Outcome-Grounded Evaluation Subsystem
Provides synthetic oracle cohort generation, paired B6 vs B5 benchmarking,
degradation error modeling, and DCRI decision loss evaluation.
"""

from src.fusion.outcome_evaluation.outcome_packet import OutcomeGroundedPacket, OutcomePacketError
from src.fusion.outcome_evaluation.oracle_generator import OracleCohortGenerator
from src.fusion.outcome_evaluation.outcome_metrics import (
    compute_prediction_metrics,
    compute_hierarchical_bootstrap_ci,
    assign_action_tier,
    compute_decision_loss,
    ACTION_COST_MATRIX,
)
from src.fusion.outcome_evaluation.outcome_runner import OutcomeEvaluationRunner
from src.fusion.outcome_evaluation.experiment_orchestrator import OutcomeExperimentOrchestrator

__all__ = [
    "OutcomeGroundedPacket",
    "OutcomePacketError",
    "OracleCohortGenerator",
    "compute_prediction_metrics",
    "compute_hierarchical_bootstrap_ci",
    "assign_action_tier",
    "compute_decision_loss",
    "ACTION_COST_MATRIX",
    "OutcomeEvaluationRunner",
    "OutcomeExperimentOrchestrator",
]
