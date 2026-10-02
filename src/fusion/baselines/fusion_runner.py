"""
FusionMedAI - Phase C11.5: Fusion Runner & Comparative Evaluation Engine
Coordinates execution of Baselines B1–B6 across decision packets and generates comparative statistics.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import math

from src.fusion.baselines.decision_packet import ControlledDecisionPacket, FusionResult
from src.fusion.baselines.b1_reliability_selected import ReliabilitySelectedBaseline
from src.fusion.baselines.b2_uniform import UniformAverageBaseline
from src.fusion.baselines.b3_confidence import ConfidenceFusionBaseline
from src.fusion.baselines.b4_confidence_reliability import ConfidenceReliabilityBaseline
from src.fusion.baselines.b5_confidence_reliability_uncertainty import ConfidenceReliabilityUncertaintyBaseline
from src.fusion.baselines.b6_acarau import FullACARAUBaseline


class FusionRunner:
    """
    Orchestrates comparative benchmarking across Baselines B1 through B6.
    """

    def __init__(self):
        self.baselines = {
            "B1": ReliabilitySelectedBaseline(),
            "B2": UniformAverageBaseline(),
            "B3": ConfidenceFusionBaseline(),
            "B4": ConfidenceReliabilityBaseline(),
            "B5": ConfidenceReliabilityUncertaintyBaseline(),
            "B6": FullACARAUBaseline(),
        }

    def evaluate_packet(
        self, packet: ControlledDecisionPacket, baseline_id: str
    ) -> FusionResult:
        """Evaluates a single baseline on a single decision packet."""
        if baseline_id not in self.baselines:
            raise ValueError(f"Unknown baseline_id '{baseline_id}'. Available: {list(self.baselines.keys())}")
        return self.baselines[baseline_id].evaluate(packet)

    def evaluate_all_baselines(
        self, packet: ControlledDecisionPacket
    ) -> Dict[str, FusionResult]:
        """Evaluates all baselines B1–B6 on a single packet."""
        return {b_id: b.evaluate(packet) for b_id, b in self.baselines.items()}

    def evaluate_cohort(
        self,
        packets: List[ControlledDecisionPacket],
        baseline_id: str,
    ) -> Dict[str, Any]:
        """
        Evaluates a single baseline across a cohort of decision packets and calculates statistical metrics.
        """
        results: List[FusionResult] = [
            self.evaluate_packet(p, baseline_id) for p in packets
        ]

        # Extract weights
        weights_r = [r.weights["retina"] for r in results if r.status == "SUCCESS"]
        weights_f = [r.weights["foot"] for r in results if r.status == "SUCCESS"]
        weights_c = [r.weights["clinical"] for r in results if r.status == "SUCCESS"]

        # Extract entropies and risks
        entropies = [r.routing_entropy for r in results if r.status == "SUCCESS"]
        risks = [r.r_fusion for r in results if r.status == "SUCCESS"]
        disagreements = [r.disagreement["X_max"] for r in results if r.status == "SUCCESS"]

        # Dominant modality frequencies
        dom_counts = {"retina": 0, "foot": 0, "clinical": 0}
        for r in results:
            if r.dominant_modality in dom_counts:
                dom_counts[r.dominant_modality] += 1
        n_success = len([r for r in results if r.status == "SUCCESS"])
        dom_rates = {
            m: round(float(count / max(n_success, 1)), 6)
            for m, count in dom_counts.items()
        }

        return {
            "baseline_id": baseline_id,
            "baseline_name": self.baselines[baseline_id].BASELINE_NAME,
            "num_packets": len(packets),
            "num_success": n_success,
            "weight_statistics": {
                "retina": self._calc_stats(weights_r),
                "foot": self._calc_stats(weights_f),
                "clinical": self._calc_stats(weights_c),
            },
            "routing_entropy_statistics": self._calc_stats(entropies),
            "r_fusion_statistics": self._calc_stats(risks),
            "max_disagreement_statistics": self._calc_stats(disagreements),
            "dominant_modality_rates": dom_rates,
            "individual_results": [r.to_dict() for r in results],
        }

    @staticmethod
    def _calc_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": round(float(np.mean(arr)), 6),
            "median": round(float(np.median(arr)), 6),
            "std": round(float(np.std(arr)), 6),
            "min": round(float(np.min(arr)), 6),
            "max": round(float(np.max(arr)), 6),
        }
