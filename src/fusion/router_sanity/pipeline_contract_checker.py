"""
FusionMedAI - Phase C11.15: Pipeline Contract Checker
Verifies the architectural and mathematical separation across the three fusion stages:
Stage 1: ACARA-U Dynamic Router (Inputs -> Weights w_i)
Stage 2: Weighted Fused Risk Engine (Weights + Risks -> R_fusion) via src.fusion.dcri.aggregation
Stage 3: DCRI Uncertainty Penalty Engine (R_fusion + Uncertainties -> DCRI_delta) via src.fusion.dcri.uncertainty_penalty

Explicitly documents and verifies that router weight monotonicity is decoupled
from downstream fused risk and DCRI response functions under frozen Theta_0 and delta* = 0.10.
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np

from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.router_input import (
    RouterInput,
    ModalityChannelInput,
    FROZEN_RELIABILITY_MAP,
)
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router_sanity.sanity_config import (
    DELTA_FROZEN,
    ROUTER_COEFFS_REF,
)
from src.fusion.dcri.aggregation import (
    compute_weighted_risk_contributions,
    compute_r_fusion,
)
from src.fusion.dcri.uncertainty_penalty import (
    compute_uncertainty_burden,
    compute_uncertainty_penalty,
)


class PipelineContractChecker:
    """Verifies pipeline stage interfaces, calculations, and property separation."""

    def __init__(
        self,
        coefficients: Optional[RouterCoefficients] = None,
        delta: float = DELTA_FROZEN,
    ):
        self.coefficients = coefficients or RouterCoefficients(
            alpha=ROUTER_COEFFS_REF["alpha"],
            beta=ROUTER_COEFFS_REF["beta"],
            gamma=ROUTER_COEFFS_REF["gamma"],
            eta=ROUTER_COEFFS_REF["eta"],
        )
        self.router = ACARAUv2Router(coefficients=self.coefficients)
        self.delta = float(delta)

    def compute_fused_risk(
        self, weights: Dict[str, float], risks: Dict[str, float], active_modalities: List[str]
    ) -> float:
        """Stage 2: Linear combination of modality risks via production aggregation engine."""
        contributions = compute_weighted_risk_contributions(
            weights=weights,
            risks=risks,
            active_modalities=active_modalities,
        )
        return compute_r_fusion(contributions, active_modalities)

    def compute_dcri(
        self, fused_risk: float, uncertainties: Dict[str, float], active_modalities: List[str]
    ) -> float:
        """Stage 3: DCRI calculation via production uncertainty penalty engine."""
        u_sum, _ = compute_uncertainty_burden(uncertainties, active_modalities)
        penalty = compute_uncertainty_penalty(u_sum, self.delta)
        return fused_risk - penalty

    def verify_pipeline_decoupling(self) -> Dict[str, Any]:
        """
        Verifies that weight monotonicity d w_i / d C_i > 0 does NOT guarantee
        d R_fusion / d C_i > 0, demonstrating proper contract separation.
        Also explicitly asserts downstream DCRI directionality using production components.
        """
        # Base state for Retina-Foot dual regime
        r_ch_base = ModalityChannelInput(
            modality="retina", confidence=0.4, uncertainty=0.1, quality=0.9, availability=True, reliability=FROZEN_RELIABILITY_MAP["retina"]
        )
        f_ch_base = ModalityChannelInput(
            modality="foot", confidence=0.4, uncertainty=0.1, quality=0.9, availability=True, reliability=FROZEN_RELIABILITY_MAP["foot"]
        )
        c_ch_inact = ModalityChannelInput(
            modality="clinical", confidence=0.0, uncertainty=1.0, quality=0.0, availability=False, reliability=FROZEN_RELIABILITY_MAP["clinical"]
        )

        inp_base = RouterInput(retina=r_ch_base, foot=f_ch_base, clinical=c_ch_inact)
        res_base = self.router.route(inp_base)

        # Perturb Retina confidence upwards: 0.4 -> 0.8
        r_ch_high = ModalityChannelInput(
            modality="retina", confidence=0.8, uncertainty=0.1, quality=0.9, availability=True, reliability=FROZEN_RELIABILITY_MAP["retina"]
        )
        inp_high = RouterInput(retina=r_ch_high, foot=f_ch_base, clinical=c_ch_inact)
        res_high = self.router.route(inp_high)

        active_mods = ["retina", "foot"]

        # 1. Verify router weight monotonicity: w_retina increases
        w_r_base = res_base.weights["retina"]
        w_r_high = res_high.weights["retina"]
        weight_increased = w_r_high > w_r_base

        # 2. Scenario A: r_retina (0.9) > r_foot (0.1) => Higher w_retina INCREASES fused risk
        risks_a = {"retina": 0.9, "foot": 0.1, "clinical": 0.0}
        r_fused_base_a = self.compute_fused_risk(res_base.weights, risks_a, active_mods)
        r_fused_high_a = self.compute_fused_risk(res_high.weights, risks_a, active_mods)
        risk_a_increased = r_fused_high_a > r_fused_base_a

        # 3. Scenario B: r_retina (0.1) < r_foot (0.9) => Higher w_retina DECREASES fused risk
        risks_b = {"retina": 0.1, "foot": 0.9, "clinical": 0.0}
        r_fused_base_b = self.compute_fused_risk(res_base.weights, risks_b, active_mods)
        r_fused_high_b = self.compute_fused_risk(res_high.weights, risks_b, active_mods)
        risk_b_decreased = r_fused_high_b < r_fused_base_b

        # 4. DCRI decoupling: DCRI = R_fusion - delta * U_sum
        u_map = {"retina": 0.1, "foot": 0.1, "clinical": 1.0}
        dcri_base_a = self.compute_dcri(r_fused_base_a, u_map, active_mods)
        dcri_high_a = self.compute_dcri(r_fused_high_a, u_map, active_mods)
        dcri_base_b = self.compute_dcri(r_fused_base_b, u_map, active_mods)
        dcri_high_b = self.compute_dcri(r_fused_high_b, u_map, active_mods)

        dcri_a_increased = dcri_high_a > dcri_base_a
        dcri_b_decreased = dcri_high_b < dcri_base_b
        delta_matches_frozen = abs(self.delta - DELTA_FROZEN) < 1e-9

        passed = (
            weight_increased
            and risk_a_increased
            and risk_b_decreased
            and dcri_a_increased
            and dcri_b_decreased
            and delta_matches_frozen
        )

        return {
            "criterion_id": "S15-13",
            "passed": passed,
            "coefficients_used": self.coefficients.to_dict(),
            "delta_used": self.delta,
            "delta_matches_frozen": delta_matches_frozen,
            "weight_delta": w_r_high - w_r_base,
            "scenario_a_risk_delta": r_fused_high_a - r_fused_base_a,
            "scenario_b_risk_delta": r_fused_high_b - r_fused_base_b,
            "scenario_a_dcri_delta": dcri_high_a - dcri_base_a,
            "scenario_b_dcri_delta": dcri_high_b - dcri_base_b,
            "downstream_components_exercised": [
                "src.fusion.dcri.aggregation.compute_weighted_risk_contributions",
                "src.fusion.dcri.aggregation.compute_r_fusion",
                "src.fusion.dcri.uncertainty_penalty.compute_uncertainty_burden",
                "src.fusion.dcri.uncertainty_penalty.compute_uncertainty_penalty",
            ],
            "interpretation": (
                "Router weight monotonicity is strictly decoupled from fused risk directionality, "
                "which is governed by relative individual risk ordering. Downstream fused risk "
                "and DCRI calculations are executed directly through production DCRI modules."
            ),
        }

    def run_all_pipeline_checks(self) -> Dict[str, Any]:
        """Runs the pipeline contract check suite."""
        decoupling_res = self.verify_pipeline_decoupling()
        return {
            "all_pipeline_checks_passed": decoupling_res["passed"],
            "decoupling_verification": decoupling_res,
        }
