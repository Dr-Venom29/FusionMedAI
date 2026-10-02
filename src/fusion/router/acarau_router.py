"""
FusionMedAI - Phase C11.4: ACARA-U v2 Dynamic Router Engine
Implements adaptive multimodal routing logit calculation, hard availability masking,
and numerically stable softmax weight allocation.
"""

from typing import Dict, Any, Optional, List, Tuple
import math
import numpy as np

from src.fusion.router.router_input import RouterInput, ModalityChannelInput
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router.router_result import RouterResult
from src.fusion.contracts.modality_output import ModalityOutput


class ACARAUv2Router:
    """
    ACARA-U v2 Dynamic Multimodal Router.
    Computes adaptive modality decision weights:
        z_i = alpha * C_i + beta * R_i - gamma * U_i + eta * Q_i
        z_tilde_i = z_i if A_i == 1 else -inf
        w_i = exp(z_tilde_i - max(z_tilde)) / sum(exp(z_tilde_j - max(z_tilde)))
    """

    ROUTER_VERSION: str = "acarau_v2.0"

    def __init__(
        self,
        coefficients: Optional[RouterCoefficients] = None,
        router_version: str = ROUTER_VERSION,
    ):
        """
        Initializes the router with frozen or configured hyperparameter coefficients.
        
        Args:
            coefficients: Hyperparameters Theta = (alpha, beta, gamma, eta). Defaults to full ACARA-U.
            router_version: Version identifier string.
        """
        self.coefficients = coefficients if coefficients is not None else RouterCoefficients.full_acarau()
        self.router_version = router_version

    def compute_logits(
        self, router_input: RouterInput
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Computes raw pre-masking logits and masked logits for all three modalities.
        
        Returns:
            Tuple of (raw_logits_dict, masked_logits_dict).
        """
        raw_logits: Dict[str, float] = {}
        masked_logits: Dict[str, float] = {}

        alpha = float(self.coefficients.alpha)
        beta = float(self.coefficients.beta)
        gamma = float(self.coefficients.gamma)
        eta = float(self.coefficients.eta)

        for mod_name, ch in router_input.channels.items():
            # Raw logit kernel: z_i = alpha*C_i + beta*R_i - gamma*U_i + eta*Q_i
            z_i = (
                alpha * float(ch.confidence)
                + beta * float(ch.reliability)
                - gamma * float(ch.uncertainty)
                + eta * float(ch.quality)
            )
            raw_logits[mod_name] = float(z_i)

            # Hard availability masking: A_i = 1 => z_i, A_i = 0 => -inf
            if ch.availability:
                masked_logits[mod_name] = float(z_i)
            else:
                masked_logits[mod_name] = float("-inf")

        return raw_logits, masked_logits

    def route(self, router_input: RouterInput) -> RouterResult:
        """
        Executes dynamic routing across the tri-modal input channels.
        
        Args:
            router_input: Canonical RouterInput containing verified channel attributes.
            
        Returns:
            RouterResult: Immutable diagnostic object with normalized weights w_i and metrics.
        """
        active_modalities = router_input.available_modalities
        num_active = len(active_modalities)

        # 1. Zero-Modality Edge Case: Safe Rejection / Graceful Failure
        if num_active == 0:
            return RouterResult.no_modality_available(
                coefficients=self.coefficients.to_dict(),
                router_version=self.router_version,
            )

        # 2. Compute Logits
        raw_logits, masked_logits = self.compute_logits(router_input)

        # 3. Numerically Stable Softmax over Active Modalities Only
        active_raw_scores = [raw_logits[m] for m in active_modalities]
        max_score = max(active_raw_scores)

        # Subtract max for numerical stability (prevents exp overflow)
        exp_scores = {
            m: math.exp(raw_logits[m] - max_score) for m in active_modalities
        }
        sum_exp = sum(exp_scores.values())

        # Construct normalized weights: w_i = exp(z_i - m) / sum(exp(z_j - m))
        weights: Dict[str, float] = {}
        for mod_name in router_input.channels.keys():
            if mod_name in active_modalities:
                w_i = exp_scores[mod_name] / sum_exp
                weights[mod_name] = float(w_i)
            else:
                weights[mod_name] = 0.0

        # Enforce exact floating point sum = 1.0 normalization across active
        norm_sum = sum(weights.values())

        # 4. Routing Entropy: H(w) = -sum_{i in A, w_i > 0} w_i * ln(w_i)
        entropy = 0.0
        for m in active_modalities:
            w = weights[m]
            if w > 1e-12:
                entropy -= w * math.log(w)

        # 5. Dominant Modality: argmax_{i in A} w_i
        dominant_modality = max(active_modalities, key=lambda m: weights[m])

        # 6. Construct Immutable RouterResult
        availability_map = {m: ch.availability for m, ch in router_input.channels.items()}

        return RouterResult(
            weights=weights,
            logits=raw_logits,
            masked_logits=masked_logits,
            availability=availability_map,
            coefficients=self.coefficients.to_dict(),
            active_modalities=active_modalities,
            num_active=num_active,
            normalization_sum=norm_sum,
            routing_entropy=entropy,
            dominant_modality=dominant_modality,
            status="SUCCESS",
            router_version=self.router_version,
        )

    def route_from_outputs(
        self,
        retina: ModalityOutput,
        foot: ModalityOutput,
        clinical: ModalityOutput,
    ) -> RouterResult:
        """Convenience method routing directly from three ModalityOutput contract objects."""
        router_input = RouterInput.from_outputs(retina=retina, foot=foot, clinical=clinical)
        return self.route(router_input)
