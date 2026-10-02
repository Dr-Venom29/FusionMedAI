"""
Unit tests for ACARA-U numerical stability and extreme boundary conditions (Phase C11.4).
"""

import math
import pytest
from src.fusion.router.router_input import ModalityChannelInput, RouterInput, FROZEN_RELIABILITY_MAP
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router.acarau_router import ACARAUv2Router


def test_extreme_coefficients_stability():
    """Tests router with maximal protocol coefficient bounds (alpha=beta=gamma=eta=5.0)."""
    coeffs = RouterCoefficients(alpha=5.0, beta=5.0, gamma=5.0, eta=5.0)
    router = ACARAUv2Router(coefficients=coeffs)

    # Extreme disparity: Retina perfect (C=1, R=0.93, U=0, Q=1 => z ~ 19.65), Clinical worst (C=0, R=0.825, U=1, Q=0 => z ~ -0.87)
    inp = RouterInput(
        retina=ModalityChannelInput("retina", 1.0, FROZEN_RELIABILITY_MAP["retina"], 0.0, 1.0, True),
        foot=ModalityChannelInput("foot", 0.5, FROZEN_RELIABILITY_MAP["foot"], 0.5, 0.5, True),
        clinical=ModalityChannelInput("clinical", 0.0, FROZEN_RELIABILITY_MAP["clinical"], 1.0, 0.0, True),
    )

    result = router.route(inp)

    assert result.status == "SUCCESS"
    assert abs(result.normalization_sum - 1.0) < 1e-7
    for mod, w in result.weights.items():
        assert math.isfinite(w)
        assert 0.0 <= w <= 1.0
    assert math.isfinite(result.routing_entropy)
    assert result.routing_entropy >= 0.0
    assert result.dominant_modality == "retina"


def test_zero_coefficients_uniformity():
    """Tests router with all zero coefficients (alpha=beta=gamma=eta=0.0)."""
    coeffs = RouterCoefficients(alpha=0.0, beta=0.0, gamma=0.0, eta=0.0)
    router = ACARAUv2Router(coefficients=coeffs)

    inp = RouterInput(
        retina=ModalityChannelInput("retina", 0.99, FROZEN_RELIABILITY_MAP["retina"], 0.01, 0.99, True),
        foot=ModalityChannelInput("foot", 0.10, FROZEN_RELIABILITY_MAP["foot"], 0.90, 0.10, True),
        clinical=ModalityChannelInput("clinical", 0.50, FROZEN_RELIABILITY_MAP["clinical"], 0.50, 0.50, True),
    )

    result = router.route(inp)

    for mod, w in result.weights.items():
        assert abs(w - (1.0 / 3.0)) < 1e-7
    assert abs(result.routing_entropy - math.log(3.0)) < 1e-7
