"""
Unit tests for ACARA-U routing weights computation, normalization, and diagnostics (Phase C11.4).
"""

import math
import pytest
from src.fusion.router.router_input import ModalityChannelInput, RouterInput, FROZEN_RELIABILITY_MAP
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router.acarau_router import ACARAUv2Router


@pytest.fixture
def standard_input():
    return RouterInput(
        retina=ModalityChannelInput("retina", 0.90, FROZEN_RELIABILITY_MAP["retina"], 0.05, 0.95, True),
        foot=ModalityChannelInput("foot", 0.80, FROZEN_RELIABILITY_MAP["foot"], 0.15, 0.85, True),
        clinical=ModalityChannelInput("clinical", 0.70, FROZEN_RELIABILITY_MAP["clinical"], 0.25, 0.75, True),
    )


def test_weight_normalization_and_bounds(standard_input):
    """Tests that weights sum to exactly 1.0 and each weight is in [0.0, 1.0]."""
    router = ACARAUv2Router()
    result = router.route(standard_input)

    assert result.status == "SUCCESS"
    assert result.num_active == 3
    assert abs(result.normalization_sum - 1.0) < 1e-7
    for mod, w in result.weights.items():
        assert 0.0 <= w <= 1.0
        assert math.isfinite(w)


def test_uniform_baseline_weighting(standard_input):
    """Tests that baseline B2 (uniform) produces exactly equal 1/3 weights for tri-modal input."""
    router = ACARAUv2Router(coefficients=RouterCoefficients.uniform_baseline())
    result = router.route(standard_input)

    for mod, w in result.weights.items():
        assert abs(w - (1.0 / 3.0)) < 1e-6
    assert abs(result.routing_entropy - math.log(3.0)) < 1e-6


def test_deterministic_reproducibility(standard_input):
    """Tests that identical inputs yield identical outputs (f(X) == f(X))."""
    router = ACARAUv2Router()
    res1 = router.route(standard_input)
    res2 = router.route(standard_input)

    assert res1.weights == res2.weights
    assert res1.logits == res2.logits
    assert res1.routing_entropy == res2.routing_entropy
    assert res1.dominant_modality == res2.dominant_modality


def test_dominant_modality_selection(standard_input):
    """Tests that dominant modality matches the maximum assigned weight."""
    router = ACARAUv2Router()
    result = router.route(standard_input)

    expected_dom = max(result.weights, key=result.weights.get)
    assert result.dominant_modality == expected_dom


def test_routing_entropy_bounds(standard_input):
    """Tests that routing entropy H(w) satisfies 0 <= H(w) <= ln(3)."""
    router = ACARAUv2Router()
    result = router.route(standard_input)

    assert 0.0 <= result.routing_entropy <= math.log(3.0) + 1e-7
