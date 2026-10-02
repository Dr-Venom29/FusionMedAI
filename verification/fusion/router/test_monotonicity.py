"""
Unit tests for ACARA-U routing monotonicity and hyperparameter isolation (Phase C11.4).
"""

import pytest
from src.fusion.router.router_input import ModalityChannelInput, RouterInput, FROZEN_RELIABILITY_MAP
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router.acarau_router import ACARAUv2Router


def test_confidence_monotonicity():
    """Tests that increasing confidence C_R strictly increases w_R (dw_R/dC_R > 0)."""
    router = ACARAUv2Router(coefficients=RouterCoefficients(alpha=2.0, beta=0.0, gamma=0.0, eta=0.0))

    weights_r = []
    for c_val in [0.2, 0.5, 0.8]:
        inp = RouterInput(
            retina=ModalityChannelInput("retina", c_val, FROZEN_RELIABILITY_MAP["retina"], 0.2, 0.8, True),
            foot=ModalityChannelInput("foot", 0.5, FROZEN_RELIABILITY_MAP["foot"], 0.2, 0.8, True),
            clinical=ModalityChannelInput("clinical", 0.5, FROZEN_RELIABILITY_MAP["clinical"], 0.2, 0.8, True),
        )
        res = router.route(inp)
        weights_r.append(res.weights["retina"])

    assert weights_r[0] < weights_r[1] < weights_r[2]


def test_uncertainty_monotonicity():
    """Tests that increasing uncertainty U_R strictly decreases w_R (dw_R/dU_R < 0)."""
    router = ACARAUv2Router(coefficients=RouterCoefficients(alpha=0.0, beta=0.0, gamma=2.0, eta=0.0))

    weights_r = []
    for u_val in [0.1, 0.5, 0.9]:
        inp = RouterInput(
            retina=ModalityChannelInput("retina", 0.8, FROZEN_RELIABILITY_MAP["retina"], u_val, 0.8, True),
            foot=ModalityChannelInput("foot", 0.8, FROZEN_RELIABILITY_MAP["foot"], 0.5, 0.8, True),
            clinical=ModalityChannelInput("clinical", 0.8, FROZEN_RELIABILITY_MAP["clinical"], 0.5, 0.8, True),
        )
        res = router.route(inp)
        weights_r.append(res.weights["retina"])

    assert weights_r[0] > weights_r[1] > weights_r[2]


def test_quality_monotonicity():
    """Tests that increasing quality Q_R strictly increases w_R (dw_R/dQ_R > 0)."""
    router = ACARAUv2Router(coefficients=RouterCoefficients(alpha=0.0, beta=0.0, gamma=0.0, eta=2.0))

    weights_r = []
    for q_val in [0.2, 0.6, 0.95]:
        inp = RouterInput(
            retina=ModalityChannelInput("retina", 0.8, FROZEN_RELIABILITY_MAP["retina"], 0.2, q_val, True),
            foot=ModalityChannelInput("foot", 0.8, FROZEN_RELIABILITY_MAP["foot"], 0.2, 0.5, True),
            clinical=ModalityChannelInput("clinical", 0.8, FROZEN_RELIABILITY_MAP["clinical"], 0.2, 0.5, True),
        )
        res = router.route(inp)
        weights_r.append(res.weights["retina"])

    assert weights_r[0] < weights_r[1] < weights_r[2]


def test_reliability_monotonicity():
    """
    Tests that under identical instance signals (C, U, Q), modality weights strictly order
    by historical frozen reliability R_R (0.929956) > R_F (0.922266) > R_C (0.825382) when beta > 0.
    """
    router = ACARAUv2Router(coefficients=RouterCoefficients(alpha=0.0, beta=2.0, gamma=0.0, eta=0.0))
    inp = RouterInput(
        retina=ModalityChannelInput("retina", 0.8, FROZEN_RELIABILITY_MAP["retina"], 0.2, 0.8, True),
        foot=ModalityChannelInput("foot", 0.8, FROZEN_RELIABILITY_MAP["foot"], 0.2, 0.8, True),
        clinical=ModalityChannelInput("clinical", 0.8, FROZEN_RELIABILITY_MAP["clinical"], 0.2, 0.8, True),
    )
    res = router.route(inp)

    assert res.weights["retina"] > res.weights["foot"] > res.weights["clinical"]
