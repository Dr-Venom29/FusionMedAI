"""
Unit tests for missing modality handling and authority reallocation (Phase C11.4).
"""

import pytest
from src.fusion.router.router_input import ModalityChannelInput, RouterInput, FROZEN_RELIABILITY_MAP
from src.fusion.router.acarau_router import ACARAUv2Router


def test_missing_modality_zero_weight_invariant():
    """
    Tests that an unavailable modality (A_i=0) receives strictly 0.0 weight even if
    its internal confidence is 1.0 and uncertainty is 0.0.
    """
    inp = RouterInput(
        retina=ModalityChannelInput("retina", 1.0, FROZEN_RELIABILITY_MAP["retina"], 0.0, 0.0, False),  # Unavailable!
        foot=ModalityChannelInput("foot", 0.6, FROZEN_RELIABILITY_MAP["foot"], 0.3, 0.7, True),
        clinical=ModalityChannelInput("clinical", 0.5, FROZEN_RELIABILITY_MAP["clinical"], 0.4, 0.6, True),
    )
    router = ACARAUv2Router()
    res = router.route(inp)

    assert res.weights["retina"] == 0.0
    assert res.weights["foot"] > 0.0
    assert res.weights["clinical"] > 0.0
    assert abs(res.weights["foot"] + res.weights["clinical"] - 1.0) < 1e-7


def test_graceful_authority_reallocation():
    """
    Tests step-by-step modality dropout:
    {R, F, C} -> {R, F} -> {R} -> {∅}
    Verifies that authority gracefully reallocates without crashes or negative weights.
    """
    router = ACARAUv2Router()

    # Step 1: Tri-modal
    inp_3 = RouterInput(
        retina=ModalityChannelInput("retina", 0.85, FROZEN_RELIABILITY_MAP["retina"], 0.10, 0.90, True),
        foot=ModalityChannelInput("foot", 0.80, FROZEN_RELIABILITY_MAP["foot"], 0.15, 0.85, True),
        clinical=ModalityChannelInput("clinical", 0.75, FROZEN_RELIABILITY_MAP["clinical"], 0.20, 0.80, True),
    )
    res_3 = router.route(inp_3)
    assert res_3.num_active == 3
    assert abs(res_3.normalization_sum - 1.0) < 1e-7

    # Step 2: Clinical drops out -> {R, F}
    inp_2 = RouterInput(
        retina=ModalityChannelInput("retina", 0.85, FROZEN_RELIABILITY_MAP["retina"], 0.10, 0.90, True),
        foot=ModalityChannelInput("foot", 0.80, FROZEN_RELIABILITY_MAP["foot"], 0.15, 0.85, True),
        clinical=ModalityChannelInput("clinical", 0.75, FROZEN_RELIABILITY_MAP["clinical"], 0.20, 0.0, False),
    )
    res_2 = router.route(inp_2)
    assert res_2.num_active == 2
    assert res_2.weights["clinical"] == 0.0
    assert res_2.weights["retina"] > res_3.weights["retina"]  # absorbed dropped share
    assert res_2.weights["foot"] > res_3.weights["foot"]

    # Step 3: Foot drops out -> {R}
    inp_1 = RouterInput(
        retina=ModalityChannelInput("retina", 0.85, FROZEN_RELIABILITY_MAP["retina"], 0.10, 0.90, True),
        foot=ModalityChannelInput("foot", 0.80, FROZEN_RELIABILITY_MAP["foot"], 0.15, 0.0, False),
        clinical=ModalityChannelInput("clinical", 0.75, FROZEN_RELIABILITY_MAP["clinical"], 0.20, 0.0, False),
    )
    res_1 = router.route(inp_1)
    assert res_1.num_active == 1
    assert res_1.weights["retina"] == 1.0
    assert res_1.weights["foot"] == 0.0
    assert res_1.weights["clinical"] == 0.0

    # Step 4: Retina drops out -> {∅}
    inp_0 = RouterInput(
        retina=ModalityChannelInput("retina", 0.85, FROZEN_RELIABILITY_MAP["retina"], 0.10, 0.0, False),
        foot=ModalityChannelInput("foot", 0.80, FROZEN_RELIABILITY_MAP["foot"], 0.15, 0.0, False),
        clinical=ModalityChannelInput("clinical", 0.75, FROZEN_RELIABILITY_MAP["clinical"], 0.20, 0.0, False),
    )
    res_0 = router.route(inp_0)
    assert res_0.status == "NO_MODALITY_AVAILABLE"
    assert res_0.num_active == 0
