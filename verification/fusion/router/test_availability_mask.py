"""
Unit tests for ACARA-U hard availability masking and 7 modality configurations (Phase C11.4).
"""

import pytest
from src.fusion.router.router_input import ModalityChannelInput, RouterInput, FROZEN_RELIABILITY_MAP
from src.fusion.router.acarau_router import ACARAUv2Router


def create_input(a_r: bool, a_f: bool, a_c: bool) -> RouterInput:
    """Helper to create RouterInput with specified availability flags."""
    return RouterInput(
        retina=ModalityChannelInput(
            "retina", 0.85, FROZEN_RELIABILITY_MAP["retina"], 0.10, 0.90 if a_r else 0.0, a_r
        ),
        foot=ModalityChannelInput(
            "foot", 0.80, FROZEN_RELIABILITY_MAP["foot"], 0.15, 0.85 if a_f else 0.0, a_f
        ),
        clinical=ModalityChannelInput(
            "clinical", 0.75, FROZEN_RELIABILITY_MAP["clinical"], 0.20, 0.80 if a_c else 0.0, a_c
        ),
    )


def test_zero_modality_graceful_rejection():
    """Config 0: None available => Safe rejection / NO_MODALITY_AVAILABLE."""
    inp = create_input(False, False, False)
    router = ACARAUv2Router()
    res = router.route(inp)

    assert res.status == "NO_MODALITY_AVAILABLE"
    assert res.num_active == 0
    assert res.active_modalities == []
    assert res.weights == {"retina": 0.0, "foot": 0.0, "clinical": 0.0}
    assert res.normalization_sum == 0.0
    assert res.routing_entropy == 0.0
    assert res.dominant_modality is None


@pytest.mark.parametrize(
    "a_r,a_f,a_c,expected_mod",
    [
        (True, False, False, "retina"),
        (False, True, False, "foot"),
        (False, False, True, "clinical"),
    ],
)
def test_unimodal_configurations(a_r, a_f, a_c, expected_mod):
    """Configs 5, 6, 7: Unimodal configurations assign exactly 1.0 weight to the sole available modality."""
    inp = create_input(a_r, a_f, a_c)
    router = ACARAUv2Router()
    res = router.route(inp)

    assert res.status == "SUCCESS"
    assert res.num_active == 1
    assert res.active_modalities == [expected_mod]
    assert res.weights[expected_mod] == 1.0
    for mod in ["retina", "foot", "clinical"]:
        if mod != expected_mod:
            assert res.weights[mod] == 0.0
    assert res.dominant_modality == expected_mod
    assert res.routing_entropy == 0.0  # -1 * ln(1) = 0


@pytest.mark.parametrize(
    "a_r,a_f,a_c,active_pair,missing_mod",
    [
        (True, True, False, ["retina", "foot"], "clinical"),
        (True, False, True, ["retina", "clinical"], "foot"),
        (False, True, True, ["foot", "clinical"], "retina"),
    ],
)
def test_bimodal_configurations(a_r, a_f, a_c, active_pair, missing_mod):
    """Configs 2, 3, 4: Bimodal configurations assign weights summing to 1.0 across active pair and 0.0 to missing."""
    inp = create_input(a_r, a_f, a_c)
    router = ACARAUv2Router()
    res = router.route(inp)

    assert res.status == "SUCCESS"
    assert res.num_active == 2
    assert set(res.active_modalities) == set(active_pair)
    assert res.weights[missing_mod] == 0.0
    assert abs(res.weights[active_pair[0]] + res.weights[active_pair[1]] - 1.0) < 1e-7
    assert res.weights[active_pair[0]] > 0.0
    assert res.weights[active_pair[1]] > 0.0


def test_trimodal_configuration():
    """Config 1: Tri-modal full examination assigns strictly positive weights summing to 1.0."""
    inp = create_input(True, True, True)
    router = ACARAUv2Router()
    res = router.route(inp)

    assert res.status == "SUCCESS"
    assert res.num_active == 3
    assert len(res.active_modalities) == 3
    for mod, w in res.weights.items():
        assert w > 0.0
    assert abs(res.normalization_sum - 1.0) < 1e-7
