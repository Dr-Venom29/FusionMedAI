"""
Unit tests for RouterInput and ModalityChannelInput contracts (Phase C11.4).
"""

import pytest
from dataclasses import FrozenInstanceError

from src.fusion.router.router_input import (
    ModalityChannelInput,
    RouterInput,
    RouterContractValidationError,
    FROZEN_RELIABILITY_MAP,
)
from src.fusion.contracts.modality_output import ModalityOutput


def test_modality_channel_input_valid():
    """Tests valid creation of ModalityChannelInput."""
    ch = ModalityChannelInput(
        modality="retina",
        confidence=0.85,
        reliability=FROZEN_RELIABILITY_MAP["retina"],
        uncertainty=0.15,
        quality=0.90,
        availability=True,
    )
    assert ch.modality == "retina"
    assert ch.confidence == 0.85
    assert ch.reliability == FROZEN_RELIABILITY_MAP["retina"]
    assert ch.uncertainty == 0.15
    assert ch.quality == 0.90
    assert ch.availability is True


def test_modality_channel_input_immutability():
    """Tests that ModalityChannelInput is immutable."""
    ch = ModalityChannelInput(
        modality="foot",
        confidence=0.80,
        reliability=FROZEN_RELIABILITY_MAP["foot"],
        uncertainty=0.20,
        quality=0.85,
        availability=True,
    )
    with pytest.raises(FrozenInstanceError):
        ch.confidence = 0.90


def test_modality_channel_invalid_name():
    """Tests that invalid modality name raises RouterContractValidationError."""
    with pytest.raises(RouterContractValidationError, match="Invalid modality"):
        ModalityChannelInput(
            modality="cardio",
            confidence=0.5,
            reliability=0.5,
            uncertainty=0.5,
            quality=0.5,
            availability=True,
        )


def test_modality_channel_out_of_bounds_scalars():
    """Tests that scalar values outside [0.0, 1.0] are rejected."""
    # Confidence > 1.0
    with pytest.raises(RouterContractValidationError, match="confidence"):
        ModalityChannelInput(
            modality="retina",
            confidence=1.05,
            reliability=FROZEN_RELIABILITY_MAP["retina"],
            uncertainty=0.1,
            quality=0.9,
            availability=True,
        )
    # Uncertainty < 0.0
    with pytest.raises(RouterContractValidationError, match="uncertainty"):
        ModalityChannelInput(
            modality="retina",
            confidence=0.8,
            reliability=FROZEN_RELIABILITY_MAP["retina"],
            uncertainty=-0.05,
            quality=0.9,
            availability=True,
        )


def test_hard_availability_quality_invariant():
    """Tests that A_i=0 with Q_i>0 raises RouterContractValidationError."""
    with pytest.raises(RouterContractValidationError, match="Hard invariant A_i=0 => Q_i=0.0 violated"):
        ModalityChannelInput(
            modality="clinical",
            confidence=0.7,
            reliability=FROZEN_RELIABILITY_MAP["clinical"],
            uncertainty=0.3,
            quality=0.8,  # Invalid when availability is False
            availability=False,
        )


def test_frozen_reliability_protection():
    """Tests that tampering with frozen reliability raises RouterContractValidationError."""
    with pytest.raises(RouterContractValidationError, match="does not match Phase C11.3 frozen prior"):
        ModalityChannelInput(
            modality="retina",
            confidence=0.9,
            reliability=0.999999,  # Tampered
            uncertainty=0.1,
            quality=0.9,
            availability=True,
        )


def test_router_input_canonical_construction():
    """Tests valid tri-modal RouterInput creation."""
    r_ch = ModalityChannelInput("retina", 0.9, FROZEN_RELIABILITY_MAP["retina"], 0.1, 0.9, True)
    f_ch = ModalityChannelInput("foot", 0.8, FROZEN_RELIABILITY_MAP["foot"], 0.2, 0.8, True)
    c_ch = ModalityChannelInput("clinical", 0.7, FROZEN_RELIABILITY_MAP["clinical"], 0.3, 0.7, True)

    r_in = RouterInput(retina=r_ch, foot=f_ch, clinical=c_ch)
    assert r_in.num_available == 3
    assert r_in.available_modalities == ["retina", "foot", "clinical"]


def test_router_input_from_modality_outputs():
    """Tests RouterInput.from_outputs correctly converts C11.1 ModalityOutput contracts."""
    out_r = ModalityOutput(0.25, [0.7, 0.2, 0.05, 0.03, 0.02], 0.5, 0.1, 0.95, True, FROZEN_RELIABILITY_MAP["retina"], "v1")
    out_f = ModalityOutput(0.33, [0.6, 0.3, 0.08, 0.02], 0.6, 0.15, 0.90, True, FROZEN_RELIABILITY_MAP["foot"], "v1")
    out_c = ModalityOutput.unavailable("v1", FROZEN_RELIABILITY_MAP["clinical"])

    r_in = RouterInput.from_outputs(out_r, out_f, out_c)
    assert r_in.num_available == 2
    assert r_in.available_modalities == ["retina", "foot"]
    assert r_in.clinical.quality == 0.0
    assert r_in.clinical.availability is False
