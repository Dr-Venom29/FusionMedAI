"""
FusionMedAI - Unit and Invariant Tests for ModalityOutput Contract (Phase C11.1).
"""

import pytest
import numpy as np
from dataclasses import FrozenInstanceError

from src.fusion.contracts.modality_output import (
    ModalityOutput,
    ModalityContractValidationError,
    project_retina_risk,
    project_foot_risk,
    project_clinical_risk,
)
from src.fusion.contracts.retina_adapter import RetinaAdapter
from src.fusion.contracts.foot_adapter import FootAdapter
from src.fusion.contracts.clinical_adapter import ClinicalAdapter


class TestModalityOutputContract:
    """Test suite for the 8-tuple ModalityOutput contract and domain invariants."""

    # -------------------------------------------------------------------------
    # 1. Valid Cases
    # -------------------------------------------------------------------------
    def test_valid_retina_output(self):
        out = ModalityOutput(
            risk=0.5,
            calibrated_probability=[0.1, 0.2, 0.4, 0.2, 0.1],
            confidence=0.8,
            uncertainty=0.15,
            quality=0.95,
            availability=True,
            reliability=0.885,
            model_version="retina_efficientnet_b3_v1.0",
        )
        assert out.risk == 0.5
        assert out.availability is True
        assert len(out.calibrated_probability) == 5
        assert sum(out.calibrated_probability) == pytest.approx(1.0, abs=1e-5)

    def test_valid_foot_output(self):
        out = ModalityOutput(
            risk=0.33333333,
            calibrated_probability=[0.1, 0.7, 0.1, 0.1],
            confidence=0.7,
            uncertainty=0.22,
            quality=0.90,
            availability=True,
            reliability=0.850,
            model_version="foot_efficientnet_b3_v1.0",
        )
        assert out.availability is True
        assert len(out.calibrated_probability) == 4

    def test_valid_clinical_output(self):
        out = ModalityOutput(
            risk=0.25,
            calibrated_probability=[0.75, 0.25],
            confidence=0.5,
            uncertainty=0.10,
            quality=1.0,
            availability=True,
            reliability=0.820,
            model_version="clinical_catboost_hpo_v1.0",
        )
        assert out.risk == 0.25
        assert len(out.calibrated_probability) == 2

    # -------------------------------------------------------------------------
    # 2. Invariant Violations (Invalid Cases)
    # -------------------------------------------------------------------------
    def test_invalid_risk_bounds(self):
        with pytest.raises(ModalityContractValidationError, match="Scalar field 'risk'"):
            ModalityOutput(
                risk=-0.1,
                calibrated_probability=[0.5, 0.5],
                confidence=0.5,
                uncertainty=0.1,
                quality=1.0,
                availability=True,
                reliability=0.8,
                model_version="v1",
            )

        with pytest.raises(ModalityContractValidationError, match="Scalar field 'risk'"):
            ModalityOutput(
                risk=1.05,
                calibrated_probability=[0.5, 0.5],
                confidence=0.5,
                uncertainty=0.1,
                quality=1.0,
                availability=True,
                reliability=0.8,
                model_version="v1",
            )

    def test_invalid_confidence_bounds(self):
        with pytest.raises(ModalityContractValidationError, match="Scalar field 'confidence'"):
            ModalityOutput(
                risk=0.5,
                calibrated_probability=[0.5, 0.5],
                confidence=-0.01,
                uncertainty=0.1,
                quality=1.0,
                availability=True,
                reliability=0.8,
                model_version="v1",
            )

    def test_invalid_uncertainty_bounds(self):
        with pytest.raises(ModalityContractValidationError, match="Scalar field 'uncertainty'"):
            ModalityOutput(
                risk=0.5,
                calibrated_probability=[0.5, 0.5],
                confidence=0.5,
                uncertainty=1.2,
                quality=1.0,
                availability=True,
                reliability=0.8,
                model_version="v1",
            )

    def test_invalid_quality_bounds(self):
        with pytest.raises(ModalityContractValidationError, match="Scalar field 'quality'"):
            ModalityOutput(
                risk=0.5,
                calibrated_probability=[0.5, 0.5],
                confidence=0.5,
                uncertainty=0.1,
                quality=-0.5,
                availability=True,
                reliability=0.8,
                model_version="v1",
            )

    def test_invalid_reliability_bounds(self):
        with pytest.raises(ModalityContractValidationError, match="Scalar field 'reliability'"):
            ModalityOutput(
                risk=0.5,
                calibrated_probability=[0.5, 0.5],
                confidence=0.5,
                uncertainty=0.1,
                quality=1.0,
                availability=True,
                reliability=1.5,
                model_version="v1",
            )

    def test_invalid_probability_sum(self):
        with pytest.raises(ModalityContractValidationError, match="sum must equal 1.0"):
            ModalityOutput(
                risk=0.5,
                calibrated_probability=[0.4, 0.4],  # sum = 0.8
                confidence=0.5,
                uncertainty=0.1,
                quality=1.0,
                availability=True,
                reliability=0.8,
                model_version="v1",
            )

    def test_empty_probability_when_available(self):
        with pytest.raises(ModalityContractValidationError, match="cannot be empty when availability is True"):
            ModalityOutput(
                risk=0.5,
                calibrated_probability=[],
                confidence=0.5,
                uncertainty=0.1,
                quality=1.0,
                availability=True,
                reliability=0.8,
                model_version="v1",
            )

    def test_immutability(self):
        out = ModalityOutput(
            risk=0.3,
            calibrated_probability=[0.7, 0.3],
            confidence=0.6,
            uncertainty=0.1,
            quality=1.0,
            availability=True,
            reliability=0.8,
            model_version="v1",
        )
        with pytest.raises((FrozenInstanceError, TypeError, AttributeError)):
            out.risk = 0.9

    # -------------------------------------------------------------------------
    # 3. Missing / Unavailable Cases
    # -------------------------------------------------------------------------
    def test_unavailable_factory(self):
        out = ModalityOutput.unavailable(model_version="retina_v1", reliability=0.885)
        assert out.availability is False
        assert out.risk == 0.0
        assert out.calibrated_probability == []
        assert out.reliability == 0.885

    def test_invalid_unavailable_non_zero_risk(self):
        with pytest.raises(ModalityContractValidationError, match="When availability is False, risk must be 0.0"):
            ModalityOutput(
                risk=0.5,
                calibrated_probability=[],
                confidence=0.0,
                uncertainty=1.0,
                quality=0.0,
                availability=False,
                reliability=0.8,
                model_version="v1",
            )


class TestDeterministicProjections:
    """Test suite for deterministic scalar risk projection functions."""

    def test_retina_risk_projections(self):
        # Grade 0 (No DR)
        assert project_retina_risk([1.0, 0.0, 0.0, 0.0, 0.0]) == 0.0
        # Grade 4 (PDR)
        assert project_retina_risk([0.0, 0.0, 0.0, 0.0, 1.0]) == 1.0
        # Grade 2 (Moderate)
        assert project_retina_risk([0.0, 0.0, 1.0, 0.0, 0.0]) == 0.5
        # Uniform distribution: 0*0.2 + 0.25*0.2 + 0.5*0.2 + 0.75*0.2 + 1.0*0.2 = 0.5
        assert project_retina_risk([0.2, 0.2, 0.2, 0.2, 0.2]) == pytest.approx(0.5)

    def test_foot_risk_projections(self):
        # Grade 0
        assert project_foot_risk([1.0, 0.0, 0.0, 0.0]) == 0.0
        # Grade 3
        assert project_foot_risk([0.0, 0.0, 0.0, 1.0]) == 1.0
        # Grade 1 (1/3)
        assert project_foot_risk([0.0, 1.0, 0.0, 0.0]) == pytest.approx(1.0 / 3.0)
        # Grade 2 (2/3)
        assert project_foot_risk([0.0, 0.0, 1.0, 0.0]) == pytest.approx(2.0 / 3.0)

    def test_clinical_risk_projections(self):
        assert project_clinical_risk([0.8, 0.2]) == pytest.approx(0.2)
        assert project_clinical_risk([0.1, 0.9]) == pytest.approx(0.9)
        assert project_clinical_risk([0.35]) == pytest.approx(0.35)


class TestAdapters:
    """Test suite for Retina, Foot, and Clinical adapters."""

    def test_retina_adapter_available_and_none(self):
        adapter = RetinaAdapter(frozen_reliability=0.885)
        # 1. Available dict
        sample_dict = {
            "calib_probabilities": [0.7, 0.2, 0.05, 0.03, 0.02],
            "mc_predictive_variance": 0.008,
            "calib_margin": 0.5,
        }
        out = adapter.adapt_prediction_dict(sample_dict, quality=0.98)
        assert isinstance(out, ModalityOutput)
        assert out.availability is True
        assert out.reliability == 0.885
        assert out.quality == 0.98
        assert out.risk == pytest.approx(project_retina_risk([0.7, 0.2, 0.05, 0.03, 0.02]))

        # 2. None input (Unavailable)
        out_none = adapter.adapt_prediction_dict(None)
        assert out_none.availability is False
        assert out_none.risk == 0.0
        assert out_none.calibrated_probability == []

    def test_foot_adapter_available_and_none(self):
        adapter = FootAdapter(frozen_reliability=0.850)
        sample_dict = {
            "calibrated_probabilities": [0.6, 0.3, 0.1, 0.0],
            "mc_predictive_entropy_norm": 0.25,
        }
        out = adapter.adapt_prediction_dict(sample_dict, quality=0.92)
        assert isinstance(out, ModalityOutput)
        assert out.availability is True
        assert out.confidence == pytest.approx(0.6)
        assert out.uncertainty == 0.25

        out_none = adapter.adapt_prediction_dict(None)
        assert out_none.availability is False
        assert out_none.risk == 0.0

    def test_clinical_adapter_available_and_none(self):
        adapter = ClinicalAdapter(frozen_reliability=0.820)
        sample_dict = {
            "calibrated_probability": 0.22,
            "confidence": 0.56,
            "uncertainty": {"std_probability": 0.03},
            "shift_detection": {"missingness_ratio": 0.05},
        }
        out = adapter.adapt_prediction_output(sample_dict)
        assert isinstance(out, ModalityOutput)
        assert out.availability is True
        assert out.risk == pytest.approx(0.22)
        assert out.quality == pytest.approx(0.95)

        out_none = adapter.adapt_prediction_output(None)
        assert out_none.availability is False
        assert out_none.risk == 0.0
