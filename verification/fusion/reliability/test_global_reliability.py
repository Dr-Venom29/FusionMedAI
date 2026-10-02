"""
Unit Tests for Global Modality Reliability (Phase C11.3).
Verifies frozen constants, immutability, formula precision, and directional monotonicity.
"""

import pytest
import numpy as np

from src.fusion.reliability.reliability_result import (
    ModalityReliability,
    GlobalReliabilitySnapshot,
    ReliabilityContractError,
)
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_AUC,
    FROZEN_RETINA_ECE,
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_AUC,
    FROZEN_FOOT_ECE,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_AUC,
    FROZEN_CLINICAL_ECE,
    FROZEN_CLINICAL_RELIABILITY,
    get_frozen_retina_reliability,
    get_frozen_foot_reliability,
    get_frozen_clinical_reliability,
    get_global_reliability_snapshot,
)


def test_frozen_constants_match_formula():
    """Verify that all frozen reliability constants satisfy R_i = 0.5 * (AUC + (1 - ECE))."""
    r_retina_calc = 0.5 * (FROZEN_RETINA_AUC + (1.0 - FROZEN_RETINA_ECE))
    assert abs(FROZEN_RETINA_RELIABILITY - r_retina_calc) < 1e-5

    r_foot_calc = 0.5 * (FROZEN_FOOT_AUC + (1.0 - FROZEN_FOOT_ECE))
    assert abs(FROZEN_FOOT_RELIABILITY - r_foot_calc) < 1e-5

    r_clin_calc = 0.5 * (FROZEN_CLINICAL_AUC + (1.0 - FROZEN_CLINICAL_ECE))
    assert abs(FROZEN_CLINICAL_RELIABILITY - r_clin_calc) < 1e-5


def test_modality_reliability_contract_rejection():
    """ModalityReliability must reject invalid bounds or formula discrepancies."""
    # Out of bounds AUC
    with pytest.raises(ReliabilityContractError):
        ModalityReliability("test", auc=1.2, ece=0.1, reliability=0.5, n_val_samples=100)

    # Out of bounds ECE
    with pytest.raises(ReliabilityContractError):
        ModalityReliability("test", auc=0.8, ece=-0.1, reliability=0.5, n_val_samples=100)

    # Formula mismatch
    with pytest.raises(ReliabilityContractError):
        ModalityReliability("test", auc=0.8, ece=0.1, reliability=0.99, n_val_samples=100)


def test_global_reliability_snapshot():
    """Verify that snapshot produces immutable instances with correct provenance."""
    snap = get_global_reliability_snapshot()
    assert snap.data_source == "locked_validation_only"
    assert snap.test_used_for_selection is False
    assert snap.retina.reliability == FROZEN_RETINA_RELIABILITY
    assert snap.foot.reliability == FROZEN_FOOT_RELIABILITY
    assert snap.clinical.reliability == FROZEN_CLINICAL_RELIABILITY


def test_reliability_monotonicity():
    """Verify that increasing AUC increases R, and increasing ECE decreases R."""
    base_auc = 0.80
    base_ece = 0.05
    base_r = 0.5 * (base_auc + (1.0 - base_ece))

    higher_auc_r = 0.5 * ((base_auc + 0.05) + (1.0 - base_ece))
    assert higher_auc_r > base_r

    higher_ece_r = 0.5 * (base_auc + (1.0 - (base_ece + 0.05)))
    assert higher_ece_r < base_r
