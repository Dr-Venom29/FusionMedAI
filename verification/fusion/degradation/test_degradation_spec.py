"""
verification/fusion/degradation/test_degradation_spec.py
Phase C11.10: Verification of Degradation Specification & Taxonomy
"""

import pytest
from src.fusion.degradation.degradation_spec import (
    SEVERITY_D0_CLEAN,
    SEVERITY_D1_MILD,
    SEVERITY_D2_MODERATE,
    SEVERITY_D3_SEVERE,
    SEVERITY_LEVELS,
    MODALITY_RETINA,
    MODALITY_FOOT,
    MODALITY_CLINICAL,
    ALL_MODALITIES,
    RETINA_OPERATORS,
    FOOT_OPERATORS,
    CLINICAL_OPERATORS,
    OPERATOR_PARAM_GRID,
    get_operator_params,
)


def test_severity_levels_ordering():
    assert len(SEVERITY_LEVELS) == 4
    assert SEVERITY_LEVELS == ("D0", "D1", "D2", "D3")


def test_modalities_definition():
    assert len(ALL_MODALITIES) == 3
    assert set(ALL_MODALITIES) == {"retina", "foot", "clinical"}


def test_operators_count():
    assert len(RETINA_OPERATORS) == 4
    assert len(FOOT_OPERATORS) == 4
    assert len(CLINICAL_OPERATORS) == 4


def test_operator_params_grid_completeness():
    all_ops = RETINA_OPERATORS + FOOT_OPERATORS + CLINICAL_OPERATORS
    assert len(all_ops) == 12

    for op in all_ops:
        assert op in OPERATOR_PARAM_GRID
        for sev in SEVERITY_LEVELS:
            params = get_operator_params(op, sev)
            assert isinstance(params, dict)
            assert "uncertainty_boost" in params
            assert 0.0 <= params["uncertainty_boost"] <= 1.0


def test_clean_severity_preservation():
    all_ops = RETINA_OPERATORS + FOOT_OPERATORS + CLINICAL_OPERATORS
    for op in all_ops:
        clean_p = get_operator_params(op, SEVERITY_D0_CLEAN)
        assert clean_p["uncertainty_boost"] == 0.0
        # Check that clean parameters represent the identity transform
        if "sigma" in clean_p:
            assert clean_p["sigma"] == 0.0
        if "factor" in clean_p:
            assert clean_p["factor"] == 1.0
        if "shift" in clean_p:
            assert clean_p["shift"] == 0
        if "artifact_count" in clean_p:
            assert clean_p["artifact_count"] == 0
        if "mask_fraction" in clean_p:
            assert clean_p["mask_fraction"] == 0.0
        if "mask_groups" in clean_p:
            assert clean_p["mask_groups"] == ()
        if "noise_std" in clean_p:
            assert clean_p["noise_std"] == 0.0
        if "omit_domains" in clean_p:
            assert clean_p["omit_domains"] == ()

