"""
verification/fusion/conflict/test_pairwise_disagreement.py
Unit tests for Pairwise Disagreement calculation and PairwiseRecord contracts.
"""

import pytest
import math
from src.fusion.conflict.pairwise_disagreement import (
    compute_pairwise_record,
    compute_all_pairwise_records,
)
from src.fusion.conflict.conflict_result import PairwiseRecord


def test_pairwise_record_calculation():
    rec = compute_pairwise_record(
        modality_a="retina",
        modality_b="foot",
        risk_a=0.80,
        risk_b=0.25,
        weight_a=0.60,
        weight_b=0.40,
        uncertainty_a=0.05,
        uncertainty_b=0.30,
        reliability_a=0.929956,
        reliability_b=0.922266,
    )

    assert rec.modality_a == "retina"
    assert rec.modality_b == "foot"
    assert pytest.approx(rec.signed_difference, abs=1e-6) == 0.55
    assert pytest.approx(rec.absolute_difference, abs=1e-6) == 0.55
    assert rec.higher_risk_modality == "retina"
    assert rec.lower_risk_modality == "foot"
    assert rec.dominant_authority_modality == "retina"
    assert rec.reliability_authority_aligned is True


def test_pairwise_symmetry():
    rec_ab = compute_pairwise_record(
        modality_a="retina",
        modality_b="clinical",
        risk_a=0.70,
        risk_b=0.30,
        weight_a=0.55,
        weight_b=0.45,
        uncertainty_a=0.10,
        uncertainty_b=0.15,
        reliability_a=0.929956,
        reliability_b=0.825382,
    )
    rec_ba = compute_pairwise_record(
        modality_a="clinical",
        modality_b="retina",
        risk_a=0.30,
        risk_b=0.70,
        weight_a=0.45,
        weight_b=0.55,
        uncertainty_a=0.15,
        uncertainty_b=0.10,
        reliability_a=0.825382,
        reliability_b=0.929956,
    )

    assert pytest.approx(rec_ab.absolute_difference, abs=1e-6) == rec_ba.absolute_difference
    assert pytest.approx(rec_ab.signed_difference, abs=1e-6) == -rec_ba.signed_difference
    assert rec_ab.higher_risk_modality == rec_ba.higher_risk_modality == "retina"
    assert rec_ab.lower_risk_modality == rec_ba.lower_risk_modality == "clinical"


def test_zero_disagreement_identity():
    rec = compute_pairwise_record(
        modality_a="foot",
        modality_b="clinical",
        risk_a=0.45,
        risk_b=0.45,
        weight_a=0.50,
        weight_b=0.50,
        uncertainty_a=0.20,
        uncertainty_b=0.20,
        reliability_a=0.922266,
        reliability_b=0.825382,
    )

    assert rec.absolute_difference == 0.0
    assert rec.signed_difference == 0.0
    assert rec.higher_risk_modality == "tied"
    assert rec.lower_risk_modality == "tied"
    assert rec.dominant_authority_modality == "tied"


def test_compute_all_pairwise_records_tri_modal():
    active_mods = ("retina", "foot", "clinical")
    risks = {"retina": 0.80, "foot": 0.20, "clinical": 0.30}
    weights = {"retina": 0.50, "foot": 0.30, "clinical": 0.20}
    uncertainties = {"retina": 0.05, "foot": 0.40, "clinical": 0.10}
    reliabilities = {"retina": 0.929956, "foot": 0.922266, "clinical": 0.825382}

    records = compute_all_pairwise_records(
        active_modalities=active_mods,
        risks=risks,
        weights=weights,
        uncertainties=uncertainties,
        reliabilities=reliabilities,
    )

    assert len(records) == 3
    pairs = [(r.modality_a, r.modality_b) for r in records]
    assert ("retina", "foot") in pairs
    assert ("retina", "clinical") in pairs
    assert ("foot", "clinical") in pairs


def test_compute_all_pairwise_records_single_and_zero():
    risks = {"retina": 0.50}
    weights = {"retina": 1.0}
    uncertainties = {"retina": 0.10}
    reliabilities = {"retina": 0.90}

    assert compute_all_pairwise_records(("retina",), risks, weights, uncertainties, reliabilities) == ()
    assert compute_all_pairwise_records((), {}, {}, {}, {}) == ()
