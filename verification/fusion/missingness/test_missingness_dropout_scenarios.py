"""
verification/fusion/missingness/test_dropout_scenarios.py
Unit tests for dropout scenarios and ladder generation.
"""

from pathlib import Path
import pytest
import numpy as np

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.missingness.dropout_scenarios import (
    get_regime_packet,
    generate_sequential_ladders,
    get_stress_packet,
    get_random_dropout_packet,
)


@pytest.fixture(scope="module")
def sample_packet():
    cohort = load_frozen_cohort(repo_root=Path("."), n_packets=500, seed=115)
    return cohort[0]


def test_get_regime_packet_all_8(sample_packet):
    for reg_name in ["tri_modal", "retina_foot", "retina_clinical", "foot_clinical",
                     "retina_only", "foot_only", "clinical_only", "zero_modality"]:
        pkt = get_regime_packet(sample_packet, reg_name)
        assert pkt.packet_id.endswith(f"_{reg_name}")


def test_generate_sequential_ladders(sample_packet):
    ladders = generate_sequential_ladders(sample_packet)
    assert "Ladder_RetinaPrimary" in ladders
    assert "Ladder_FootPrimary" in ladders
    assert "Ladder_ClinicalPrimary" in ladders

    retina_primary = ladders["Ladder_RetinaPrimary"]
    assert len(retina_primary) == 3
    assert retina_primary[0][0] == "tri_modal"
    assert retina_primary[1][0] == "retina_foot"
    assert retina_primary[2][0] == "retina_only"


def test_get_stress_packet(sample_packet):
    stress_pkt, dropped_mod, crit = get_stress_packet(sample_packet, "missing_highest_reliability")
    assert dropped_mod == "retina"
    assert "R_retina=" in crit
    assert stress_pkt.available_modalities == ["foot", "clinical"]


def test_get_random_dropout_packet(sample_packet):
    rng = np.random.default_rng(115)
    rnd_pkt, chosen_reg = get_random_dropout_packet(sample_packet, rng)
    assert chosen_reg in ["tri_modal", "retina_foot", "retina_clinical", "foot_clinical",
                          "retina_only", "foot_only", "clinical_only"]
    assert rnd_pkt.num_available > 0
