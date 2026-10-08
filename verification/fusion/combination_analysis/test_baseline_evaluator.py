"""
verification/fusion/combination_analysis/test_baseline_evaluator.py
Phase C11.9: Unit tests for comparative baseline evaluation across modality combinations.
"""

import pytest
import numpy as np

from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord
from src.fusion.combination_analysis.distribution_generator import (
    generate_stratified_distribution_cohort,
    DISTRIBUTION_D2_MODERATE_HEAD_TAIL,
)
from src.fusion.combination_analysis.baseline_evaluator import (
    evaluate_baselines_on_combinations,
    BASELINE_IDS,
)


from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)


def make_test_cohort(n: int = 20) -> list[ControlledDecisionPacket]:
    cohort = []
    for i in range(n):
        r = ModalityRecord(
            sample_id=f"s_{i}", modality="retina", risk=0.25, calibrated_probability=(0.75, 0.25),
            confidence=0.8, uncertainty=0.1, quality=0.9, availability=True,
            reliability=FROZEN_RETINA_RELIABILITY, model_version="v1"
        )
        f = ModalityRecord(
            sample_id=f"s_{i}", modality="foot", risk=0.50, calibrated_probability=(0.50, 0.50),
            confidence=0.7, uncertainty=0.2, quality=0.8, availability=True,
            reliability=FROZEN_FOOT_RELIABILITY, model_version="v1"
        )
        c = ModalityRecord(
            sample_id=f"s_{i}", modality="clinical", risk=0.10, calibrated_probability=(0.90, 0.10),
            confidence=0.9, uncertainty=0.05, quality=0.95, availability=True,
            reliability=FROZEN_CLINICAL_RELIABILITY, model_version="v1"
        )
        cohort.append(ControlledDecisionPacket(packet_id=f"pkt_{i:02d}", retina=r, foot=f, clinical=c, seed=115))
    return cohort


from src.fusion.baselines.fusion_runner import FusionRunner
from src.fusion.combination_analysis.combination_definition import COMBINATION_RFC, apply_combination_to_packet


def test_baseline_evaluator_runs_all_baselines():
    cohort = make_test_cohort(20)
    assigned = generate_stratified_distribution_cohort(cohort, DISTRIBUTION_D2_MODERATE_HEAD_TAIL, seed=115)

    head = ["RFC", "RF"]
    tail = ["R", "F", "C"]

    eval_res = evaluate_baselines_on_combinations(assigned, cohort, head, tail)
    summary = eval_res["summary_table"]

    assert set(summary.keys()) == set(BASELINE_IDS)
    for b_id in BASELINE_IDS:
        assert summary[b_id]["head_mean_r"] >= 0.0
        assert summary[b_id]["tail_mean_r"] >= 0.0


def test_baseline_mathematical_properties():
    runner = FusionRunner()
    cohort = make_test_cohort(1)
    pkt = cohort[0]

    evals = runner.evaluate_all_baselines(pkt)

    # B1: Reliability-selected chooses Retina (highest R=0.929956)
    assert evals["B1"].weights["retina"] == 1.0
    assert evals["B1"].weights["foot"] == 0.0
    assert evals["B1"].weights["clinical"] == 0.0

    # B2: Uniform average produces exactly 1/3 for all 3 active channels
    assert np.isclose(evals["B2"].weights["retina"], 1.0 / 3.0, atol=1e-5)
    assert np.isclose(evals["B2"].weights["foot"], 1.0 / 3.0, atol=1e-5)
    assert np.isclose(evals["B2"].weights["clinical"], 1.0 / 3.0, atol=1e-5)

    # B3: Confidence weighting assigns highest weight to Clinical (conf=0.9)
    assert evals["B3"].weights["clinical"] > evals["B3"].weights["retina"] > evals["B3"].weights["foot"]

    # B4: Incorporating reliability shifts weight back towards Retina and Foot
    assert evals["B4"].weights["retina"] > evals["B3"].weights["retina"]

    # B5: Penalizing uncertainty boosts Clinical (uncertainty=0.05) over Foot (0.2)
    assert evals["B5"].weights["clinical"] > evals["B4"].weights["clinical"]

    # B6: Full ACARA-U incorporates image/tabular quality
    assert np.isclose(sum(evals["B6"].weights.values()), 1.0, atol=1e-5)
