"""
FusionMedAI - Phase C11.1: Unified Modality Output Contract Verification Gate
Automated verification script executing all 8 contract validation and integration tests.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def run_c11_1_verification():
    print("=" * 60)
    print("FusionMedAI: C11.1 Unified Modality Output Contract")
    print("=" * 60)

    passed_checks = 0
    total_checks = 8

    # -------------------------------------------------------------------------
    # [1/8] Schema validation
    # -------------------------------------------------------------------------
    print("\n[1/8] Testing schema validation...")
    try:
        sample = ModalityOutput(
            risk=0.4,
            calibrated_probability=[0.1, 0.2, 0.4, 0.2, 0.1],
            confidence=0.75,
            uncertainty=0.12,
            quality=0.95,
            availability=True,
            reliability=0.885,
            model_version="test_v1.0",
        )
        assert hasattr(sample, "risk")
        assert hasattr(sample, "calibrated_probability")
        assert hasattr(sample, "confidence")
        assert hasattr(sample, "uncertainty")
        assert hasattr(sample, "quality")
        assert hasattr(sample, "availability")
        assert hasattr(sample, "reliability")
        assert hasattr(sample, "model_version")
        print("  -> Schema validation: PASS")
        passed_checks += 1
    except Exception as e:
        print(f"  -> Schema validation: FAIL ({e})")

    # -------------------------------------------------------------------------
    # [2/8] Range invariants
    # -------------------------------------------------------------------------
    print("\n[2/8] Testing scalar range invariants [0.0, 1.0]...")
    range_tests_passed = True
    test_cases = [
        ("risk", -0.1), ("risk", 1.05),
        ("confidence", -0.01), ("confidence", 1.2),
        ("uncertainty", -0.5), ("uncertainty", 1.1),
        ("quality", -0.1), ("quality", 2.0),
        ("reliability", -0.1), ("reliability", 1.5),
    ]
    for field_name, bad_val in test_cases:
        kwargs = {
            "risk": 0.5,
            "calibrated_probability": [0.5, 0.5],
            "confidence": 0.5,
            "uncertainty": 0.1,
            "quality": 1.0,
            "availability": True,
            "reliability": 0.8,
            "model_version": "v1",
        }
        kwargs[field_name] = bad_val
        try:
            ModalityOutput(**kwargs)
            range_tests_passed = False
            print(f"  -> ERROR: Failed to reject {field_name}={bad_val}")
        except ModalityContractValidationError:
            pass

    if range_tests_passed:
        print("  -> Range invariants: PASS")
        passed_checks += 1
    else:
        print("  -> Range invariants: FAIL")

    # -------------------------------------------------------------------------
    # [3/8] Probability normalization
    # -------------------------------------------------------------------------
    print("\n[3/8] Testing probability vector normalization...")
    prob_ok = True
    try:
        # Invalid sum
        ModalityOutput(
            risk=0.5,
            calibrated_probability=[0.3, 0.3],  # sum = 0.6
            confidence=0.5,
            uncertainty=0.1,
            quality=1.0,
            availability=True,
            reliability=0.8,
            model_version="v1",
        )
        prob_ok = False
    except ModalityContractValidationError:
        pass

    try:
        # Empty when available
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
        prob_ok = False
    except ModalityContractValidationError:
        pass

    if prob_ok:
        print("  -> Probability normalization: PASS")
        passed_checks += 1
    else:
        print("  -> Probability normalization: FAIL")

    # -------------------------------------------------------------------------
    # [4/8] Risk projections
    # -------------------------------------------------------------------------
    print("\n[4/8] Testing deterministic risk projections...")
    try:
        # Retina
        assert project_retina_risk([1, 0, 0, 0, 0]) == 0.0
        assert project_retina_risk([0, 0, 0, 0, 1]) == 1.0
        assert project_retina_risk([0, 0, 1, 0, 0]) == 0.5
        assert abs(project_retina_risk([0.2, 0.2, 0.2, 0.2, 0.2]) - 0.5) < 1e-6

        # Foot
        assert project_foot_risk([1, 0, 0, 0]) == 0.0
        assert project_foot_risk([0, 0, 0, 1]) == 1.0
        assert abs(project_foot_risk([0, 1, 0, 0]) - 1.0/3.0) < 1e-6
        assert abs(project_foot_risk([0, 0, 1, 0]) - 2.0/3.0) < 1e-6

        # Clinical
        assert abs(project_clinical_risk([0.8, 0.2]) - 0.2) < 1e-6
        assert abs(project_clinical_risk([0.1, 0.9]) - 0.9) < 1e-6
        assert abs(project_clinical_risk([0.45]) - 0.45) < 1e-6

        print("  -> Risk projections: PASS")
        passed_checks += 1
    except Exception as e:
        print(f"  -> Risk projections: FAIL ({e})")

    # -------------------------------------------------------------------------
    # [5/8] Availability handling
    # -------------------------------------------------------------------------
    print("\n[5/8] Testing availability handling...")
    avail_ok = True
    try:
        unavail = ModalityOutput.unavailable(model_version="retina_v1", reliability=0.885)
        assert unavail.availability is False
        assert unavail.risk == 0.0
        assert unavail.calibrated_probability == []
        assert unavail.reliability == 0.885

        # Attempt invalid unavail with non-zero risk
        try:
            ModalityOutput(
                risk=0.5,
                calibrated_probability=[],
                confidence=0.0,
                uncertainty=1.0,
                quality=0.0,
                availability=False,
                reliability=0.885,
                model_version="v1",
            )
            avail_ok = False
        except ModalityContractValidationError:
            pass

    except Exception as e:
        avail_ok = False
        print(f"  -> Availability handling error: {e}")

    if avail_ok:
        print("  -> Availability handling: PASS")
        passed_checks += 1
    else:
        print("  -> Availability handling: FAIL")

    # -------------------------------------------------------------------------
    # [6/8] Immutability
    # -------------------------------------------------------------------------
    print("\n[6/8] Testing contract immutability...")
    try:
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
        try:
            out.risk = 0.9
            print("  -> Immutability: FAIL (attribute modification succeeded)")
        except (FrozenInstanceError, TypeError, AttributeError):
            print("  -> Immutability: PASS")
            passed_checks += 1
    except Exception as e:
        print(f"  -> Immutability: FAIL ({e})")

    # -------------------------------------------------------------------------
    # [7/8] Retina/Foot/Clinical adapters
    # -------------------------------------------------------------------------
    print("\n[7/8] Testing Retina, Foot, and Clinical adapters...")
    try:
        # Retina Adapter
        r_adapter = RetinaAdapter(frozen_reliability=0.885)
        r_out = r_adapter.adapt_prediction_dict({
            "calib_probabilities": [0.6, 0.2, 0.1, 0.05, 0.05],
            "mc_predictive_variance": 0.005,
            "calib_margin": 0.4,
        }, quality=0.96)
        assert isinstance(r_out, ModalityOutput)
        assert r_out.availability is True
        assert r_out.reliability == 0.885

        r_none = r_adapter.adapt_prediction_dict(None)
        assert r_none.availability is False
        assert r_none.risk == 0.0

        # Foot Adapter
        f_adapter = FootAdapter(frozen_reliability=0.850)
        f_out = f_adapter.adapt_prediction_dict({
            "calibrated_probabilities": [0.7, 0.2, 0.1, 0.0],
            "mc_predictive_entropy_norm": 0.18,
        }, quality=0.91)
        assert isinstance(f_out, ModalityOutput)
        assert f_out.availability is True
        assert abs(f_out.confidence - 0.7) < 1e-6

        f_none = f_adapter.adapt_prediction_dict(None)
        assert f_none.availability is False
        assert f_none.risk == 0.0

        # Clinical Adapter
        c_adapter = ClinicalAdapter(frozen_reliability=0.820)
        c_out = c_adapter.adapt_prediction_output({
            "calibrated_probability": 0.18,
            "confidence": 0.64,
            "uncertainty": {"std_probability": 0.02},
            "shift_detection": {"missingness_ratio": 0.08},
        })
        assert isinstance(c_out, ModalityOutput)
        assert c_out.availability is True
        assert c_out.risk == 0.18
        assert abs(c_out.quality - 0.92) < 1e-6

        c_none = c_adapter.adapt_prediction_output(None)
        assert c_none.availability is False
        assert c_none.risk == 0.0

        print("  -> Retina/Foot/Clinical adapters: PASS")
        passed_checks += 1
    except Exception as e:
        print(f"  -> Retina/Foot/Clinical adapters: FAIL ({e})")

    # -------------------------------------------------------------------------
    # [8/8] Frozen-module integration
    # -------------------------------------------------------------------------
    print("\n[8/8] Testing frozen-module integration...")
    try:
        # Test simulated and active pipelines
        retina_payload = r_adapter.adapt_prediction_dict({
            "calib_probabilities": [0.8, 0.1, 0.05, 0.03, 0.02],
            "mc_predictive_variance": 0.002,
            "calib_margin": 0.7,
        })
        foot_payload = f_adapter.adapt_prediction_dict({
            "calibrated_probabilities": [0.05, 0.85, 0.08, 0.02],
            "mc_predictive_entropy_norm": 0.12,
        })
        clinical_payload = c_adapter.adapt_prediction_output({
            "calibrated_probability": 0.35,
            "confidence": 0.70,
            "uncertainty": {"std_probability": 0.04},
        })

        assert isinstance(retina_payload, ModalityOutput)
        assert isinstance(foot_payload, ModalityOutput)
        assert isinstance(clinical_payload, ModalityOutput)

        assert retina_payload.risk == project_retina_risk(retina_payload.calibrated_probability)
        assert foot_payload.risk == project_foot_risk(foot_payload.calibrated_probability)
        assert clinical_payload.risk == project_clinical_risk(clinical_payload.calibrated_probability)

        print("  -> Frozen-module integration: PASS")
        passed_checks += 1
    except Exception as e:
        print(f"  -> Frozen-module integration: FAIL ({e})")

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"C11.1 CONTRACT VERIFICATION: {passed_checks}/{total_checks} PASSED")
    print("=" * 60)

    return passed_checks == total_checks


if __name__ == "__main__":
    import pytest
    success = run_c11_1_verification()
    sys.exit(0 if success else 1)
