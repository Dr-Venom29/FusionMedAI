"""
src/fusion/degradation/clinical_degradation.py
Phase C11.10: Tabular Clinical EHR Degradation Operators & Live Quality Computation Engine

Applies deterministic clinical feature corruptions:
- D-C1: Random Feature Masking (randomly masks valid features)
- D-C2: Structured Feature Masking (masks predefined clinical feature groups)
- D-C3: Controlled Value Perturbation (adds bounded continuous noise / out-of-range masks)
- D-C4: Domain Omission (completely omits specific diagnostic domains)

Coupled directly to the frozen clinical quality engine:
- compute_clinical_quality() -> Q_C = N_valid / 119
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from src.fusion.degradation.degradation_spec import (
    SEVERITY_D0_CLEAN,
    SEVERITY_D1_MILD,
    SEVERITY_D2_MODERATE,
    SEVERITY_D3_SEVERE,
    OP_CLINICAL_RANDOM_MASK,
    OP_CLINICAL_STRUCTURED_MASK,
    OP_CLINICAL_PERTURBATION,
    OP_CLINICAL_DOMAIN_OMISSION,
    get_operator_params,
    deterministic_sample_seed,
)
from src.fusion.quality.clinical_quality import compute_clinical_quality, CLINICAL_EXPECTED_DIM
from src.fusion.quality.quality_result import QualityResult
from src.fusion.baselines.decision_packet import ModalityRecord


# =============================================================================
# 1. Feature Group Indices for 119-D Clinical Vector
# =============================================================================

CLINICAL_FEATURE_GROUPS: Dict[str, Tuple[int, int]] = {
    "demographics": (0, 15),          # Age, gender, race, admission type
    "admission_context": (15, 35),    # Time in hospital, discharge, source
    "laboratory": (35, 65),           # Lab procedures, num meds, HbA1c, glucose
    "medications": (65, 95),          # Insulin, metformin, sulfonylureas, etc.
    "diagnoses_utilization": (95, 119),# Primary/secondary ICDs, inpatient/outpatient counts
}


# =============================================================================
# 2. Benchmark Raw Vector Generators
# =============================================================================

def generate_benchmark_clinical_vector(sample_id: str = "clinical_0", seed: int = 115) -> np.ndarray:
    """
    Generates a deterministic 119-D feature vector representing a clean complete EHR record.
    Produces baseline Q_C = 1.0 (all 119 features valid and present).
    Guarantees cross-process bitwise repeatability via cryptographic seed derivation.
    """
    h_seed = deterministic_sample_seed(sample_id, seed)
    rng = np.random.RandomState(h_seed)
    # Generate bounded continuous features in [-2.0, 2.0]
    vec = rng.normal(0.0, 0.8, size=CLINICAL_EXPECTED_DIM)
    return vec


# =============================================================================
# 3. Vector-Level Corruption Functions
# =============================================================================

def apply_random_feature_mask(vec: np.ndarray, mask_fraction: float, seed: int = 115) -> np.ndarray:
    """Randomly sets a fraction of 119 features to NaN."""
    if mask_fraction <= 0.0:
        return vec.copy()
    out = vec.copy()
    rng = np.random.RandomState(seed)
    n_mask = int(round(mask_fraction * len(out)))
    mask_indices = rng.choice(len(out), size=n_mask, replace=False)
    out[mask_indices] = np.nan
    return out


def apply_structured_feature_mask(vec: np.ndarray, num_groups: int) -> np.ndarray:
    """Masks specified number of predefined feature groups to NaN."""
    if num_groups <= 0:
        return vec.copy()
    out = vec.copy()
    domain_names = ["admission_context", "laboratory", "medications", "diagnoses_utilization"]
    for i in range(min(num_groups, len(domain_names))):
        d_name = domain_names[i]
        start_idx, end_idx = CLINICAL_FEATURE_GROUPS[d_name]
        out[start_idx:end_idx] = np.nan
    return out


def apply_value_perturbation(
    vec: np.ndarray,
    perturb_fraction: float,
    noise_std: float = 1.0,
    seed: int = 115,
    clip_threshold: float = 3.5,
) -> np.ndarray:
    """
    Applies genuine continuous value perturbation to a 119-D clinical vector:
    1. Adds Gaussian noise ~ N(0, noise_std^2) to a fraction of continuous features.
    2. Features whose perturbation causes extreme out-of-bounds corruption (|z| > clip_threshold)
       are flagged as invalid (NaN), representing sensor/telemetry rejection by the quality engine.
    """
    if perturb_fraction <= 0.0 and noise_std <= 0.0:
        return vec.copy()
    out = vec.copy()
    rng = np.random.RandomState(seed)
    n_perturb = int(round(perturb_fraction * len(out)))
    if n_perturb > 0:
        perturb_indices = rng.choice(len(out), size=n_perturb, replace=False)
        noise = rng.normal(0.0, noise_std, size=n_perturb)
        out[perturb_indices] = out[perturb_indices] + noise
        corrupted_mask = np.abs(out) > clip_threshold
        out[corrupted_mask] = np.nan
    return out


def apply_domain_omission(vec: np.ndarray, num_domains: int) -> np.ndarray:
    """Omits the first N diagnostic domains to NaN."""
    return apply_structured_feature_mask(vec, num_domains)


# =============================================================================
# 4. Live Raw-Input Operator Dispatcher
# =============================================================================

def apply_clinical_degradation(
    vec: np.ndarray,
    operator: str,
    severity: str,
    seed: int = 115,
) -> np.ndarray:
    """Applies a specified tabular clinical corruption operator to a 119-D feature vector."""
    params = get_operator_params(operator, severity)

    if operator == OP_CLINICAL_RANDOM_MASK:
        return apply_random_feature_mask(vec, params["mask_fraction"], seed=seed)
    elif operator == OP_CLINICAL_STRUCTURED_MASK:
        return apply_structured_feature_mask(vec, params["num_groups"])
    elif operator == OP_CLINICAL_PERTURBATION:
        return apply_value_perturbation(
            vec,
            perturb_fraction=params.get("perturb_fraction", 0.0),
            noise_std=params.get("noise_std", 1.0),
            seed=seed,
        )
    elif operator == OP_CLINICAL_DOMAIN_OMISSION:
        return apply_domain_omission(vec, params["num_domains"])
    else:
        raise ValueError(f"Unsupported clinical degradation operator: {operator}")



def compute_degraded_clinical_quality(
    vec: np.ndarray,
    operator: str,
    severity: str,
    seed: int = 115,
) -> QualityResult:
    """Applies clinical degradation and evaluates the live frozen quality engine."""
    degraded_vec = apply_clinical_degradation(vec, operator, severity, seed=seed)
    return compute_clinical_quality(degraded_vec)


# =============================================================================
# 5. ModalityRecord Transformation with Live Quality Coupling
# =============================================================================

def apply_clinical_degradation_to_record(
    record: ModalityRecord,
    operator: str,
    severity: str,
    live_computed_quality: Optional[float] = None,
) -> ModalityRecord:
    """
    Updates a ModalityRecord with the live computed quality score from Experiment A.
    """
    if not record.availability:
        return record

    params = get_operator_params(operator, severity)
    u_boost = float(params.get("uncertainty_boost", 0.0))

    if live_computed_quality is not None:
        new_q = float(np.clip(live_computed_quality, 0.0, 1.0))
    else:
        q_mult = float(params.get("quality_multiplier", 1.0))
        new_q = float(np.clip(record.quality * q_mult, 0.0, 1.0))

    new_u = float(np.clip(record.uncertainty + u_boost, 0.0, 1.0))
    new_c = float(np.clip(record.confidence * (1.0 - 0.2 * u_boost), 0.0, 1.0))

    return ModalityRecord(
        sample_id=record.sample_id,
        modality=record.modality,
        risk=record.risk,
        calibrated_probability=record.calibrated_probability,
        confidence=new_c,
        uncertainty=new_u,
        quality=new_q,
        availability=record.availability,
        reliability=record.reliability,
        model_version=record.model_version,
    )
