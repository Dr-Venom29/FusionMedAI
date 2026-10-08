"""
src/fusion/degradation/degradation_spec.py
Phase C11.10: Input Degradation Specification & Taxonomy

Defines:
1. Degradation Severity Levels: D0 (Clean), D1 (Mild), D2 (Moderate), D3 (Severe).
2. Modality Identifiers: Retina, Foot, Clinical.
3. Modality-Specific Degradation Operators:
   - Retina: D-R1 (Gaussian Blur), D-R2 (Contrast Attenuation), D-R3 (Illumination Shift), D-R4 (Synthetic Artifact)
   - Foot: D-F1 (Gaussian Blur), D-F2 (Contrast Attenuation), D-F3 (Illumination Shift), D-F4 (Synthetic Artifact)
   - Clinical: D-C1 (Random Feature Masking), D-C2 (Structured Feature Masking), D-C3 (Controlled Value Perturbation), D-C4 (Domain Omission)
4. Multi-Modality Degradation Scenarios: Single, Pairwise, All.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Sequence
import numpy as np


# =============================================================================
# 1. Severity Levels & Canonical Constants
# =============================================================================

SEVERITY_D0_CLEAN = "D0"
SEVERITY_D1_MILD = "D1"
SEVERITY_D2_MODERATE = "D2"
SEVERITY_D3_SEVERE = "D3"

SEVERITY_LEVELS: Tuple[str, ...] = (
    SEVERITY_D0_CLEAN,
    SEVERITY_D1_MILD,
    SEVERITY_D2_MODERATE,
    SEVERITY_D3_SEVERE,
)

SEVERITY_DESCRIPTIONS: Dict[str, str] = {
    SEVERITY_D0_CLEAN: "Clean unperturbed baseline input",
    SEVERITY_D1_MILD: "Mild input signal degradation",
    SEVERITY_D2_MODERATE: "Moderate input signal degradation",
    SEVERITY_D3_SEVERE: "Severe input signal degradation",
}

# Modality Identifiers
MODALITY_RETINA = "retina"
MODALITY_FOOT = "foot"
MODALITY_CLINICAL = "clinical"

ALL_MODALITIES: Tuple[str, ...] = (
    MODALITY_RETINA,
    MODALITY_FOOT,
    MODALITY_CLINICAL,
)


# =============================================================================
# 2. Modality Degradation Operators
# =============================================================================

# Retina Operators
OP_RETINA_BLUR = "D-R1_gaussian_blur"
OP_RETINA_CONTRAST = "D-R2_contrast_attenuation"
OP_RETINA_ILLUMINATION = "D-R3_illumination_shift"
OP_RETINA_ARTIFACT = "D-R4_synthetic_artifact"

RETINA_OPERATORS: Tuple[str, ...] = (
    OP_RETINA_BLUR,
    OP_RETINA_CONTRAST,
    OP_RETINA_ILLUMINATION,
    OP_RETINA_ARTIFACT,
)

# Foot Operators
OP_FOOT_BLUR = "D-F1_gaussian_blur"
OP_FOOT_CONTRAST = "D-F2_contrast_attenuation"
OP_FOOT_ILLUMINATION = "D-F3_illumination_shift"
OP_FOOT_ARTIFACT = "D-F4_synthetic_artifact"

FOOT_OPERATORS: Tuple[str, ...] = (
    OP_FOOT_BLUR,
    OP_FOOT_CONTRAST,
    OP_FOOT_ILLUMINATION,
    OP_FOOT_ARTIFACT,
)

# Clinical Operators
OP_CLINICAL_RANDOM_MASK = "D-C1_random_feature_masking"
OP_CLINICAL_STRUCTURED_MASK = "D-C2_structured_feature_masking"
OP_CLINICAL_PERTURBATION = "D-C3_controlled_value_perturbation"
OP_CLINICAL_DOMAIN_OMISSION = "D-C4_domain_omission"

CLINICAL_OPERATORS: Tuple[str, ...] = (
    OP_CLINICAL_RANDOM_MASK,
    OP_CLINICAL_STRUCTURED_MASK,
    OP_CLINICAL_PERTURBATION,
    OP_CLINICAL_DOMAIN_OMISSION,
)

OPERATOR_DESCRIPTIONS: Dict[str, str] = {
    # Retina
    OP_RETINA_BLUR: "Retinal fundus Gaussian blur degradation (attenuates Laplacian sharpness)",
    OP_RETINA_CONTRAST: "Retinal fundus contrast attenuation (compresses dynamic range)",
    OP_RETINA_ILLUMINATION: "Retinal fundus illumination shift (underexposure into dark penalty zone)",
    OP_RETINA_ARTIFACT: "Retinal fundus synthetic opacity smudge artifact / optical occlusion",
    # Foot
    OP_FOOT_BLUR: "Foot ulcer Gaussian blur degradation (attenuates Sobel boundary clarity)",
    OP_FOOT_CONTRAST: "Foot ulcer contrast attenuation (reduces Otsu CNR)",
    OP_FOOT_ILLUMINATION: "Foot ulcer illumination compression (shifts mean intensity and dynamic range)",
    OP_FOOT_ARTIFACT: "Foot ulcer synthetic wound dressing opacity artifact",
    # Clinical
    OP_CLINICAL_RANDOM_MASK: "Random missingness masking of valid features from 119-D contract",
    OP_CLINICAL_STRUCTURED_MASK: "Structured masking of predefined clinical feature groups",
    OP_CLINICAL_PERTURBATION: "Bounded continuous feature corruption and outlier invalidation",
    OP_CLINICAL_DOMAIN_OMISSION: "Complete omission of entire clinical diagnostic domains",
}


# =============================================================================
# 3. Deterministic Parameter Configurations per Operator & Severity
# =============================================================================

OPERATOR_PARAM_GRID: Dict[str, Dict[str, Dict[str, Any]]] = {
    # Retina Operators
    OP_RETINA_BLUR: {
        SEVERITY_D0_CLEAN: {"sigma": 0.0, "kernel_size": 1, "uncertainty_boost": 0.00},
        SEVERITY_D1_MILD: {"sigma": 1.5, "kernel_size": 5, "uncertainty_boost": 0.03},
        SEVERITY_D2_MODERATE: {"sigma": 3.5, "kernel_size": 11, "uncertainty_boost": 0.07},
        SEVERITY_D3_SEVERE: {"sigma": 7.0, "kernel_size": 21, "uncertainty_boost": 0.14},
    },
    OP_RETINA_CONTRAST: {
        SEVERITY_D0_CLEAN: {"factor": 1.00, "uncertainty_boost": 0.00},
        SEVERITY_D1_MILD: {"factor": 0.70, "uncertainty_boost": 0.04},
        SEVERITY_D2_MODERATE: {"factor": 0.45, "uncertainty_boost": 0.08},
        SEVERITY_D3_SEVERE: {"factor": 0.20, "uncertainty_boost": 0.15},
    },
    OP_RETINA_ILLUMINATION: {
        SEVERITY_D0_CLEAN: {"shift": 0, "uncertainty_boost": 0.00},
        SEVERITY_D1_MILD: {"shift": -25, "uncertainty_boost": 0.03},
        SEVERITY_D2_MODERATE: {"shift": -45, "uncertainty_boost": 0.07},
        SEVERITY_D3_SEVERE: {"shift": -60, "uncertainty_boost": 0.13},
    },
    OP_RETINA_ARTIFACT: {
        SEVERITY_D0_CLEAN: {"radius": 0, "uncertainty_boost": 0.00},
        SEVERITY_D1_MILD: {"radius": 45, "uncertainty_boost": 0.04},
        SEVERITY_D2_MODERATE: {"radius": 75, "uncertainty_boost": 0.09},
        SEVERITY_D3_SEVERE: {"radius": 110, "uncertainty_boost": 0.16},
    },
    # Foot Operators
    OP_FOOT_BLUR: {
        SEVERITY_D0_CLEAN: {"sigma": 0.0, "kernel_size": 1, "uncertainty_boost": 0.00},
        SEVERITY_D1_MILD: {"sigma": 1.5, "kernel_size": 5, "uncertainty_boost": 0.03},
        SEVERITY_D2_MODERATE: {"sigma": 3.5, "kernel_size": 11, "uncertainty_boost": 0.08},
        SEVERITY_D3_SEVERE: {"sigma": 9.0, "kernel_size": 25, "uncertainty_boost": 0.15},
    },
    OP_FOOT_CONTRAST: {
        SEVERITY_D0_CLEAN: {"factor": 1.00, "uncertainty_boost": 0.00},
        SEVERITY_D1_MILD: {"factor": 0.70, "uncertainty_boost": 0.04},
        SEVERITY_D2_MODERATE: {"factor": 0.45, "uncertainty_boost": 0.08},
        SEVERITY_D3_SEVERE: {"factor": 0.20, "uncertainty_boost": 0.14},
    },
    OP_FOOT_ILLUMINATION: {
        SEVERITY_D0_CLEAN: {"lum_factor": 1.00, "uncertainty_boost": 0.00},
        SEVERITY_D1_MILD: {"lum_factor": 0.75, "uncertainty_boost": 0.03},
        SEVERITY_D2_MODERATE: {"lum_factor": 0.50, "uncertainty_boost": 0.07},
        SEVERITY_D3_SEVERE: {"lum_factor": 0.25, "uncertainty_boost": 0.13},
    },
    OP_FOOT_ARTIFACT: {
        SEVERITY_D0_CLEAN: {"radius": 0, "uncertainty_boost": 0.00},
        SEVERITY_D1_MILD: {"radius": 40, "uncertainty_boost": 0.04},
        SEVERITY_D2_MODERATE: {"radius": 70, "uncertainty_boost": 0.09},
        SEVERITY_D3_SEVERE: {"radius": 100, "uncertainty_boost": 0.16},
    },
    # Clinical Operators
    OP_CLINICAL_RANDOM_MASK: {
        SEVERITY_D0_CLEAN: {"mask_fraction": 0.00, "uncertainty_boost": 0.00},
        SEVERITY_D1_MILD: {"mask_fraction": 0.15, "uncertainty_boost": 0.02},
        SEVERITY_D2_MODERATE: {"mask_fraction": 0.35, "uncertainty_boost": 0.05},
        SEVERITY_D3_SEVERE: {"mask_fraction": 0.65, "uncertainty_boost": 0.10},
    },
    OP_CLINICAL_STRUCTURED_MASK: {
        SEVERITY_D0_CLEAN: {"num_groups": 0, "uncertainty_boost": 0.00},
        SEVERITY_D1_MILD: {"num_groups": 1, "uncertainty_boost": 0.03},
        SEVERITY_D2_MODERATE: {"num_groups": 2, "uncertainty_boost": 0.06},
        SEVERITY_D3_SEVERE: {"num_groups": 3, "uncertainty_boost": 0.11},
    },
    OP_CLINICAL_PERTURBATION: {
        SEVERITY_D0_CLEAN: {"noise_std": 0.00, "perturb_fraction": 0.00, "uncertainty_boost": 0.00},
        SEVERITY_D1_MILD: {"noise_std": 0.50, "perturb_fraction": 0.15, "uncertainty_boost": 0.03},
        SEVERITY_D2_MODERATE: {"noise_std": 1.50, "perturb_fraction": 0.35, "uncertainty_boost": 0.06},
        SEVERITY_D3_SEVERE: {"noise_std": 3.00, "perturb_fraction": 0.65, "uncertainty_boost": 0.12},
    },

    OP_CLINICAL_DOMAIN_OMISSION: {
        SEVERITY_D0_CLEAN: {"num_domains": 0, "uncertainty_boost": 0.00},
        SEVERITY_D1_MILD: {"num_domains": 1, "uncertainty_boost": 0.03},
        SEVERITY_D2_MODERATE: {"num_domains": 2, "uncertainty_boost": 0.07},
        SEVERITY_D3_SEVERE: {"num_domains": 3, "uncertainty_boost": 0.12},
    },
}


# =============================================================================
# 4. Multi-Modality Degradation Scenarios
# =============================================================================

SCENARIO_SINGLE_RETINA = "R_degraded"
SCENARIO_SINGLE_FOOT = "F_degraded"
SCENARIO_SINGLE_CLINICAL = "C_degraded"

SCENARIO_PAIR_RETINA_FOOT = "RF_degraded"
SCENARIO_PAIR_RETINA_CLINICAL = "RC_degraded"
SCENARIO_PAIR_FOOT_CLINICAL = "FC_degraded"

SCENARIO_ALL_MODALITIES = "RFC_degraded"

ALL_SCENARIOS: Tuple[str, ...] = (
    SCENARIO_SINGLE_RETINA,
    SCENARIO_SINGLE_FOOT,
    SCENARIO_SINGLE_CLINICAL,
    SCENARIO_PAIR_RETINA_FOOT,
    SCENARIO_PAIR_RETINA_CLINICAL,
    SCENARIO_PAIR_FOOT_CLINICAL,
    SCENARIO_ALL_MODALITIES,
)

SCENARIO_DEGRADED_MODALITIES: Dict[str, Tuple[str, ...]] = {
    SCENARIO_SINGLE_RETINA: (MODALITY_RETINA,),
    SCENARIO_SINGLE_FOOT: (MODALITY_FOOT,),
    SCENARIO_SINGLE_CLINICAL: (MODALITY_CLINICAL,),
    SCENARIO_PAIR_RETINA_FOOT: (MODALITY_RETINA, MODALITY_FOOT),
    SCENARIO_PAIR_RETINA_CLINICAL: (MODALITY_RETINA, MODALITY_CLINICAL),
    SCENARIO_PAIR_FOOT_CLINICAL: (MODALITY_FOOT, MODALITY_CLINICAL),
    SCENARIO_ALL_MODALITIES: (MODALITY_RETINA, MODALITY_FOOT, MODALITY_CLINICAL),
}


import hashlib


def deterministic_sample_seed(sample_id: str, base_seed: int = 115) -> int:
    """
    Derives a deterministic, process-independent integer seed in [0, 2^31 - 1]
    using cryptographic SHA-256 hashing of (sample_id, base_seed).
    Guarantees bitwise identical repeatability across independent Python processes
    regardless of PYTHONHASHSEED.
    """
    token = f"{sample_id}_{base_seed}".encode("utf-8")
    digest = hashlib.sha256(token).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def get_operator_params(operator: str, severity: str) -> Dict[str, Any]:
    """Retrieves frozen parameters for a specific operator and severity level."""
    if operator not in OPERATOR_PARAM_GRID:
        raise ValueError(f"Unknown degradation operator: {operator}")
    if severity not in OPERATOR_PARAM_GRID[operator]:
        raise ValueError(f"Unknown severity level '{severity}' for operator {operator}")
    return OPERATOR_PARAM_GRID[operator][severity]

