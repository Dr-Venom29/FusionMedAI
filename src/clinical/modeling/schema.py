"""Clinical schema definitions, feature categories, and semantic mappings.

Codified according to the frozen C3 Feature Representation Contract
(datasets/clinical/metadata/eda/reports/feature_representation_contract.json).
"""

from typing import Dict, List, Set

# Identifiers strictly dropped from feature matrix
IDENTIFIERS: List[str] = ["encounter_id", "patient_nbr"]

# Zero-variance attributes dropped from feature matrix
ZERO_VARIANCE_FEATURES: List[str] = ["examide", "citoglipton"]

# Dropped due to extreme missingness (96.88%)
HIGH_MISSINGNESS_DROPS: List[str] = ["weight"]

# Exact expected feature dimension after one-hot and ordinal encoding
EXPECTED_FEATURE_DIM: int = 119

# Target column and mapping definition
TARGET_COLUMN: str = "readmitted"
TARGET_MAPPING: Dict[str, int] = {
    "<30": 1,
    ">30": 0,
    "NO": 0,
}

# 8 Numerical features
NUMERICAL_FEATURES: List[str] = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# Categorical Demographics
CATEGORICAL_DEMOGRAPHICS: List[str] = [
    "race",
    "gender",
    "age",
]

# Categorical Encounter Context & Administration
CATEGORICAL_CONTEXT: List[str] = [
    "admission_type_id",
    "admission_source_id",
    "medical_specialty",
    "payer_code",
]

# Glycemic Monitoring Labs (4-state ordinal or categorical with informative 'None')
GLYCEMIC_LABS: List[str] = [
    "max_glu_serum",
    "A1Cresult",
]

GLYCEMIC_ORDINAL_MAPS: Dict[str, Dict[str, int]] = {
    "max_glu_serum": {
        "None": 0,
        "Norm": 1,
        ">200": 2,
        ">300": 3,
    },
    "A1Cresult": {
        "None": 0,
        "Norm": 1,
        ">7": 2,
        ">8": 3,
    },
}

# Age Band Ordinal Mapping
AGE_ORDINAL_MAP: Dict[str, int] = {
    "[0-10)": 0,
    "[10-20)": 1,
    "[20-30)": 2,
    "[30-40)": 3,
    "[40-50)": 4,
    "[50-60)": 5,
    "[60-70)": 6,
    "[70-80)": 7,
    "[80-90)": 8,
    "[90-100)": 9,
}

# 3 ICD-9 Diagnosis Columns
DIAGNOSIS_COLUMNS: List[str] = [
    "diag_1",
    "diag_2",
    "diag_3",
]

# 21 Active Diabetic Medications (excluding examide, citoglipton)
ACTIVE_MEDICATIONS: List[str] = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# 4-level medication exposure mapping
MEDICATION_EXPOSURE_MAP: Dict[str, int] = {
    "No": 0,
    "Steady": 1,
    "Up": 2,
    "Down": 3,
}

# Treatment Dynamics (Binary Flags)
TREATMENT_DYNAMICS: List[str] = [
    "change",
    "diabetesMed",
]

TREATMENT_DYNAMICS_MAPS: Dict[str, Dict[str, int]] = {
    "change": {
        "No": 0,
        "Ch": 1,
    },
    "diabetesMed": {
        "No": 0,
        "Yes": 1,
    },
}

# Clinical ICD-9 Chapter Categories
ICD9_CHAPTERS: List[str] = [
    "Circulatory",
    "Respiratory",
    "Diabetes",
    "Digestive",
    "Injury_Poisoning",
    "Genitourinary",
    "Musculoskeletal",
    "Neoplasms",
    "Other",
    "Supplementary_External",
    "Missing/Unknown",
]


def map_icd9_to_chapter(code: str) -> str:
    """Map a raw ICD-9 string code into one of the clinical disease chapters.
    
    Mapping hierarchy follows clinical literature (Strack et al., 2014):
    - '?' / empty / NaN -> 'Missing/Unknown'
    - 'V' or 'E' prefix -> 'Supplementary_External'
    - '250' prefix -> 'Diabetes'
    - 390-459, 785 -> 'Circulatory'
    - 460-519, 786 -> 'Respiratory'
    - 520-579, 787 -> 'Digestive'
    - 580-629, 788 -> 'Genitourinary'
    - 800-999 -> 'Injury_Poisoning'
    - 710-739 -> 'Musculoskeletal'
    - 140-239 -> 'Neoplasms'
    - All others -> 'Other'
    """
    if code is None:
        return "Missing/Unknown"
    code_str = str(code).strip()
    if code_str in ["?", "", "nan", "None", "NULL"]:
        return "Missing/Unknown"
    
    if code_str.startswith("V") or code_str.startswith("E"):
        return "Supplementary_External"
    
    if code_str.startswith("250"):
        return "Diabetes"
    
    try:
        val = float(code_str)
    except ValueError:
        return "Other"
    
    if (390 <= val <= 459) or val == 785:
        return "Circulatory"
    if (460 <= val <= 519) or val == 786:
        return "Respiratory"
    if (520 <= val <= 579) or val == 787:
        return "Digestive"
    if (580 <= val <= 629) or val == 788:
        return "Genitourinary"
    if 800 <= val <= 999:
        return "Injury_Poisoning"
    if 710 <= val <= 739:
        return "Musculoskeletal"
    if 140 <= val <= 239:
        return "Neoplasms"
    
    return "Other"
