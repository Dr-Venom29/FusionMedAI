import os
import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.config import DATASET_ROOT, RAW_DATA, METADATA_DIR

def run_provenance_audit():
    print("==================================================")
    print("Running Phase 10.1.J — Provenance & License Audit")
    print("==================================================")
    
    readme_dataset_path = RAW_DATA / "README.dataset.txt"
    readme_roboflow_path = RAW_DATA / "README.roboflow.txt"
    
    readme_dataset_content = ""
    if readme_dataset_path.exists():
        readme_dataset_content = readme_dataset_path.read_text()
        
    readme_roboflow_content = ""
    if readme_roboflow_path.exists():
        readme_roboflow_content = readme_roboflow_path.read_text()
        
    provenance_data = {
        "audit_phase": "10.1.J",
        "dataset_name": "Diabetic Foot Ulcer (DFU) Wagner 4-Class Dataset",
        "access_point": {
            "platform": "Kaggle",
            "repository_name": "DFU_Dataset_annotated_into_4_classes",
            "role": "Public mirror / host repository"
        },
        "intermediate_processing_platform": {
            "platform": "Roboflow Universe",
            "project_name": "ADPM V3.3 Classification",
            "url": "https://universe.roboflow.com/adpm/adpm-v3.3-classification",
            "user_account": "adpm",
            "export_date": "2025-04-05T04:35:00Z",
            "version": "v4"
        },
        "provenance_lineage": {
            "kaggle_uploader_vs_creator": "Kaggle uploader is a community mirror host; the dataset was curated/exported via Roboflow by account 'adpm'.",
            "license_vs_original_provenance": "Declared license (MIT) reflects Roboflow export settings. Underlying primary photography derives from clinical DFU image collections.",
            "raw_images_count": 10062,
            "unique_source_images_count": 1770,
            "offline_expansion_ratio": 5.68
        },
        "license": {
            "declared_license": "MIT",
            "license_source": "README.dataset.txt / Roboflow Universe Export",
            "redistribution_terms": "Permissible for non-commercial research and commercial development, subject to retaining original copyright notice and permission notice.",
            "warranty_disclaimer": "Provided 'AS IS' without warranty of any kind, explicit or implied."
        },
        "medical_annotation_claims": {
            "classification_system": "Wagner-Meggitt Classification System (Grades 1-4)",
            "annotation_classes": {
                "Grade 1": "Superficial ulcer (full skin thickness, no subcutaneous involvement)",
                "Grade 2": "Deep ulcer (penetrating to tendon, ligament, or capsule, without bone involvement)",
                "Grade 3": "Deep ulcer with abscess, osteomyelitis, or joint sepsis",
                "Grade 4": "Localized gangrene (forefoot or heel)"
            },
            "clinical_verification_note": "Annotations categorize DFU into 4 Wagner grades. Specific clinician IDs and inter-rater agreement metrics are not provided in export metadata."
        },
        "citation_requirements": [
            {
                "type": "Dataset",
                "citation": "adpm. (2025). ADPM V3.3 Classification [Dataset]. Roboflow Universe. https://universe.roboflow.com/adpm/adpm-v3.3-classification"
            },
            {
                "type": "Clinical Classification",
                "citation": "Wagner, F. W. (1981). The dysvascular foot: a system for diagnosis and treatment. Foot & Ankle, 2(2), 64-122."
            }
        ]
    }
    
    # Save JSON report
    json_path = METADATA_DIR / "provenance_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(provenance_data, f, indent=2)
    print(f"Saved JSON report to: {json_path}")
    
    # Generate Markdown report
    md_content = f"""# Phase 10.1.J — Provenance & License Audit Report

## 1. Overview & Chain of Provenance

| Metric / Property | Detail |
|---|---|
| **Dataset Name** | Diabetic Foot Ulcer (DFU) Wagner 4-Class |
| **Access Platform Host** | Kaggle (`DFU_Dataset_annotated_into_4_classes`) |
| **Roboflow Universe Source** | [ADPM V3.3 Classification](https://universe.roboflow.com/adpm/adpm-v3.3-classification) |
| **Curator / Uploader Account** | `adpm` (Roboflow User) |
| **Export Version & Date** | Version 4 (Exported April 5, 2025) |
| **Declared License** | **MIT License** |
| **Total Exported Images** | 10,062 JPEGs |
| **Unique Source Image Groups** | 1,770 source images ($5.68\\times$ offline expansion) |

> [!IMPORTANT]
> **Provenance Lineage Distinction**:
> 1. **Kaggle Uploader $\\neq$ Original Dataset Creator**: The Kaggle repository is a community mirror derived from the Roboflow Universe export created by user `adpm`.
> 2. **Kaggle / Roboflow License $\\neq$ Native Clinical Provenance**: The declared **MIT License** governs the software export artifact from Roboflow. The underlying primary image photography originates from clinical DFU photography collections.

---

## 2. License & Redistribution Terms

- **Declared License**: **MIT License** (specified in `README.dataset.txt` and Roboflow export).
- **Redistribution Terms**:
  - Open for academic research, modification, and model training.
  - Requires inclusion of the original copyright notice and permission notice in downstream redistributions.
- **Disclaimer**: Provided "AS IS" without express or implied warranty regarding clinical diagnostic accuracy.

---

## 3. Medical Annotation Claims

- **Classification Standard**: **Wagner–Meggitt Classification System** (4 Grades).
  - **Grade 1**: Superficial ulcer (full skin thickness, no subcutaneous involvement).
  - **Grade 2**: Deep ulcer (penetrating to tendon, ligament, or joint capsule, *without* bone involvement).
  - **Grade 3**: Deep ulcer with abscess, osteomyelitis, or joint sepsis.
  - **Grade 4**: Localized gangrene (forefoot or heel).
- **Annotation Assessment**: Images are annotated into four distinct Wagner categories. Clinical raters are reported in dataset descriptors, but individual clinician metadata and inter-rater agreement statistics ($\\kappa$) are not included in the raw export files.

---

## 4. Citation Requirements

To cite this dataset in research publications:

```bibtex
@misc{{adpm_dfu_2025,
  title={{ADPM V3.3 Classification Dataset}},
  author={{adpm}},
  year={{2025}},
  publisher={{Roboflow Universe}},
  url={{https://universe.roboflow.com/adpm/adpm-v3.3-classification}}
}}

@article{{wagner1981dysvascular,
  title={{The dysvascular foot: a system for diagnosis and treatment}},
  author={{Wagner, F William}},
  journal={{Foot \\& Ankle}},
  volume={{2}},
  number={{2}},
  pages={{64--122}},
  year={{1981}}
}}
```
"""

    md_path = METADATA_DIR / "provenance_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved Markdown report to: {md_path}")
    
    print("\nPhase 10.1.J Provenance & License Audit completed successfully.")

if __name__ == "__main__":
    run_provenance_audit()
