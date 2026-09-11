# Phase 10.1.J — Provenance & License Audit Report

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
| **Unique Source Image Groups** | 1,770 source images ($5.68\times$ offline expansion) |

> [!IMPORTANT]
> **Provenance Lineage Distinction**:
> 1. **Kaggle Uploader $\neq$ Original Dataset Creator**: The Kaggle repository is a community mirror derived from the Roboflow Universe export created by user `adpm`.
> 2. **Kaggle / Roboflow License $\neq$ Native Clinical Provenance**: The declared **MIT License** governs the software export artifact from Roboflow. The underlying primary image photography originates from clinical DFU photography collections.

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
- **Annotation Assessment**: Images are annotated into four distinct Wagner categories. Clinical raters are reported in dataset descriptors, but individual clinician metadata and inter-rater agreement statistics ($\kappa$) are not included in the raw export files.

---

## 4. Citation Requirements

To cite this dataset in research publications:

```bibtex
@misc{adpm_dfu_2025,
  title={ADPM V3.3 Classification Dataset},
  author={adpm},
  year={2025},
  publisher={Roboflow Universe},
  url={https://universe.roboflow.com/adpm/adpm-v3.3-classification}
}

@article{wagner1981dysvascular,
  title={The dysvascular foot: a system for diagnosis and treatment},
  author={Wagner, F William},
  journal={Foot \& Ankle},
  volume={2},
  number={2},
  pages={64--122},
  year={1981}
}
```
