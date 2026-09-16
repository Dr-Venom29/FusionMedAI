# Volume IX — Foot Ulcer Module Integration

> **Phase 10.9 Research Volume**  
> *Unified Inference Interface & Schema Contract Integration for Diabetic Foot Ulcer Classification*

---

## Executive Summary

Phase 10.9 establishes the unified, self-contained inference interface (**`FootModule`**) for Diabetic Foot Ulcer (DFU) Wagner classification (`src/foot/foot_module.py`). The module integrates the frozen **EfficientNet-B3** primary classifier (`best_model.pt`), frozen **Vector Scaling** probability calibrator ($w^{*}, b^{*}$), stochastic **MC Dropout** uncertainty estimator ($N^{*}=10$ passes, Option B pipeline), and deterministic **Grad-CAM** explainability overlays (`backbone.features.8`).

The interface supports both fast standard inference (`generate_cam=False`) and full explainable inference (`generate_cam=True`), incorporates boundary input validation safety, passes a 12-point automated verification suite (**12/12 PASS**), completes end-to-end acceptance across all 4 Wagner grades, and achieves schema contract parity with the existing `RetinaModule`.

---

## Volume Index & Sitemap

1. [Chapter 01 — Objectives & Frozen Asset Specifications](01_Objectives.md)
2. [Chapter 02 — Module Interface Contract & Input Validation](02_Module_Interface.md)
3. [Chapter 03 — End-to-End Inference Execution Pipeline](03_Inference_Pipeline.md)
4. [Chapter 04 — Unified Output Schema & Contract Specification](04_Output_Schema.md)
5. [Chapter 05 — Integration Verification Protocol & Results](05_Integration_Verification.md)
6. [Chapter 06 — End-to-End Acceptance Results & Retina/Foot Contract Comparison](06_Acceptance.md)

---

## Key Verification & Acceptance Summary

| Verification Layer | Target Component | Protocol / Suite | Result | Status |
| :--- | :--- | :--- | :---: | :---: |
| **Artifact Integrity** | `best_model.pt`, `calibration.json`, `uncertainty.json` | Programmatic file & shape validation | 4/4 Verified | **PASS** |
| **Input Validation Safety**| Non-existent file, invalid type, corrupt image | Exception intercept boundary test | 4/4 Intercepted | **PASS** |
| **12-Point Automated Suite**| `FootModule` interface integrity | `verification/foot/model/verify_module.py` | 12/12 PASS | **PASS** |
| **End-to-End Acceptance**| Wagner Grade 1, Grade 2, Grade 3, Grade 4 scans | `scratch/run_foot_acceptance.py` | 4/4 Evaluated | **PASS** |
| **Contract Parity** | `RetinaModule` vs `FootModule` output schema | Key-by-key concept comparison | 12/12 Aligned | **MATCH** |

---

## Artifact Locations

- **Primary Source Code**: [`src/foot/foot_module.py`](../../../src/foot/foot_module.py)
- **Package Export**: [`src/foot/__init__.py`](../../../src/foot/__init__.py)
- **12-Point Verification Suite**: [`verification/foot/model/verify_module.py`](../../../verification/foot/model/verify_module.py)
- **End-to-End Acceptance Script**: [`scratch/run_foot_acceptance.py`](../../../scratch/run_foot_acceptance.py)
- **Frozen Model Configuration**: [`experiments/foot/final_model/model_selection.json`](../../../experiments/foot/final_model/model_selection.json)
