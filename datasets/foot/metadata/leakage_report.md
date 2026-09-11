# Phase 10.1.I — Data Leakage & Patient-Case Investigation Report

## Executive Summary
- **Total Image Files**: `10,062`
- **Unique Source Images Discovered**: `1,770`
- **Augmentation Expansion Factor**: `5.68×` (Source images expanded via offline Roboflow augmentations)
- **Source Image Groups with Variants**: `1,357`
- **Patient Identifier Availability**: `UNAVAILABLE` (Limitation Documented)

## Data Leakage Audit Findings
| Leakage Audit Check | Count Detected | Status | Clinical & Evaluation Risk Assessment |
| :--- | :---: | :---: | :--- |
| **Source Image Cross-Split Leakage** | `10` | ⚠️ LEAK DETECTED | Augmented variants of the same source image exist in both train and test/valid splits |
| **Source Image Cross-Class Conflict** | `23` | ⚠️ CONFLICT DETECTED | Augmented variants of the same source image labeled under different Wagner grades |
| **Perceptual Cross-Split Leakage** | `0` | ✅ PASS | Visually near-identical images shared between train and test splits |

### Sample Source Image Cross-Split Leakage Instances
**Source Group 1** (`24_jpg` — 8 variants):
  - `test/Grade 2/24_jpg.rf.054e4113acb58ff05047216f70d2f788.jpg` (Split: **test**, Class: **Grade 2**)
  - `train/Grade 4/24_jpg.rf.15b3184a94346865edf368bda34aa19f.jpg` (Split: **train**, Class: **Grade 4**)
  - `train/Grade 4/24_jpg.rf.49c0a7e4eb145bb6816c848177c2050d.jpg` (Split: **train**, Class: **Grade 4**)
  - `train/Grade 4/24_jpg.rf.50d4851e5a4a02c6e7fca5fc1f03775b.jpg` (Split: **train**, Class: **Grade 4**)
  - `train/Grade 4/24_jpg.rf.51a91b2dda947fb21e80aa1e51dd8e6c.jpg` (Split: **train**, Class: **Grade 4**)
  - `train/Grade 4/24_jpg.rf.65cbce4fab66c9d49c5812f0675661e5.jpg` (Split: **train**, Class: **Grade 4**)
  - `train/Grade 4/24_jpg.rf.74f3a81059f06f300879bb82e379a537.jpg` (Split: **train**, Class: **Grade 4**)
  - `train/Grade 4/24_jpg.rf.fb061a1412241d7865086614bf4845b5.jpg` (Split: **train**, Class: **Grade 4**)
**Source Group 2** (`29_jpg` — 8 variants):
  - `test/Grade 2/29_jpg.rf.8e7a7900343ab80324ab3bf27d4dc7d7.jpg` (Split: **test**, Class: **Grade 2**)
  - `train/Grade 3/29_jpg.rf.1c529d195d28fde567f5fbc03c5961e7.jpg` (Split: **train**, Class: **Grade 3**)
  - `train/Grade 3/29_jpg.rf.662df89973b763174ff80189b988d50a.jpg` (Split: **train**, Class: **Grade 3**)
  - `train/Grade 3/29_jpg.rf.6f347376723963832870629077c2ac8b.jpg` (Split: **train**, Class: **Grade 3**)
  - `train/Grade 3/29_jpg.rf.91961bdba99f57df746a0ef2d8b8a888.jpg` (Split: **train**, Class: **Grade 3**)
  - `train/Grade 3/29_jpg.rf.cf7fdf0e18cb39ab9bc3e86414bb62d2.jpg` (Split: **train**, Class: **Grade 3**)
  - `train/Grade 3/29_jpg.rf.e0448a254e565007eb7be44667990d8c.jpg` (Split: **train**, Class: **Grade 3**)
  - `train/Grade 3/29_jpg.rf.e1ca30dea4606da24a0860ac5d0c34f9.jpg` (Split: **train**, Class: **Grade 3**)
**Source Group 3** (`7_jpg` — 8 variants):
  - `test/Grade 2/7_jpg.rf.9d42721e5267dd74b9b4a62dce8d7373.jpg` (Split: **test**, Class: **Grade 2**)
  - `train/Grade 1/7_jpg.rf.07748e56e6f45d71225a5e7e7e898fe8.jpg` (Split: **train**, Class: **Grade 1**)
  - `train/Grade 1/7_jpg.rf.2b3e2f9a5ea7fb0305a26b110afe730e.jpg` (Split: **train**, Class: **Grade 1**)
  - `train/Grade 1/7_jpg.rf.65c44067badfe1a66636726d8f5ab856.jpg` (Split: **train**, Class: **Grade 1**)
  - `train/Grade 1/7_jpg.rf.8d4c5211593d9ff8aec39a432505d63c.jpg` (Split: **train**, Class: **Grade 1**)
  - `train/Grade 1/7_jpg.rf.ae6c3245acaadd2faa4bc5fd3483bed8.jpg` (Split: **train**, Class: **Grade 1**)
  - `train/Grade 1/7_jpg.rf.b6293673cac8c6a7182f09a99a0fb33e.jpg` (Split: **train**, Class: **Grade 1**)
  - `train/Grade 1/7_jpg.rf.d23dfcfd5e96443200dcd114ff0ab220.jpg` (Split: **train**, Class: **Grade 1**)
**Source Group 4** (`21_jpg` — 8 variants):
  - `test/Grade 3/21_jpg.rf.8bebd7b1836cbf251f35ae2c0e3af167.jpg` (Split: **test**, Class: **Grade 3**)
  - `train/Grade 3/21_jpg.rf.006fae65302b3033765512f24a1591a0.jpg` (Split: **train**, Class: **Grade 3**)
  - `train/Grade 3/21_jpg.rf.095925033371edd5c80cad631e982d7c.jpg` (Split: **train**, Class: **Grade 3**)
  - `train/Grade 3/21_jpg.rf.4016be5afe1893450f5f12c332876dda.jpg` (Split: **train**, Class: **Grade 3**)
  - `train/Grade 3/21_jpg.rf.5051580cfb5c16f4ecdb6e983df0abda.jpg` (Split: **train**, Class: **Grade 3**)
  - `train/Grade 3/21_jpg.rf.616e7e6e2375433219e4702f664ffaa4.jpg` (Split: **train**, Class: **Grade 3**)
  - `train/Grade 3/21_jpg.rf.971998e850e4bad936917e45a6538cc6.jpg` (Split: **train**, Class: **Grade 3**)
  - `train/Grade 3/21_jpg.rf.9a1ac09af1cbba6af6ba9bedf794c9b7.jpg` (Split: **train**, Class: **Grade 3**)
**Source Group 5** (`4_jpg` — 8 variants):
  - `test/Grade 3/4_jpg.rf.9f9060acb97d787c059b080c7b351c1a.jpg` (Split: **test**, Class: **Grade 3**)
  - `train/Grade 4/4_jpg.rf.819e31f00f41b06a54f3a46683492acd.jpg` (Split: **train**, Class: **Grade 4**)
  - `train/Grade 4/4_jpg.rf.9c96e4f54ed01eed77b2ea522cb9afe7.jpg` (Split: **train**, Class: **Grade 4**)
  - `train/Grade 4/4_jpg.rf.a943a4abde2fba62719b1a3e3e7f0476.jpg` (Split: **train**, Class: **Grade 4**)
  - `train/Grade 4/4_jpg.rf.aca10f5cabb6eff19047a210f7da7d5d.jpg` (Split: **train**, Class: **Grade 4**)
  - `train/Grade 4/4_jpg.rf.ce37761a1f727149e83b0d885918c0f1.jpg` (Split: **train**, Class: **Grade 4**)
  - `train/Grade 4/4_jpg.rf.fc77a7855cc2c32a638b9f57424a7813.jpg` (Split: **train**, Class: **Grade 4**)
  - `train/Grade 4/4_jpg.rf.feaa881c75a25b0918732a55b2e31f79.jpg` (Split: **train**, Class: **Grade 4**)

## Patient-Level Separation Limitation Notice
> **Scientific Integrity Notice**: 
> Patient-level metadata (e.g. Patient ID / Case ID) is NOT provided in the raw dataset. Images are grouped by Roboflow source image IDs (extractable via filename prefixes). Patient-level separation cannot be guaranteed beyond source-image group separation. This limitation is explicitly documented to prevent false claims of patient-level isolation.

## Downstream Action Plan for Step 10.2 (Data Pipeline)
1. **Group-Aware Splitting**: In Step 10.2, dataset partitions must be constructed by grouping all augmented variants of each `source_image_id` into the SAME partition split (GroupKFold / Group-Based Stratified Split).
2. **Leakage Elimination**: Source-level group partitioning eliminates cross-split leakage between training and evaluation sets.