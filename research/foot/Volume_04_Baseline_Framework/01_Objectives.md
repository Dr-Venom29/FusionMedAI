# 01 Objectives — Foot Ulcer Baseline Framework

## Phase 10.4 Objectives

1. **Establish Standard Reference Benchmark**: Create an un-augmented, standardized baseline using ResNet-50 on the frozen canonical 10,050 image population.
2. **Freeze Experimental Parameters**: Explicitly freeze all hyperparameters, data transforms, normalization stats, random seeds (`42`), and loss functions prior to evaluation.
3. **Isolate Single Modality**: Restrict baseline scope strictly to 4-class Wagner classification (`Grade 1`, `Grade 2`, `Grade 3`, `Grade 4`) without fusion, uncertainty, calibration, or ensembling.
4. **Evaluate Single-Run Frozen Test Set**: Evaluate the best validation-loss checkpoint once against the frozen test partition (1,006 images / 181 source groups).
5. **Characterize Errors & Overfitting**: Document exact confusion boundaries (especially Grade 2 vs Grade 3) and baseline error rates to establish a defensible baseline score for Phase 10.5 architecture benchmarking.
