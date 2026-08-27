# Chapter 1: Introduction

## 1.1 Purpose of Benchmarking
In Step 5, the objective shifts from establishing a baseline to identifying the optimal architectural backbone for retinal image classification. A rigorous benchmarking protocol is required to objectively evaluate various state-of-the-art architectures without hyperparameter bias.

## 1.2 Comparing CNNs and Transformers
Convolutional Neural Networks (CNNs) have historically dominated medical image analysis due to their strong inductive bias for local spatial features. However, Vision Transformers (ViTs) and hierarchical transformers (Swin) have recently shown exceptional capabilities in capturing global contextual relationships. This phase empirically compares these paradigms directly on the APTOS 2019 dataset to determine which architecture is best suited for diabetic retinopathy grading.

## 1.3 Experimental Objectives
- Compare multiple CNN and Transformer architectures under an identical experimental setup.
- Eliminate confounding variables by freezing hyperparameters and preprocessing pipelines.
- Evaluate models on diagnostic performance (Accuracy, QWK, ROC-AUC).
- Evaluate models on computational efficiency (Latency, Throughput, VRAM).
- Select a single, robust backbone for the final Retinal Module.
