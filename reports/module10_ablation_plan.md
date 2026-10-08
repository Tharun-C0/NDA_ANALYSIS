# Module 10: Research Ablation Plan

## Executive Summary

This document specifies the four core ablation studies designed to isolate the performance impact of model architecture, loss formulation, training label quality, and post-hoc threshold calibration in the NDA multi-label clause classification project.

---

## Ablation A: Model Architecture Comparison

- **Objective**: Assess how domain-specific pre-training (Legal-RoBERTa, Legal-BERT) compares against modern general disentangled attention architectures (DeBERTa-v3).
- **Varied Factor**: Transformer backbone model identifier (`saibo/legal-roberta-base`, `nlpaueb/legal-bert-base-uncased`, `microsoft/deberta-v3-base`).
- **Fixed Factors**:
  - Loss function: Standard BCE (`bce`)
  - Training label mode: `all_valid`
  - Max sequence length: 256
  - Learning rate: $2 \times 10^{-5}$
  - Batch size: 8
  - Seed: 42
  - Classification threshold: 0.5
- **Primary Metric**: Macro F1
- **Secondary Metrics**: Micro F1, Weighted F1, Hamming Loss, Macro MCC, Inference Latency
- **Leakage Controls**: Strict document-disjoint split across Train / Val / Test. Zero document overlap.

---

## Ablation B: Loss Function Formulation

- **Objective**: Measure the ability of Focal Loss and Class-Weighted BCE to mitigate severe category imbalance across the 14 NDA categories.
- **Varied Factor**: Training loss function (`bce`, `focal` [$\gamma=2.0, \alpha=0.25$], `class_weighted_bce`).
- **Fixed Factors**:
  - Evaluated separately within each model architecture (Legal-RoBERTa, Legal-BERT, DeBERTa-v3)
  - Training label mode: `all_valid`
  - Hyperparameters: LR = $2 \times 10^{-5}$, Batch = 8, Epochs = 5, Seed = 42
- **Primary Metric**: Macro F1
- **Secondary Metrics**: Per-category F1 on minority categories (e.g., *Remedies for Breach*, *No Assignment of Rights*), Micro F1, Hamming Loss
- **Leakage Controls**: Positive class weights $w_c$ for `class_weighted_bce` are calculated **EXCLUSIVELY from the training split**. Zero validation or test stats used.

---

## Ablation C: Label Quality Mode (`all_valid` vs. `high_confidence`)

- **Objective**: Determine whether training exclusively on high-confidence pseudo-labels (excluding medium confidence and model disagreements) improves test set generalization.
- **Varied Factor**: Training subset filtering (`all_valid` containing 100% valid records vs. `high_confidence` containing only HIGH_CONFIDENCE records).
- **Fixed Factors**:
  - Model architecture: `saibo/legal-roberta-base`
  - Loss function: `bce`
  - Validation and Test evaluation sets remain 100% identical and un-filtered across both runs.
- **Primary Metric**: Macro F1 on Test split
- **Secondary Metrics**: Precision, Recall, Macro MCC
- **Leakage Controls**: Filtering applies ONLY to the training set (`df_train`). Validation (`df_val`) and Test (`df_test`) remain complete and static.

---

## Ablation D: Classification Threshold Calibration

- **Objective**: Evaluate the impact of tuning category classification thresholds $\tau \in [0.1, 0.9]$ compared to the standard default $\tau=0.5$.
- **Varied Factor**: Decision threshold selection strategy (Default $\tau=0.5$ vs. Validation-Tuned $\tau^*$).
- **Fixed Factors**:
  - Trained model weights (evaluating existing checkpoints of EXP-01 through EXP-09)
  - Validation and Test split data
- **Primary Metric**: Macro F1
- **Secondary Metrics**: Micro F1, Precision-Recall Trade-off, Hamming Loss
- **Leakage Controls**: Threshold tuning grid search is conducted **EXCLUSIVELY on Validation logits/labels**. The optimal $\tau^*$ is frozen and applied to Test logits without accessing Test ground-truth labels during tuning.

---

## Summary Ablation Matrix

| Ablation Study | Experiments Involved | Varied Factor | Primary Metric | Core Safeguard |
| :--- | :--- | :--- | :--- | :--- |
| **A. Architecture** | EXP-01, EXP-04, EXP-07 | Model Backbone | Macro F1 | Identical hyperparams & splits |
| **B. Loss Function** | EXP-01 to EXP-09 | Loss Formulation | Macro F1 | Train-only class weight derivation |
| **C. Label Quality** | EXP-01 vs. EXP-10 | Train Quality Filter | Macro F1 | Identical Val/Test evaluation sets |
| **D. Threshold Tuning** | EXP-01 Post-Processing | Threshold $\tau \in [0.1, 0.9]$ | Macro F1 | Val-only tuning grid search |
