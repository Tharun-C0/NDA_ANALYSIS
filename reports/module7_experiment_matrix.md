# Module 7 - Experiment Matrix & Fixed Hyperparameter Specifications

> **Status**: EXPERIMENTAL MATRIX DEFINED ONLY.
> **DO NOT RUN EXPERIMENTS** until Phase 2 dataset construction is complete.

---

## 1. Primary Experiment Matrix (3 Architectures x 3 Loss Functions)

The primary evaluation grid tests 3 transformer model families against 3 multi-label loss formulations:

| Experiment ID | Model Architecture | HuggingFace Hub ID | Loss Function | Data Subset |
| :---: | :--- | :--- | :--- | :--- |
| **EXP-01a** | Legal-RoBERTa Baseline | saibo/legal-roberta-base | BCEWithLogitsLoss | All Valid Labels |
| **EXP-01b** | Legal-RoBERTa Baseline | saibo/legal-roberta-base | Multi-Label Focal Loss | All Valid Labels |
| **EXP-01c** | Legal-RoBERTa Baseline | saibo/legal-roberta-base | Class-Weighted BCE | All Valid Labels |
| **EXP-02a** | Legal-BERT Alternative | nlpaueb/legal-bert-base-uncased | BCEWithLogitsLoss | All Valid Labels |
| **EXP-02b** | Legal-BERT Alternative | nlpaueb/legal-bert-base-uncased | Multi-Label Focal Loss | All Valid Labels |
| **EXP-02c** | Legal-BERT Alternative | nlpaueb/legal-bert-base-uncased | Class-Weighted BCE | All Valid Labels |
| **EXP-03a** | DeBERTa-v3 Alternative | microsoft/deberta-v3-base | BCEWithLogitsLoss | All Valid Labels |
| **EXP-03b** | DeBERTa-v3 Alternative | microsoft/deberta-v3-base | Multi-Label Focal Loss | All Valid Labels |
| **EXP-03c** | DeBERTa-v3 Alternative | microsoft/deberta-v3-base | Class-Weighted BCE | All Valid Labels |

---

## 2. Pseudo-Label Quality Ablations

To isolate the downstream impact of distant supervision noise vs sample quantity:

| Experiment ID | Model Architecture | Loss Function | Label Selection Subset | Description |
| :---: | :--- | :--- | :--- | :--- |
| **ABL-A** | Legal-RoBERTa | BCEWithLogitsLoss | high_confidence | Trained ONLY on HIGH_CONFIDENCE + MEDIUM_CONFIDENCE records |
| **ABL-B** | Legal-RoBERTa | BCEWithLogitsLoss | all_valid | Trained on all valid labels (identical to EXP-01a) |

---

## 3. Decision Threshold Analysis Protocol

Evaluated for all models on the **Validation Split Only**:
- **Threshold Values**: 0.3 / 0.4 / 0.5 / 0.6 / 0.7
- Optimal threshold selected by highest validation Macro F1, then applied once to frozen Test split.

---

## 4. Verified Training Hyperparameters

All values below match configs/experiments_config.json:

| Parameter | Configuration Value | Status |
| :--- | :---: | :--- |
| **Random Seed** | 42 | VERIFIED |
| **Max Sequence Length** | 256 | VERIFIED |
| **Batch Size** | 8 | VERIFIED |
| **Learning Rate** | 2e-5 | VERIFIED |
| **Weight Decay** | 0.01 | VERIFIED |
| **Training Epochs** | 5 | VERIFIED |
| **Warmup Ratio** | 0.1 | VERIFIED |
| **Default Threshold** | 0.5 | VERIFIED |
| **Focal Loss Gamma** | 2.0 | VERIFIED |
| **Focal Loss Alpha** | 0.25 | VERIFIED |
| **Optimizer** | AdamW | VERIFIED |
| **LR Scheduler** | linear with warmup | VERIFIED |
| **Early Stopping Patience** | TBD | TO BE DETERMINED BEFORE FINAL TRAINING |
| **Gradient Accumulation** | TBD | TO BE DETERMINED BEFORE FINAL TRAINING |

---
*Module 7 - Research Analysis and Experimental Framework*
*Created: 2026-10-03*
