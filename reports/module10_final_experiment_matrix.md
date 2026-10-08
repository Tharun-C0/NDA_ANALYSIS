# Module 10: Final Experiment Matrix

## Executive Summary

This document defines the official 10-experiment matrix for the NDA multi-label clause classification project.

- **Total Planned Experiments**: 10
- **Execution Status**: **NOT_RUN** (All experiments are paused awaiting dataset expansion and benchmark authorization).
- **Core Comparisons**: Architecture (Legal-RoBERTa vs. Legal-BERT vs. DeBERTa-v3), Loss Function (BCE vs. Focal vs. Class-Weighted BCE), and Label Quality (`all_valid` vs. `high_confidence`).

---

## Final 10-Experiment Matrix

| Experiment ID | Model Architecture | Loss Function | Label Mode | Intended Purpose / Research Question | Execution Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **EXP-01** | `saibo/legal-roberta-base` | `bce` | `all_valid` | Baseline Legal-RoBERTa model with standard BCE loss. Primary anchor for all model & loss comparisons. | **NOT_RUN** |
| **EXP-02** | `saibo/legal-roberta-base` | `focal` | `all_valid` | Tests Focal Loss ($\gamma=2.0, \alpha=0.25$) on Legal-RoBERTa to address hard/minority category imbalance. | **NOT_RUN** |
| **EXP-03** | `saibo/legal-roberta-base` | `class_weighted_bce` | `all_valid` | Tests Class-Weighted BCE (weights computed strictly on train split) on Legal-RoBERTa. | **NOT_RUN** |
| **EXP-04** | `nlpaueb/legal-bert-base-uncased` | `bce` | `all_valid` | Evaluates domain-adapted Legal-BERT baseline with standard BCE loss. | **NOT_RUN** |
| **EXP-05** | `nlpaueb/legal-bert-base-uncased` | `focal` | `all_valid` | Evaluates Legal-BERT with Multi-Label Focal Loss. | **NOT_RUN** |
| **EXP-06** | `nlpaueb/legal-bert-base-uncased` | `class_weighted_bce` | `all_valid` | Evaluates Legal-BERT with Class-Weighted BCE loss. | **NOT_RUN** |
| **EXP-07** | `microsoft/deberta-v3-base` | `bce` | `all_valid` | Evaluates state-of-the-art DeBERTa-v3 architecture with standard BCE loss. | **NOT_RUN** |
| **EXP-08** | `microsoft/deberta-v3-base` | `focal` | `all_valid` | Evaluates DeBERTa-v3 with Multi-Label Focal Loss. | **NOT_RUN** |
| **EXP-09** | `microsoft/deberta-v3-base` | `class_weighted_bce` | `all_valid` | Evaluates DeBERTa-v3 with Class-Weighted BCE loss. | **NOT_RUN** |
| **EXP-10** | `saibo/legal-roberta-base` | `bce` | `high_confidence` | **Label Quality Ablation**: Trains Legal-RoBERTa exclusively on HIGH_CONFIDENCE pseudo-labels. | **NOT_RUN** |

---

## Experiment Grouping & Research Rationale

### Group A: Model Architecture Comparison (EXP-01, EXP-04, EXP-07)
- **Goal**: Compare domain-pretrained Legal-RoBERTa and Legal-BERT against modern DeBERTa-v3 under identical standard BCE loss conditions.
- **Fixed Parameters**: Loss = `bce`, Label Mode = `all_valid`, Seed = 42, Epochs = 5, LR = 2e-5.

### Group B: Loss Function Comparison (EXP-01 through EXP-09)
- **Goal**: Measure the efficacy of Focal Loss and Class-Weighted BCE in mitigating heavy category imbalance across all three model architectures.
- **Fixed Parameters**: Architecture fixed within triplets, Label Mode = `all_valid`, Seed = 42.

### Group C: Label Quality Ablation (EXP-01 vs. EXP-10)
- **Goal**: Test whether filtering out medium-confidence and disagreement pseudo-labels improves model generalization on unseen document test sets.
- **Fixed Parameters**: Model = `saibo/legal-roberta-base`, Loss = `bce`, Seed = 42.

---

## Execution Protocol Reminder

> [!IMPORTANT]
> No experiment entries in this matrix shall be populated with mock or fabricated numbers. All metrics will be recorded automatically into `reports/experiments/results/` upon authorized pipeline execution.
