# Research Contribution Framing

> **Status**: Methodological framing established.
> All claims are framed neutrally as exploratory research investigations.

---

## 1. Guiding Reporting Principles

Until the final document-disjoint benchmark experiments are fully executed, evaluated, and verified:

- **DO NOT** claim:
  - 'our model is better'
  - 'our method is superior'
  - 'state-of-the-art performance achieved'
  - 'improves classification performance'
- **DO** frame all contributions around what the study **investigates**, **systematizes**, and **evaluates**.

---

## 2. Core Contributions of this Study

This study contributes to legal NLP and multi-label text classification in the following concrete ways:

### 2.1 Systematic Transformer Architecture Comparison
- **Investigation**: Evaluates and compares three transformer encoder families (Legal-RoBERTa, Legal-BERT, and DeBERTa-v3) under identical training, evaluation, and loss conditions for 14-category NDA clause classification.
- **Value**: Provides controlled empirical evidence on whether domain-specific legal pre-training outperforms general-domain architecture improvements (e.g. disentangled attention) on NDA clause text.

### 2.2 Imbalance-Aware Multi-Label Loss Evaluation
- **Investigation**: Compares standard Binary Cross-Entropy (BCEWithLogitsLoss), Multi-Label Focal Loss, and Class-Weighted BCE across minority legal categories.
- **Value**: Establishes how loss re-weighting impacts per-category recall and Macro F1 in highly imbalanced legal taxonomies.

### 2.3 Downstream Effects of LLM Pseudo-Label Confidence
- **Investigation**: Analyzes downstream classifier performance when trained on all valid pseudo-labels vs high-confidence ensemble pseudo-labels.
- **Value**: Quantifies the noise-versus-quantity tradeoff when using LLM ensembles for distant supervision in legal tasks.

### 2.4 Document-Disjoint Evaluation Protocol
- **Investigation**: Implements strict document-level data partitioning where no NDA document spans multiple splits.
- **Value**: Ensures realistic evaluation by preventing clause leakage from the same contract template into test results.

### 2.5 Granular 14-Category Diagnostic Benchmark
- **Investigation**: Provides per-category evaluation across all 14 NDA categories alongside error mode breakdowns (FP/FN, support correlation, multi-label confusion).
- **Value**: Identifies specific legal clause types that remain challenging for transformer classification models.

---

## 3. Expected Outputs & Deliverables

1. Reproducible experiment pipeline (scripts/train_multilabel_classifier.py, scripts/evaluate_multilabel_classifier.py).
2. Public paper tables (	able_1_dataset_statistics.csv through 	able_7_error_analysis.csv).
3. Complete empirical completion report and error analysis.

---
*Module 7 - Research Analysis and Experimental Framework*
*Created: 2026-10-03*
