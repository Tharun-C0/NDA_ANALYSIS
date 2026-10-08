# Research Questions — NDA Multi-Label Clause Classification Study

> **Status**: These research questions define the investigation to be conducted.
> They have **NOT** been answered yet. All experiments remain pending completion
> of the pseudo-labeling pipeline and construction of the final document-disjoint dataset.

---

## Overview

This study investigates multi-label classification of NDA (Non-Disclosure Agreement)
clauses across 14 legally-defined categories. The work extends an existing pipeline
(LLaMA-3.1-8B-Instruct for segmentation + Legal-RoBERTa for classification) by
comparing alternative transformer architectures, loss functions, and training strategies
under conditions of class imbalance and LLM-generated pseudo-label noise.

Classification labels are **LLM pseudo-labels** produced by a 3-model Gemini ensemble.
They are **not human ground-truth annotations** and should be interpreted accordingly.

---

## RQ1 — Transformer Architecture Comparison

**How do different transformer architectures perform for multi-label NDA clause
classification under identical training conditions?**

### Scope
| Model | Hub ID | Notes |
| :--- | :--- | :--- |
| Legal-RoBERTa | `saibo/legal-roberta-base` | Baseline (replicates original paper model family) |
| Legal-BERT | `nlpaueb/legal-bert-base-uncased` | Legal-domain BERT variant |
| DeBERTa-v3 | `microsoft/deberta-v3-base` | General-domain disentangled-attention model |

### Expected Output
- Per-model Macro F1, Micro F1, Weighted F1, Hamming Loss, MCC
- Per-category precision, recall, F1, support
- Comparison table aggregated across all loss conditions

### Constraints
- All models trained on identical document-disjoint training split
- Evaluation performed exclusively on the frozen test split

---

## RQ2 — Loss Function Comparison Under Class Imbalance

**How do different loss functions affect multi-label NDA clause classification
performance, particularly for minority categories, under class imbalance?**

### Scope
| Loss Function | Description |
| :--- | :--- |
| BCEWithLogitsLoss | Standard binary cross-entropy baseline |
| Focal Loss (gamma=2.0, alpha=0.25) | Penalises easy examples; designed for imbalance |
| Class-Weighted BCE | Per-class weights derived from training split frequency |

### Expected Output
- Per-loss Macro F1 averaged across models
- Loss function x architecture interaction table
- Per-category improvement/degradation relative to BCE baseline

### Constraints
- Class weights computed from training data only (no leakage)
- Focal Loss gamma/alpha treated as fixed

---

## RQ3 — Pseudo-Label Confidence Effect on Classification

**How does pseudo-label confidence level affect downstream multi-label
classification performance?**

### Scope
| Condition | Clauses Included |
| :--- | :--- |
| All valid pseudo-labels | HIGH_CONFIDENCE + MEDIUM_CONFIDENCE + LOW_CONFIDENCE + DISAGREEMENT |
| High-confidence only | HIGH_CONFIDENCE + MEDIUM_CONFIDENCE (agreement_score >= 0.5, confidence >= 0.6) |

### Expected Output
- Macro F1 comparison between conditions per model
- Analysis of which categories are most affected by label quality filtering
- Discussion of the noise-vs-quantity tradeoff

---

## RQ4 — Minority Category Classification Consistency

**How consistently can the models classify minority NDA clause categories?**

### Scope
For every one of the 14 categories, analyze:

| Metric | Purpose |
| :--- | :--- |
| Per-category Precision | Fraction of positive predictions that are correct |
| Per-category Recall | Fraction of true positives correctly retrieved |
| Per-category F1 | Harmonic mean of precision and recall |
| Per-category Support | Number of positive instances in the test split |
| False Positives | Predicted positive but actually negative |
| False Negatives | Predicted negative but actually positive |

### Expected Output
- Complete 14-category performance table for each model x loss condition
- Identification of consistently underperforming categories
- Correlation between category support and per-category F1

---

## Status

| RQ | Status | Blocked By |
| :--- | :--- | :--- |
| RQ1 | NOT STARTED | Final pseudo-labeling + document-disjoint split |
| RQ2 | NOT STARTED | Final pseudo-labeling + document-disjoint split |
| RQ3 | NOT STARTED | Final pseudo-labeling + document-disjoint split |
| RQ4 | NOT STARTED | Final pseudo-labeling + document-disjoint split |

---
*Module 7 — Research Analysis and Experimental Framework*
*Created: 2026-10-03*
