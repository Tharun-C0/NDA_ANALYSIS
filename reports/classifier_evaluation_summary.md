# Multi-Label Classifier Evaluation Report — EXP-01-saibo/legal-roberta-base-SmokeTest [DEBUG / PIPELINE VALIDATION ONLY]

> [!IMPORTANT]
> **RESEARCH DISCLAIMER**: Training & evaluation labels in this pipeline are **LLM-generated pseudo-labels** (produced by a 3-model ensemble: Gemini 3.6 Flash, Gemini 3.5 Flash, Gemini 3.1 Flash Lite). They represent automated pre-annotations for machine learning experimentation and **have NOT yet been fully human-verified**.

> [!NOTE]
> **PRIMARY EVALUATION METRIC**: **Macro F1** is designated as the primary benchmark metric due to category imbalance across the 14 NDA categories.

> [!WARNING]
> **TEMPORARY SPLIT NOTICE**: These evaluation metrics were computed on a temporary 7-document debugging split. They are for code verification only and **MUST NOT be reported as final research results**.

## 1. Global Multi-Label Performance Summary

| Metric Name | Value | Role / Description |
| :--- | :--- | :--- |
| **Macro F1** | `0.0432` | **PRIMARY METRIC** — Unweighted average F1 across 14 categories |
| **Weighted F1** | `0.0380` | Support-weighted average F1 |
| **Micro F1** | `0.0878` | Global instance-level F1 score |
| **Macro Precision** | `0.0235` | Mean precision across 14 categories |
| **Macro Recall** | `0.4286` | Mean recall across 14 categories |
| **Hamming Loss** | `0.4882` | Fraction of misclassified binary labels (lower is better) |
| **Macro MCC** | `-0.0097` | Multi-label Matthews Correlation Coefficient |

## 2. Per-Category Breakdown & Confusion Statistics

| Category Name | Precision | Recall | F1 Score | MCC | Actual Pos | Pred Pos | TP | FP | FN | TN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Party Identification** | 0.1139 | 1.0 | 0.2045 | 0.0 | 9 | 79 | 9 | 70 | 0 | 0 |
| **Purpose** | 0.0 | 0.0 | 0.0 | 0.0 | 6 | 0 | 0 | 0 | 6 | 73 |
| **NDA Type** | 0.0253 | 1.0 | 0.0494 | 0.0 | 2 | 79 | 2 | 77 | 0 | 0 |
| **Definition of Confidential Information** | 0.0506 | 1.0 | 0.0964 | 0.0 | 4 | 79 | 4 | 75 | 0 | 0 |
| **Confidentiality Obligations** | 0.0 | 0.0 | 0.0 | -0.099 | 10 | 5 | 0 | 5 | 10 | 64 |
| **Authorized Disclosure** | 0.1139 | 1.0 | 0.2045 | 0.0 | 9 | 79 | 9 | 70 | 0 | 0 |
| **Non-Confidential Information** | 0.0 | 0.0 | 0.0 | 0.0 | 7 | 0 | 0 | 0 | 7 | 72 |
| **Liability for Damages** | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 0 | 0 | 0 | 1 | 78 |
| **Competition Rights** | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 0 | 0 | 0 | 1 | 78 |
| **Term and Termination** | 0.0 | 0.0 | 0.0 | -0.0372 | 4 | 2 | 0 | 2 | 4 | 73 |
| **Intellectual Property** | 0.0127 | 1.0 | 0.025 | 0.0 | 1 | 79 | 1 | 78 | 0 | 0 |
| **Employees** | 0.0 | 0.0 | 0.0 | 0.0 | 4 | 0 | 0 | 0 | 4 | 75 |
| **Governing Law and Jurisdiction** | 0.0127 | 1.0 | 0.025 | 0.0 | 1 | 79 | 1 | 78 | 0 | 0 |
| **Additional Information** | 0.0 | 0.0 | 0.0 | 0.0 | 52 | 0 | 0 | 0 | 52 | 27 |