# Module 8 - Human Verification & Quality Estimation Plan

> **Status**: **COMPLETE**
> Establishes the protocol for using a stratified human-verified sample to quantify LLM pseudo-label noise.
> **No manual annotation is performed in this module.**

---

## 1. Core Distinction: Distant Supervision vs Human Ground Truth

To preserve research integrity, this project strictly distinguishes two separate evaluation paradigms:

| Evaluation Paradigm | Ground Truth Source | Dataset Scope | Research Purpose |
| :--- | :--- | :--- | :--- |
| **1. Pseudo-Label Evaluation** | 3-Model Gemini LLM Ensemble | Full Document Corpus (668 clauses) | Primary benchmark evaluating multi-label transformer architectures under distant supervision. |
| **2. Human-Verified Quality Audit** | Legal Expert Annotators | Stratified Sample (198 clauses in human_verification_sample.csv) | Quality estimation measuring precision, recall, and error rate of LLM pseudo-labels. |

**CRITICAL RULE**: LLM pseudo-labels are treated as distant supervision signals, **NOT** human ground truth.

---

## 2. Sampling Protocol for human_verification_sample.csv

The human verification set contains **198 clauses** sampled using quality-tier stratification:

| Quality Tier | Sample Count | Sampling Strategy | Rationale |
| :--- | :---: | :--- | :--- |
| **DISAGREEMENT** | 98 clauses | High sampling fraction | Ensemble conflict cases contain highest potential label ambiguity. |
| **LOW_CONFIDENCE** | 46 clauses | Moderate sampling fraction | Model uncertainty cases require verification of true positive boundaries. |
| **HIGH_CONFIDENCE** | 38 clauses | Representative sample | Verifies false positive rate in perfect consensus records. |
| **MEDIUM_CONFIDENCE** | 16 clauses | Representative sample | Verifies majority consensus reliability. |
| **Total Sample** | **198 clauses** | **Stratified Sub-sample** | **Sufficient power to estimate pseudo-label noise with <= 5% margin of error** |

---

## 3. Human Quality Estimation Metrics

Once human annotations are collected on human_verification_sample.csv, the following pseudo-label quality metrics will be computed:

1. **Pseudo-Label Precision**: Fraction of positive LLM pseudo-labels confirmed as correct by human annotator.
2. **Pseudo-Label Recall**: Fraction of true human-annotated positive labels retrieved by LLM ensemble.
3. **Tier-Specific Error Rate**: False positive and false negative error rates broken down by Quality Tier (HIGH, MEDIUM, LOW, DISAGREEMENT).
4. **Category Noise Profile**: Category-level confusion matrix between LLM ensemble and human ground truth.

---

## 4. Operational Guidelines for Future Human Review

1. Annotators must follow reports/annotation_guidelines.md.
2. Annotations are multi-label across all 14 NDA categories.
3. Inter-annotator agreement (Cohen / Fleiss Kappa) will be reported on a 20% double-annotated subset.
4. Human verification labels must **NEVER** be mixed silently into training sets without explicit ablation flags.

---
*Module 8 - Human Verification Plan*
*Created: 2026-10-03*
