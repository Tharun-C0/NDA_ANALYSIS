# Module 12: Publication Readiness & Research Ethics Report

## Executive Summary

This document evaluates the scientific rigor, provenance, evaluation protocols, and publication readiness of the NDA multi-label clause classification project. It explicitly distinguishes supportable research claims from unsupported claims.

---

## 1. Publication Safeguards & Provenance Matrix

### A. Dataset Provenance
- Sourced from the public **Kleister-NDA** corpus (540 Non-Disclosure Agreement PDF documents).
- Clause segmentation performed using layout-aware paragraph boundaries (`data/annotations/annotation_queue.csv`).

### B. Public Kleister-NDA Source
- The original Kleister-NDA dataset provides PDF documents and document-level key-value extraction targets (e.g., party names, effective dates).

### C. Key Difference: Document-Level vs. Clause-Level Dataset
- **Public Kleister-NDA**: Document-level metadata extraction.
- **Our Benchmark**: Fine-grained **clause-level segmentation** and 14-category **multi-label classification** annotations (e.g., *Confidentiality Obligations*, *Authorized Disclosure*, *Liability for Damages*).

### D. LLM Pseudo-Label Provenance
- All operational labels bear explicit provenance: `label_source = "llm_pseudo_label"`.
- Produced by a 3-model ensemble (`gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.1-flash-lite`).
- Metadata preserved per row: `model_a_confidence`, `model_b_confidence`, `model_c_confidence`, `average_confidence`, `agreement_score`, `pseudo_label_quality`.

### E. Human Verification Requirement
- LLM pseudo-labels are automated pre-annotations and **are NOT human ground truth**.
- A stratified human verification sample of 30 clauses (`data/annotations/human_verification_sample.csv`) has been generated.
- **Publication Rule**: Final paper results MUST report metrics evaluated against human-verified gold labels for the test split.

### F. Document-Disjoint Evaluation Requirement
- All evaluations MUST enforce strict document-disjoint partitioning ($Train \cap Val = \emptyset, Train \cap Test = \emptyset, Val \cap Test = \emptyset$).
- Clause-level random splitting is strictly prohibited to prevent data leakage from identical boilerplate legal phrasing across clauses of the same document.

### G. Leakage Controls
- Enforced via `scripts/validate_experiment_integrity.py`:
  1. Document disjointness
  2. Clause ID uniqueness
  3. Exact clause text isolation
  4. Class-weight derivation from Train split ONLY
  5. Threshold tuning on Validation split ONLY

### H. Class Imbalance Concerns
- Heavy long-tail imbalance: *Additional Information* (164 clauses) vs. *Liability for Damages* (4 clauses).
- **Protocol**: Models MUST be evaluated using **Macro F1** as the primary metric, and trained using imbalanced loss formulations (Multi-Label Focal Loss, Class-Weighted BCE).

### I. Reproducibility Requirements
- Deterministic random seed (`seed = 42`).
- Fixed model identifiers (`saibo/legal-roberta-base`, `nlpaueb/legal-bert-base-uncased`, `microsoft/deberta-v3-base`).
- Structured JSON experiment logging in `reports/experiments/results/`.

---

## 2. Claim Classification Audit

### J. What Claims Are Currently Supportable
1. **Pipeline Infrastructure**: Complete, fully tested, and zero-leakage training/evaluation code pipeline.
2. **Dataset Creation Methodology**: Reproducible multi-model LLM pseudo-labeling and confidence-tiering methodology.
3. **Category Redundancy Analysis**: Empirical identification of document-level category concentration bottlenecks (*Liability for Damages* 2 docs).
4. **Feasibility Audit**: Rigorous empirical proof showing why static document count targets do not guarantee benchmark readiness.

### K. What Claims Are NOT Currently Supportable
1. ❌ **"Model X achieves Y% Macro F1 on NDA classification"** (Models have not been trained on final benchmark).
2. ❌ **"LLM pseudo-labels equal human ground truth"** (Pseudo-labels carry automated model noise and biases).
3. ❌ **"DeBERTa-v3 outperforms Legal-RoBERTa on NDA clauses"** (Final experiments are NOT RUN).
4. ❌ **"The evaluation benchmark is complete and ready for release"** (Benchmark is BLOCKED by document diversity).
