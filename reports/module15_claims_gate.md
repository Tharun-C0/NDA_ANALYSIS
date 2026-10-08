# Module 15: Publication Claims Gate & Boundary Audit

## Executive Summary

This document enforces strict boundaries on all research claims made in project documentation, papers, and presentations.

---

## 1. Research Claims Classification Matrix

### Category A: CLAIMS CURRENTLY SUPPORTED BY EMPIRICAL EVIDENCE
- [x] "A zero-leakage, document-disjoint multi-label classification pipeline for NDA clauses has been constructed and verified."
- [x] "An ensemble pseudo-labeling pipeline (`gemini-3.6-flash`, `3.5-flash`, `3.1-flash-lite`) generates reproducible multi-label clause annotations."
- [x] "Category coverage concentration in a small subset of documents acts as a bottleneck for document-disjoint benchmark construction."
- [x] "Document diversity supersedes static clause volume targets for legal contract benchmark construction."

---

### Category B: CLAIMS THAT REQUIRE MORE DATA (DEPENDENT ON EXPANSION)
- [ ] "The evaluation benchmark covers all 14 NDA categories across Train, Validation, and Test splits simultaneously with multi-document redundancy."
- [ ] "The dataset contains 20 represented legal contracts."

---

### Category C: CLAIMS THAT REQUIRE HUMAN VERIFICATION
- [ ] "The dataset provides a human-verified gold-standard benchmark for NDA clause classification."
- [ ] "Model predictions achieve X% Macro F1 against human ground truth."

---

### Category D: CLAIMS THAT MUST NOT BE MADE (STRICTLY PROHIBITED)
- ❌ **"Legal-RoBERTa / DeBERTa-v3 achieves state-of-the-art Macro F1 on NDA classification."** (Models have not been trained on final benchmark).
- ❌ **"LLM pseudo-labels equal human ground-truth labels."** (Pseudo-labels carry automated model biases).
- ❌ **"The benchmark dataset is finalized and ready for release."** (Benchmark remains BLOCKED by category redundancy).
