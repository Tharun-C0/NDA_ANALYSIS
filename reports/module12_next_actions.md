# Module 12: Actionable Research Roadmap & Next Steps

## Executive Summary

This document categorizes all immediate, short-term, and long-term actions for the NDA research project into three operational tiers based on external API dependencies and benchmark readiness.

---

## 1. Action Tiers

### TIER 1: NOW (Can Be Done Immediately Without Gemini API)
- [x] Execute pseudo-label dataset validation (`scripts/validate_pseudolabels.py`).
- [x] Execute experiment integrity & zero-leakage checks (`scripts/validate_experiment_integrity.py`).
- [x] Run benchmark readiness audit (`scripts/check_benchmark_readiness.py`).
- [x] Generate Document $\times$ Category Matrix (`reports/module12_document_category_matrix.csv`, `.md`).
- [x] Generate Category Document Coverage Report (`reports/module12_category_document_coverage.csv`).
- [x] Simulate document-disjoint split strategies (Strategies A, B, C, D via `scripts/_simulate_module12_splits.py`).
- [x] Audit publication readiness and claim constraints (`reports/module12_publication_readiness.md`).
- [x] Issue formal No-Go decision for classifier training (`reports/module12_experiment_go_no_go.md`).
- [ ] Draft research paper methodology section (Model architectures, Focal Loss, Class-Weighted BCE, Document-Disjoint split protocol).
- [ ] Prepare human annotation guidelines for the test set gold verification.

---

### TIER 2: WAIT (Requires Gemini API Quota Reset)
- [ ] Await Gemini API quota recovery (~17.68h retry window).
- [ ] Execute `python scripts/resume_runner.py --max-clauses 45` to process the diversity-first 45-clause batch across the 9 Tier 1 unseen documents (`586c367e`, `64303e5a`, `9a5cb310`, `b82a10c4`, `c58882f7`, `d4566b17`, `d714d261`, `e52e4a13`, `f28c4f3d`).
- [ ] Audit post-batch dataset expansion to confirm represented documents increase from 11 to 14+.
- [ ] Verify if *Liability for Damages* reaches $\ge 4$ represented documents.

---

### TIER 3: LATER (Requires a Defensible Benchmark & Authorized Expansion)
- [ ] Generate final research benchmark dataset splits (`data/classification/train.csv`, `val.csv`, `test.csv`).
- [ ] Execute the official 10-experiment matrix (`EXP-01` through `EXP-10`) across Legal-RoBERTa, Legal-BERT, and DeBERTa-v3.
- [ ] Perform validation threshold tuning ($\tau^* \in [0.1, 0.9]$) and evaluate test set Macro F1.
- [ ] Populate paper result tables (`reports/paper_tables/*.csv`).
- [ ] Conduct per-category error analysis and confusion matrix breakdown.
