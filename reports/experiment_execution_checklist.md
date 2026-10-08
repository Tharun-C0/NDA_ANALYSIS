# Experiment Execution Checklist

> **Status**: Active Project Tracking Checklist.
> Current Status: **Phase 1 IN PROGRESS** / **Phase 2 BLOCKED**.

---

## Phase Execution Overview

| Phase | Description | Status | Blocker / Dependency |
| :---: | :--- | :---: | :--- |
| **Phase 1** | LLM Pseudo-Labeling Pipeline Completion | **IN PROGRESS** | Gemini API free-tier rate limits (~465 clauses remaining) |
| **Phase 2** | Final Document-Disjoint Dataset Construction | **BLOCKED** | Blocked by Phase 1 (requires diverse document coverage) |
| **Phase 3** | Legal-RoBERTa Baseline Experiments (EXP-01a..c) | **NOT STARTED** | Blocked by Phase 2 |
| **Phase 4** | Alternative Architectures (EXP-02a..c, EXP-03a..c) | **NOT STARTED** | Blocked by Phase 3 |
| **Phase 5** | Quality & Threshold Ablations (ABL-A, ABL-B, Sweeps)| **NOT STARTED** | Blocked by Phase 4 |
| **Phase 6** | Final Evaluation & Error Analysis | **NOT STARTED** | Blocked by Phase 5 |
| **Phase 7** | Paper Tables & Final Manuscript Writing | **NOT STARTED** | Blocked by Phase 6 |

---

## Detailed Task Checklist

### Phase 1: Pseudo-Labeling & Active Queue
- [x] Implement Gemini 3-model ensemble classifier (scripts/llm_clause_classifier.py)
- [x] Implement resumable queue runner with lock protection (scripts/resume_runner.py)
- [x] Implement document-diversity priority queuing
- [ ] Process remaining ~465 clauses in queue across 13 unseen documents
- [ ] Validate 100% of final pseudo-labels with zero schema errors

### Phase 2: Final Dataset Construction
- [ ] Inspect full multi-document category distribution
- [ ] Construct document-disjoint 3-way split (Train / Validation / Test)
- [ ] Ensure 14/14 categories represented in Train, Validation, and Test splits
- [ ] Freeze final dataset split (data/classification/)
- [ ] Populate 	able_1_dataset_statistics.csv with true empirical stats

### Phase 3: Legal-RoBERTa Baseline
- [ ] Run EXP-01a: Legal-RoBERTa + BCE (all-valid labels)
- [ ] Run EXP-01b: Legal-RoBERTa + Focal Loss 
- [ ] Run EXP-01c: Legal-RoBERTa + Class-Weighted BCE 
- [ ] Record validation metrics and save model checkpoints

### Phase 4: Alternative Architectures
- [ ] Run EXP-02a: Legal-BERT + BCE 
- [ ] Run EXP-02b: Legal-BERT + Focal Loss 
- [ ] Run EXP-02c: Legal-BERT + Class-Weighted BCE 
- [ ] Run EXP-03a: DeBERTa-v3 + BCE 
- [ ] Run EXP-03b: DeBERTa-v3 + Focal Loss 
- [ ] Run EXP-03c: DeBERTa-v3 + Class-Weighted BCE 

### Phase 5: Ablations & Thresholds
- [ ] Run ABL-A: Legal-RoBERTa + BCE (high-confidence labels only)
- [ ] Execute threshold sweep (0.3, 0.4, 0.5, 0.6, 0.7) on validation split
- [ ] Select optimal per-model thresholds on validation set

### Phase 6: Frozen Test Evaluation & Diagnostics
- [ ] Evaluate all 10 experiment checkpoints on frozen Test split
- [ ] Populate 	able_3_overall_results.csv, 	able_4_per_category_results.csv 
- [ ] Execute Error Analysis Protocol (FP/FN, minority categories, co-occurrence)
- [ ] Populate 	able_7_error_analysis.csv 

### Phase 7: Synthesis & Documentation
- [ ] Verify no placeholder values remain in paper tables
- [ ] Synthesize empirical answers to RQ1, RQ2, RQ3, RQ4
- [ ] Finalize paper submission draft

---
*Module 7 - Research Analysis and Experimental Framework*
*Created: 2026-10-03*
