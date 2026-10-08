# Active Learning Research Log — Module 5B

## Project
NDA Clause Classification — Active Learning Pipeline  
Paper reference: *"A Two-Stage Architecture for NDA Analysis: LLM-based Segmentation and Transformer-based Clause Classification"*

---

## Round 0 — Seed Annotation (Initial Human Annotation)

**Status:** IN PROGRESS

| Metric | Value |
|---|---|
| Seed set size | 100 clauses |
| Documents covered | 20 / 20 |
| Human annotations completed | 0 |
| Human annotations verified | 0 |
| Baseline model trained | NO |
| Uncertainty sampling performed | NO |

**Seed selection method:**  
Document-spread + length-bucket diversity (see `reports/active_learning_seed_selection.md`).  
No LLM or automated labelling used.

**Categories covered so far:** None (0 annotations)

**Next action:**  
Complete annotation of 100 seed clauses using `app/annotation_app.py --mode seed`.  
After all 100 seed clauses are annotated, proceed to Round 1.

---

## Round 1 — Baseline + Uncertainty Sampling

**Status:** NOT STARTED — awaiting Round 0 completion

**Planned steps:**
1. Train TF-IDF + One-vs-Rest Logistic Regression on seed annotations.
2. Predict remaining ~599 unlabeled clauses.
3. Compute uncertainty score (mean margin from 0.5 across all labels).
4. Select top-50 most uncertain clauses → `data/annotations/next_active_learning_batch.csv`.
5. Human annotator reviews and labels 50 uncertain clauses.

**Model save path:** `models/active_learning_baseline/round_1/`  
**Report path:** `reports/active_learning_baseline.md`

---

## Round 2+ — Iterative Refinement

**Status:** NOT STARTED

Each subsequent round follows the same loop:
- Retrain on all accumulated human annotations
- Predict remaining unlabeled clauses
- Select 50 most uncertain
- Human annotation
- Repeat

---

## Data Integrity Rules (all rounds)

- Human annotations = ground truth
- LLM predictions are never automatically promoted to annotations
- No unlabeled clauses are used as training examples
- Document-disjoint splits enforced before final evaluation
- All annotation provenance recorded: reviewer, timestamp, round, source

---

*Log is updated automatically by `scripts/run_active_learning_round.py`.*
