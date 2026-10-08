# Active Learning Baseline Report — Module 5B

> **Status: NOT READY — no genuine human annotations yet**

## Condition for Baseline Training

Baseline training requires at minimum:
- At least **10 annotated clauses** across at least **3 different categories**.
- Annotations must have `annotation_status = ANNOTATED` or `VERIFIED`.
- Annotations must have `label_source = human_annotation`.
- Labels must not be LLM pseudo-labels.

**Current annotation count: 0**

Run `scripts/train_active_learning_baseline.py` after completing seed annotation.

---

## Planned Baseline Architecture

Once sufficient annotations exist:

| Component | Choice |
|---|---|
| Feature extraction | TF-IDF (char n-grams 2-4, word unigrams) |
| Classifier | One-vs-Rest Logistic Regression |
| Problem type | Multi-label classification (14 binary classifiers) |
| Split strategy | Document-disjoint (no clause from same document in train + val) |
| Threshold | 0.5 per label (adjustable) |

## Planned Metrics

- Macro F1 (unweighted average across 14 categories)
- Weighted F1 (weighted by support)
- Micro F1 (global TP/FP/FN)
- Hamming loss (fraction of wrong label assignments)
- Per-category precision, recall, F1, support

## Expected Limitations at Round 0

- Very small training set (≤ 100 clauses)
- High class imbalance likely (some categories may have 0 examples)
- Model used only for **uncertainty estimation**, not final classification
- Results will NOT be reported as research-quality metrics at this stage

---

*This report is generated and updated by `scripts/train_active_learning_baseline.py`.*
