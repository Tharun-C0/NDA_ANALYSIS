# Evaluation Protocol

> Status: Protocol defined. No experiments executed yet.
> All metric values are currently PLACEHOLDER / NOT COMPUTED.

---

## 1. Primary Metric: Macro F1

Macro-F1 = (1/14) * sum_c [ 2 * P_c * R_c / (P_c + R_c) ]

### Why Macro F1 for this problem

1. The 14-category NDA taxonomy is imbalanced. Categories such as Party Identification
   and Additional Information appear far more frequently than categories such as
   Employees or Competition Rights.

2. Accuracy is misleading for imbalanced multi-label problems: a model that predicts
   only majority categories can achieve high accuracy while failing on minority
   categories. Accuracy is NOT used as a primary or secondary metric.

3. Macro F1 gives equal weight to every category regardless of support. Poor
   minority-category performance cannot be hidden by majority-category gains.

4. Macro F1 is the standard primary metric in legal NLP multi-label classification
   tasks (ECHR, EUR-Lex, MultiEURLEX benchmarks) and is consistent with the original
   paper approach.

---

## 2. Secondary Metrics

| Metric | Interpretation |
| :--- | :--- |
| Micro F1 | Performance weighted by overall label frequency |
| Weighted F1 | F1 weighted by per-category test support |
| Hamming Loss | Fraction of incorrect label decisions across all categories |
| MCC | Multi-label Matthews Correlation Coefficient; balanced under imbalance |

All secondary metrics are reported alongside Macro F1 in every result table.
No secondary metric overrides the primary metric for model selection.

---

## 3. Per-Category Evaluation

For each of the 14 NDA categories, report per model x loss condition:

| Field | Description |
| :--- | :--- |
| Support | Number of positive instances in the test split |
| TP | True positives |
| FP | False positives |
| FN | False negatives |
| TN | True negatives |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2 * P * R / (P + R) |

Per-category results are the primary evidence for RQ4 (minority category analysis).

---

## 4. Decision Thresholds

The classifier outputs sigmoid probabilities per category.
A decision threshold converts probabilities to binary predictions.

### Threshold Tuning Protocol
- Default threshold: 0.5
- Threshold sweep values: 0.3 / 0.4 / 0.5 / 0.6 / 0.7
- Selection criterion: Macro F1 on the VALIDATION split
- Selected threshold applied ONCE to TEST split
- Test split threshold must NOT be optimised using test labels

### Category-Specific Thresholds
- Evaluated only after global threshold analysis
- Same validation-only protocol applies

---

## 5. Split Protocol

| Split | Purpose | Constraints |
| :--- | :--- | :--- |
| Train | Parameter optimisation | Never used for evaluation reporting |
| Validation | Threshold/model/HP selection | Never used for final metric reporting |
| Test | Final evaluation only | Frozen; accessed once per experiment |

The split is DOCUMENT-DISJOINT.
No document appears in more than one split.

---

## 6. What Is NOT Used

| Metric / Practice | Reason Excluded |
| :--- | :--- |
| Accuracy | Misleading under class imbalance in multi-label settings |
| Test-set threshold tuning | Would constitute data leakage |
| Cross-validation over documents | Not feasible given limited NDA document count |

---

## 7. Reporting Standards

- All metrics reported to 4 decimal places in machine-readable output
- Rounded to 2 decimal places in paper tables
- Pseudo-label quality conditions always noted alongside results
- All PLACEHOLDER cells remain empty until experiments are complete

---
*Module 7 - Research Analysis and Experimental Framework*
*Created: 2026-10-03*
