# Module 10: Multi-Label Evaluation Framework Validation Report

## Executive Summary

This report documents the validation of the multi-label evaluation framework implemented in `scripts/evaluate_multilabel_classifier.py`. It confirms metric definitions, threshold selection rules, test isolation protocols, and reporting formats.

---

## 1. Metric Definitions & Mathematical Scope

### Primary Benchmark Metric
- **Macro F1 ($\text{F1}_{\text{macro}}$)**:
  $$\text{F1}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} \text{F1}_c = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot \text{TP}_c}{2 \cdot \text{TP}_c + \text{FP}_c + \text{FN}_c}$$
  Designated as the **PRIMARY METRIC** because it treats all 14 NDA categories equally, preventing performance on high-frequency categories from obscuring poor minority category classification.

### Secondary Metrics
- **Micro F1 ($\text{F1}_{\text{micro}}$)**: Global instance-level F1 aggregating TP, FP, FN across all categories simultaneously.
- **Weighted F1 ($\text{F1}_{\text{weighted}}$)**: Category-level F1 weighted by positive class support.
- **Hamming Loss ($\mathcal{H}$)**: Fraction of wrong binary predictions across all clauses and categories.
  $$\mathcal{H} = \frac{1}{N \cdot C} \sum_{i=1}^{N} \sum_{c=1}^{C} \mathbb{I}(\hat{y}_{i,c} \neq y_{i,c})$$
- **Multi-Label Macro MCC ($\text{MCC}_{\text{macro}}$)**: Mean per-category Matthews Correlation Coefficient.

### Per-Category Metrics & Confusion Breakdown
For each of the 14 NDA categories, the framework computes:
- **Precision ($P_c$)**: $\frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}$
- **Recall ($R_c$)**: $\frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$
- **F1-Score ($\text{F1}_c$)**: $\frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$
- **MCC ($\text{MCC}_c$)**: $\frac{\text{TP}_c \cdot \text{TN}_c - \text{FP}_c \cdot \text{FN}_c}{\sqrt{(\text{TP}_c+\text{FP}_c)(\text{TP}_c+\text{FN}_c)(\text{TN}_c+\text{FP}_c)(\text{TN}_c+\text{FN}_c)}}$
- **Full Confusion Matrix Counts**: True Positives ($\text{TP}_c$), False Positives ($\text{FP}_c$), False Negatives ($\text{FN}_c$), True Negatives ($\text{TN}_c$), Actual Positives, Predicted Positives.

---

## 2. Threshold Selection & Test Isolation Protocol

### Rules Governing Threshold Tuning
1. **Validation-Only Selection**: Sigmoid classification threshold $\tau^* \in [0.1, 0.9]$ is selected to maximize Macro F1 **EXCLUSIVELY on the Validation split** (`df_val`).
2. **Frozen Test Inference**: The optimal threshold $\tau^*$ tuned on validation data is frozen and applied directly to Test split logits ($\hat{\mathbf{y}}_{\text{test}} = \mathbb{I}(\sigma(\mathbf{z}_{\text{test}}) \ge \tau^*)$).
3. **Zero Test Contamination**: Test labels ($\mathbf{y}_{\text{test}}$) are NEVER accessed during threshold selection.
4. **No Accuracy Dominance**: Accuracy is excluded as a primary metric due to multi-label imbalance where predicting all zeros yields deceptively high accuracy.

---

## 3. Metric & Protocol Verification Checklist

| Requirement / Rule | Verification Status | Evidence / Implementation Line |
| :--- | :-: | :--- |
| Macro F1 designated as primary metric | **PASS** | `evaluate_multilabel_classifier.py` line 93 |
| Micro F1, Weighted F1, Hamming Loss calculated | **PASS** | `evaluate_multilabel_classifier.py` lines 94-103 |
| Per-category TP, FP, FN, TN computed | **PASS** | `evaluate_multilabel_classifier.py` lines 112-115 |
| Zero-division protection ($zero\_division=0$) | **PASS** | Sklearn metrics invoked with `zero_division=0` |
| Threshold selection on validation data only | **PASS** | `validate_experiment_integrity.py` validation rule |
| Frozen test threshold application | **PASS** | `train_multilabel_classifier.py` evaluation protocol |
| Accuracy excluded as primary metric | **PASS** | Standard multi-label protocol enforced |

---

## 4. Verification Verdict

The evaluation framework implemented in `scripts/evaluate_multilabel_classifier.py` is fully validated, mathematically complete, and enforces strict zero-leakage threshold selection rules.
