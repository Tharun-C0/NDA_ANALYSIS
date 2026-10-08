# Empirical Classifier Experiment Comparison Report

> [!IMPORTANT]
> **RESEARCH METHODOLOGY DISCLOSURE**: All metrics reported in this table are calculated on **LLM-generated pseudo-labels** (produced by a 3-model ensemble: Gemini 3.6 Flash, Gemini 3.5 Flash, Gemini 3.1 Flash Lite). They represent automated experimental pre-annotations and **have NOT yet been fully human-verified**.

> [!NOTE]
> **PRIMARY EVALUATION METRIC**: **Macro F1** is designated as the primary benchmark metric for multi-label clause classification due to category imbalance across the 14 NDA categories.

## 1. Measured Empirical Performance Table

| Experiment ID | Model Identifier | Loss Function | Label Mode | Status | Macro F1 | Weighted F1 | Micro F1 | Macro Precision | Macro Recall | Hamming Loss | Macro MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP-01` | `saibo/legal-roberta-base` | `bce` | `all_valid` | `DEBUG / PIPELINE VALIDATION ONLY — NOT A RESEARCH RESULT` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.5048` | `0.4286` |
| `EXP-01` | `saibo/legal-roberta-base` | `bce` | `all_valid` | `DEBUG / PIPELINE VALIDATION ONLY — NOT A RESEARCH RESULT` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.5048` | `0.4286` |

## 2. Objective Metric Interpretation Guidelines
- **Macro F1**: Unweighted arithmetic mean of per-category F1 scores across 14 categories. Evaluates performance equally on minority and majority categories.
- **Weighted F1**: Category F1 scores weighted by true positive support.
- **Micro F1**: Global instance-level F1 score over all binary decisions.
- **Hamming Loss**: Fraction of misclassified binary labels (lower indicates higher classification accuracy).
- **Macro MCC**: Multi-label Matthews Correlation Coefficient reflecting binary correlation across all categories.

> [!CAUTION]
> **EVALUATION SPLIT NOTICE**: All runs executed on temporary 7-document debugging splits are strictly labeled as `DEBUG_SMOKE_TEST`. Definitive scientific comparison will occur after additional pseudo-labeling is completed across all 717 V3 clauses.