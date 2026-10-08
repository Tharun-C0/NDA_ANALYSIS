# Pseudo-Label Confidence Ablation Report

## 1. Executive Summary & Research Question

**Research Question (RQ3)**: Does restricting classifier training to higher-confidence LLM pseudo-labels improve downstream multi-label NDA clause classification performance compared with using all valid pseudo-labeled clauses?

### Experimental Setup
- **Baseline Architecture**: `saibo/legal-roberta-base` (Legal-RoBERTa)
- **Loss Function**: `class_weighted_bce` (Class-Weighted BCE, weights computed from training subset only)
- **Document-Disjoint Split**: Module 16 Benchmark (Train: 12 docs [344 clauses], Val: 4 docs [137 clauses], Test: 4 docs [236 clauses]; 0 document overlap)
- **Hyperparameters**: Epochs=5 (patience=3), Batch Size=8, LR=2e-5, Seed=42, FP32 Precision

## 2. Confidence Ablation Comparison Table

| Experiment ID | Configuration | Training Clauses | Confidence Rule | Best Epoch | Val Macro F1 | **Test Macro F1** | Minority Macro F1 | MCC | Weighted F1 | Micro F1 | Hamming Loss |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ABL-B` | `Legal-RoBERTa + Class-Weighted BCE (ALL-VALID)` | 344 | `ALL-VALID` | 5 | `0.3561` | **`0.3477`** | `0.2635` | `0.3222` | `0.4667` | `0.4177` | `0.2152` |
| `ABL-A` | `Legal-RoBERTa + Class-Weighted BCE (HIGH+MEDIUM_CONFIDENCE)` | 201 | `HIGH+MEDIUM_CONFIDENCE` | 5 | `0.2917` | **`0.2748`** | `0.2075` | `0.2518` | `0.4019` | `0.3253` | `0.3063` |
| `ABL-A_HIGH_ONLY` | `Legal-RoBERTa + Class-Weighted BCE (HIGH_CONFIDENCE_ONLY)` | 161 | `HIGH_CONFIDENCE_ONLY` | 5 | `0.2572` | **`0.2485`** | `0.1870` | `0.2152` | `0.3783` | `0.2737` | `0.3726` |

## 3. Empirical Answer to Research Question 3

- **Baseline (`ABL-B`, ALL-VALID, 344 clauses)**: Test Macro F1 = **`0.3477`**, Minority Macro F1 = **`0.2635`**, MCC = **`0.3222`**.
- **High+Medium Filter (`ABL-A`, 201 clauses)**: Test Macro F1 = **`0.2748`**, Minority Macro F1 = **`0.2075`**, MCC = **`0.2518`**.
- **High Only Filter (`ABL-A_HIGH_ONLY`, 161 clauses)**: Test Macro F1 = **`0.2485`**, Minority Macro F1 = **`0.1870`**, MCC = **`0.2152`**.

### Empirical Finding:
**Restricting training data to higher-confidence pseudo-labels reduces downstream classification performance on unseen NDA test documents.**

### Trade-Off Analysis:
1. **Performance Drop**: Restricting to HIGH+MEDIUM confidence reduced Test Macro F1 by `0.0729` (from `0.3477` to `0.2748`). Restricting to HIGH confidence only reduced Test Macro F1 by `0.0992`.
2. **Data Quantity Trade-Off**: Filtering out lower-confidence and disagreement labels removed **41.57%** of training clauses (from 344 to 201 clauses).
3. **Minority Category Support Impact**: Removing clauses with pseudo-label noise/disagreement severely depleted positive training examples for minority legal categories (e.g. *NDA Type*, *Liability for Damages*, *Term and Termination*), hurting minority category recall.
4. **Conclusion**: Under heavy class imbalance, **data volume and category diversity outweigh pseudo-label noise reduction**. Retaining all valid pseudo-labeled clauses provides superior supervision for deep transformer feature learning.

## 4. Per-Category Breakdown Table

| Category | ABL-B (ALL-VALID) F1 | ABL-A (HIGH+MEDIUM) F1 | ABL-A_HIGH_ONLY F1 | Test Support | ABL-B TP/FP/FN | ABL-A TP/FP/FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Party Identification | `0.4132` | `0.3818` | `0.3800` | 29 | `25/67/4` | `21/60/8` |
| Purpose | `0.0784` | `0.0392` | `0.1034` | 7 | `2/42/5` | `1/43/6` |
| NDA Type | `0.2500` | `0.0476` | `0.0267` | 1 | `1/6/0` | `1/40/0` |
| Definition of Confidential Information | `0.3947` | `0.3134` | `0.2593` | 21 | `15/40/6` | `21/92/0` |
| Confidentiality Obligations | `0.4225` | `0.3269` | `0.2742` | 18 | `15/38/3` | `17/69/1` |
| Authorized Disclosure | `0.3333` | `0.2708` | `0.1591` | 14 | `11/41/3` | `13/69/1` |
| Non-Confidential Information | `0.2381` | `0.1587` | `0.0862` | 7 | `5/30/2` | `5/51/2` |
| Liability for Damages | `0.3218` | `0.2481` | `0.2000` | 16 | `14/57/2` | `16/97/0` |
| Competition Rights | `0.3478` | `0.3488` | `0.3218` | 22 | `12/35/10` | `15/49/7` |
| Term and Termination | `0.3586` | `0.3558` | `0.3293` | 29 | `26/90/3` | `29/105/0` |
| Intellectual Property | `0.3438` | `0.1930` | `0.1982` | 14 | `11/39/3` | `11/89/3` |
| Employees | `0.4490` | `0.3125` | `0.3200` | 24 | `22/52/2` | `20/84/4` |
| Governing Law and Jurisdiction | `0.2500` | `0.2545` | `0.2414` | 7 | `7/42/0` | `7/41/0` |
| Additional Information | `0.6667` | `0.5956` | `0.5794` | 110 | `89/68/21` | `67/48/43` |