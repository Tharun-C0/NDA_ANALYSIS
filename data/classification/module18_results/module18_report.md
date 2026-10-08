# Module 18: Complete 9-Experiment Classifier Benchmark Report

## 1. Executive Summary & Framework Overview

This report presents the complete empirical comparison of **3 transformer classifier architectures** across **3 loss functions** (9 total experiments) on the official **Module 16 NDA Multi-Label Clause Benchmark**.

### Experimental Scope & Protocol
- **Benchmark Split**: Module 16 NDA Benchmark (20 disjoint documents, 717 clauses total; Train=478, Val=133, Test=106).
- **Target Categories**: 14 multi-label legal categories.
- **Architectures**: Legal-RoBERTa (`saibo/legal-roberta-base`), Legal-BERT (`nlpaueb/legal-bert-base-uncased`), DeBERTa-v3 (`microsoft/deberta-v3-base`).
- **Loss Functions**: Binary Cross-Entropy (`bce`), Multi-Label Focal Loss (`focal`, $\gamma=2.0, \alpha=0.25$), Class-Weighted BCE (`class_weighted_bce`).
- **Primary Metric**: **TEST Macro F1** across all 14 categories.

## 2. Full 9-Experiment Results Table (Ranked by TEST Macro F1)

| Rank | Exp ID | Model Architecture | Loss Function | Test Macro F1 | Minority Macro F1 | MCC | Weighted F1 | Micro F1 | Hamming Loss | Best Epoch | Val Macro F1 |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `EXP-03` | `saibo/legal-roberta-base` | `class_weighted_bce` | **`0.3675`** | `0.2806` | `0.3455` | `0.4852` | `0.4416` | `0.1967` | 3 | `0.3875` |
| **2** | `EXP-06` | `nlpaueb/legal-bert-base-uncased` | `class_weighted_bce` | **`0.3047`** | `0.2288` | `0.2758` | `0.4510` | `0.3636` | `0.2648` | 3 | `0.3264` |
| **3** | `EXP-09` | `microsoft/deberta-v3-base` | `class_weighted_bce` | **`0.2739`** | `0.1866` | `0.2345` | `0.4378` | `0.3406` | `0.3012` | 4 | `0.3037` |
| **4** | `EXP-01` | `saibo/legal-roberta-base` | `bce` | **`0.0492`** | `0.0000` | `0.0345` | `0.2072` | `0.2959` | `0.0893` | 3 | `0.0499` |
| **5** | `EXP-07` | `microsoft/deberta-v3-base` | `bce` | **`0.0415`** | `0.0000` | `0.0213` | `0.2006` | `0.2815` | `0.0881` | 5 | `0.0451` |
| **6** | `EXP-04` | `nlpaueb/legal-bert-base-uncased` | `bce` | **`0.0385`** | `0.0000` | `0.0261` | `0.1858` | `0.2394` | `0.0866` | 3 | `0.0446` |
| **7** | `EXP-02` | `saibo/legal-roberta-base` | `focal` | **`0.0330`** | `0.0000` | `0.0223` | `0.1592` | `0.1973` | `0.0887` | 3 | `0.0392` |
| **8** | `EXP-05` | `nlpaueb/legal-bert-base-uncased` | `focal` | **`0.0000`** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0965` | 3 | `0.0000` |
| **9** | `EXP-08` | `microsoft/deberta-v3-base` | `focal` | **`0.0000`** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0965` | 1 | `0.0000` |

## 3. Detailed Comparative Findings

### A. Best Overall Configuration
- **Winner**: **`EXP-03`** (`saibo/legal-roberta-base` + `class_weighted_bce`)
- **Test Macro F1**: **`0.3675`** (Highest across all 9 experiments)
- **Minority Macro F1**: **`0.2806`** (Highest minority category recall/F1)
- **MCC**: **`0.3455`**

### B. Model Architecture Comparison
1. **`saibo/legal-roberta-base`** (Legal-RoBERTa): **Best performing architecture** overall.
   - With Class-Weighted BCE: **0.3675** Test Macro F1.
   - Consistently outperformed Legal-BERT and DeBERTa-v3 across corresponding loss functions.
2. **`nlpaueb/legal-bert-base-uncased`** (Legal-BERT): **Second best architecture**.
   - With Class-Weighted BCE: **0.3047** Test Macro F1.
3. **`microsoft/deberta-v3-base`** (DeBERTa-v3): **Third place**.
   - With Class-Weighted BCE: **0.2739** Test Macro F1.

### C. Loss Function Comparison
1. **Class-Weighted BCE (`class_weighted_bce`)**: **Decisive Winner** across all 3 architectures.
   - Achieved 0.3675 (Legal-RoBERTa), 0.3047 (Legal-BERT), and 0.2739 (DeBERTa-v3).
   - Effectively combats heavy multi-label class imbalance by scaling positive loss terms by inverse class frequency.
2. **Standard BCE (`bce`)**: **Severely limited** by class imbalance.
   - Test Macro F1 between 0.0385 and 0.0492 across models.
   - Completely failed on minority categories (Minority Macro F1 = 0.0000).
3. **Multi-Label Focal Loss (`focal`)**: **Worst performing loss formulation** under standard 0.5 decision threshold.
   - Test Macro F1 between 0.0000 and 0.0330.
   - Probability attenuation from $(1-p_t)^2$ prevents logits from crossing the default 0.5 classification threshold without explicit probability threshold tuning.

## 4. Artifact Manifest

- **Complete 9-Experiment Comparison CSV**: `data/classification/module18_results/complete_9_experiment_comparison.csv`
- **Summary Experiment Results**: `data/classification/module18_results/experiment_results.csv`
- **Checkpoints Directory**: `data/classification/module18_results/checkpoints/` (EXP-01 through EXP-09)