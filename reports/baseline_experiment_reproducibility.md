# Module 6 — Baseline Experiment Reproducibility Log & Configuration

> [!NOTE]
> This document logs the exact environment, data specifications, tokenizer parameters, and training hyperparameter defaults established in **Module 6** to ensure 100% reproducible execution when training is initiated.

---

## 1. Dataset Specifications & Lineage

- **Source File**: `data/annotations/llm_pseudo_labels.csv`
- **Total V3 Segmentation Clauses**: 717
- **Usable Pseudo-Label Records**: 203 clauses (excluding 49 `API_ERROR` and 1 `API_QUOTA_EXHAUSTED` records)
- **Derived Classification Files**:
  - `data/classification/classifier_dataset_all_valid.csv` (203 clauses)
  - `data/classification/classifier_dataset_high_confidence.csv` (38 clauses)
  - `data/classification/classifier_dataset_train.csv` (109 clauses)
  - `data/classification/classifier_dataset_val.csv` (49 clauses)
  - `data/classification/classifier_dataset_test.csv` (45 clauses)
- **Target Vocabulary**: 14 multi-hot approved categories

---

## 2. Document-Disjoint Split Statistics

- **Random Seed**: `42`
- **Train Split**: 109 clauses across 3 documents (`0a42e159`, `0859334b`, `266929af`) — 13/14 categories represented
- **Validation Split**: 49 clauses across 2 documents (`0b59dfc4`, `293f5937`) — 12/14 categories represented
- **Test Split**: 45 clauses across 2 documents (`2268c5d1`, `247166e0`) — 1/14 categories represented
- **Document Overlap**: **0 documents** (Strictly document-disjoint, zero leakage)

---

## 3. Model & Tokenizer Configurations

### Baseline Model: Legal-RoBERTa
- **Model Identifier**: `saibo/legal-roberta-base`
- **Architecture**: `RobertaForSequenceClassification` (14 binary output logits)
- **Tokenizer**: `AutoTokenizer.from_pretrained('saibo/legal-roberta-base')`
- **Max Sequence Length**: `256` tokens (truncation=True, padding='max_length')

### Alternative Models
1. **Legal-BERT**: `nlpaueb/legal-bert-base-uncased` (`BertForSequenceClassification`)
2. **DeBERTa-v3**: `microsoft/deberta-v3-base` (`DebertaV2ForSequenceClassification`)
3. **General RoBERTa**: `roberta-base` (`RobertaForSequenceClassification`)

---

## 4. Default Training Hyperparameters

- **Optimizer**: AdamW (`weight_decay=0.01`, `eps=1e-8`)
- **Learning Rate**: `2e-5`
- **Batch Size**: `8`
- **Epochs**: `5`
- **Warmup Ratio**: `0.1` (Linear scheduler)
- **Supported Loss Functions**:
  1. `bce`: Standard `BCEWithLogitsLoss`
  2. `focal`: Multi-Label Focal Loss ($\gamma=2.0$, $\alpha=0.25$)
  3. `class_weighted_bce`: Class-Weighted BCE ($w_c = N_{neg} / N_{pos}$)

---

## 5. Verification Log

- **Environment**: Python 3.14.6, PyTorch 2.12.1+cpu, Transformers 5.12.1, Scikit-Learn 1.9.0
- **Data Integrity Validation**: Passed 8/8 checks (`scripts/validate_classification_splits.py`)
- **Forward Pass Smoke Test**: Passed 2-clause forward pass & loss computation (`scripts/train_multilabel_classifier.py --smoke-test`)
- **Evaluation Pipeline Smoke Test**: Passed metric computation (`scripts/evaluate_multilabel_classifier.py --smoke-test`)
