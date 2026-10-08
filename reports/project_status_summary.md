# NDA Analysis Research Project: Status & Implementation Summary

**A Two-Stage Architecture for NDA Analysis: LLM-based Segmentation and Transformer-based Clause Classification**

---

## 1. Project Input & Output Specifications

### Input
- **Raw Documents**: PDF contracts from the Kleister-NDA corpus (`external_data/kleister-nda/`).
- **Clause Text**: Individual text segments extracted from non-disclosure agreements (min length $\ge 30$ characters).
- **Target Categories (14 Approved Legal Classes)**:
  1. Party Identification
  2. Purpose
  3. NDA Type
  4. Definition of Confidential Information
  5. Confidentiality Obligations
  6. Authorized Disclosure
  7. Non-Confidential Information
  8. Liability for Damages
  9. Competition Rights
  10. Term and Termination
  11. Intellectual Property
  12. Employees
  13. Governing Law and Jurisdiction
  14. Additional Information

### Output
- **Multi-Label Predictions**: 14-dimensional binary classification vector $[p_1, p_2, \dots, p_{14}] \in \{0, 1\}^{14}$ indicating presence of each clause category.
- **Trained Model Checkpoints**: Fine-tuned PyTorch transformer models (`data/classification/module18_results/checkpoints/`).
- **Benchmark Datasets**: Document-disjoint CSV splits (`train.csv`, `validation.csv`, `test.csv`).
- **Evaluation Reports**: Experiment performance CSVs, category-level F1 breakdown, and markdown research reports.

---

## 2. Dataset Preprocessing Pipeline

The data preprocessing workflow follows a rigorous, multi-stage pipeline:

```
[ Raw NDA PDFs ]
       │
       ▼ (PyMuPDF / Fitz)
[ Text Extraction & Normalization ]
       │
       ▼ (Rule-Based Legal Segmenter v1 -> v2 -> v3)
[ Boundary Detection & Segmentation ] (Numbered sections, legal headers, ALL CAPS)
       │
       ▼ (Filtering & Quality Control)
[ Minimum Length Guard (<30 chars) & Leakage Checks ]
       │
       ▼ (LLM Multi-Label Pseudo-Labeling via Gemini / OpenRouter)
[ 14-Category Multi-Label Annotation ]
       │
       ▼ (Confidence Filtering & Human Review Gates)
[ High-Confidence Dataset & Verification ]
       │
       ▼ (Document-Disjoint Splitting)
[ Final Module 16 Benchmark Dataset ] (20 Disjoint Documents, 717 Clauses)
```

### Key Preprocessing Steps Completed
1. **Extraction**: Text extracted from raw PDF files while preserving section breaks and character offsets.
2. **Segmentation Refinement**: Iteratively improved rule-based segmentation (`v1` $\rightarrow$ `v2` $\rightarrow$ `v3`), detecting numbered clauses (`1.1`), section headers (`ARTICLE I`), and parenthesized sub-clauses (`(a)`).
3. **Multi-Label Labeling**: Annotated 717 clauses using LLM pseudo-labeling and human validation protocols.
4. **Document-Disjoint Splitting**: Created benchmark splits such that all clauses from a single NDA document belong strictly to **one split** (Train, Validation, or Test) to eliminate data leakage:
   - **Train**: 12 Documents (478 clauses)
   - **Validation**: 4 Documents (133 clauses)
   - **Test**: 4 Documents (106 clauses)

---

## 3. How Models Are Trained

### Multi-Label Classification Framework
- **Task Formulation**: Multi-label classification with binary cross-entropy variants.
- **Input Representation**: Text tokenized with model-specific tokenizers, padded/truncated to sequence length 256.

### Experimental Matrix (3 Models $\times$ 3 Loss Functions = 9 Experiments)

| Experiment ID | Base Architecture | Loss Function | Status |
|---|---|---|---|
| **EXP-01** | `saibo/legal-roberta-base` | BCEWithLogitsLoss (`bce`) | Completed |
| **EXP-02** | `saibo/legal-roberta-base` | Multi-Label Focal Loss (`focal`) | Completed |
| **EXP-03** | `saibo/legal-roberta-base` | Class-Weighted BCE (`class_weighted_bce`) | Completed |
| **EXP-04** | `nlpaueb/legal-bert-base-uncased` | BCEWithLogitsLoss (`bce`) | Completed |
| **EXP-05** | `nlpaueb/legal-bert-base-uncased` | Multi-Label Focal Loss (`focal`) | Completed |
| **EXP-06** | `nlpaueb/legal-bert-base-uncased` | Class-Weighted BCE (`class_weighted_bce`) | Completed |
| **EXP-07** | `microsoft/deberta-v3-base` | BCEWithLogitsLoss (`bce`) | Ready for Rerun (FP32 Fixed) |
| **EXP-08** | `microsoft/deberta-v3-base` | Multi-Label Focal Loss (`focal`) | Ready for Rerun (FP32 Fixed) |
| **EXP-09** | `microsoft/deberta-v3-base` | Class-Weighted BCE (`class_weighted_bce`) | Ready for Rerun (FP32 Fixed) |

### Training Hyperparameters & Protocol
- **Precision**: Full Precision `torch.float32` (eliminates FP16 underflow/overflow in DeBERTa-v3).
- **Optimizer**: `AdamW` (learning rate $= 2\times 10^{-5}$, weight decay $= 0.01$).
- **Learning Rate Schedule**: Linear warmup over initial 10% steps followed by linear decay.
- **Batch Size**: 8 samples per GPU step.
- **Epochs & Early Stopping**: Up to 5 epochs with patience of 3 epochs monitoring **Validation Macro F1**.
- **Focal Loss Parameters**: $\gamma = 2.0$, $\alpha = 0.25$.
- **Class Weights**: Positives weighted by $\frac{N_{\text{neg}}}{N_{\text{pos}}}$ computed strictly on training set.
- **Evaluation Metrics**: Macro F1, Minority Category Macro F1, Micro F1, Weighted F1, Matthews Correlation Coefficient (MCC), and Hamming Loss.

---

## 4. Overall Implementation Plan & Progress Status

### Progress Summary: **89.5% Completed** (17 / 19 Modules Fully Done)

- [x] **Module 1: Project Foundation & Dataset Preparation**
  - Data directory layout, central config system (`configs/config.yaml`), raw annotation parsers.
- [x] **Module 1b: Dataset Acquisition & Inspection**
  - Downloaded Kleister-NDA corpus, audited document-level vs clause-level schema differences.
- [x] **Module 2: NDA Text Extraction & Clause Segmentation**
  - PDF extraction pipeline, baseline legal segmenter, review queue generation.
- [x] **Module 3: Multi-Label Annotation Protocol**
  - Defined 14 approved NDA clause categories, established annotation guidelines.
- [x] **Module 4: Zero-Shot / Few-Shot LLM Pseudo-Labeling**
  - Integrated Gemini / OpenRouter API for automated multi-label clause tagging.
- [x] **Module 5: Pseudo-Label Quality Audit**
  - Validated pseudo-label consistency and error patterns across category splits.
- [x] **Module 6: Dataset Cleaning & Feature Formatting**
  - Sanitized clause texts, built binary multi-label ground truth matrices.
- [x] **Module 7: Initial Baseline Classifier Setup**
  - Implemented initial PyTorch DataLoader, training loops, and evaluation metrics.
- [x] **Module 8: Active Learning Seed Selection**
  - Selected representative clause subsets using diversity sampling.
- [x] **Module 9: Benchmark Split & Leakage Audit**
  - Verified 0 document overlap between splits; created document-disjoint split metadata.
- [x] **Module 10: Ablation & Loss Function Design**
  - Designed Multi-Label Focal Loss and Class-Weighted BCE implementations.
- [x] **Module 11: Category Redundancy & Expansion Analysis**
  - Audited category co-occurrence and sample frequencies across minority classes.
- [x] **Module 12: Expansion & Coverage Optimization**
  - Targeted document selection to ensure all 14 categories have adequate positive samples.
- [x] **Module 13: Targeted Priority Queue Generation**
  - Prioritized under-represented categories for verification.
- [x] **Module 14: Quality Assurance & Human Review Gates**
  - Scorecard audit and human verification checks on dataset splits.
- [x] **Module 15: Final Claims Gate & Benchmark Freeze**
  - Verified benchmark stability and froze dataset specifications.
- [x] **Module 16: Construction of 717-Clause Benchmark**
  - Built final benchmark: 717 clauses from 20 disjoint documents across 14 categories (`data/classification/module16_benchmark/`).
- [x] **Module 17: Active Learning Batch Evaluation**
  - Evaluated active learning iteration impact on classifier baseline performance.
- [/] **Module 18: Controlled Classifier Experiment Matrix (In Progress)**
  - [x] Framework implementation with `--experiments` CLI filtering and dry-run preflight checks.
  - [x] `EXP-01` to `EXP-06` trained and evaluated (`saibo/legal-roberta-base` & `nlpaueb/legal-bert-base-uncased`).
  - [x] NaN loss root cause diagnosis in `microsoft/deberta-v3-base` (FP16 underflow/overflow).
  - [x] Code fix & 1-epoch diagnostic verification on `EXP-07` in FP32 precision.
  - [ ] Rerun full 5-epoch training for `EXP-07`, `EXP-08`, `EXP-09`.
- [ ] **Module 19: Comparative Analysis & Project Final Report**
  - Generate comprehensive cross-model performance tables, per-category F1 figures, and final paper artifact.

---

## 5. Next Execution Action

To complete the remaining experimental runs (`EXP-07`, `EXP-08`, and `EXP-09`), run:

```powershell
python scripts/module18_classifier_experiments.py --experiments EXP-07 EXP-08 EXP-09
```
