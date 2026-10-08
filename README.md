# NDA Analysis & Clause Classification

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14.1%2Bcu132-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 13.2](https://img.shields.io/badge/CUDA-13.2-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A two-stage architecture for Non-Disclosure Agreement (NDA) contract analysis combining LLM-based clause extraction/segmentation and deep transformer-based multi-label clause classification across 14 standardized legal categories.

---

## Executive Summary

Non-Disclosure Agreements (NDAs) are among the most common legal contracts, yet manual review is time-consuming and error-prone. This repository implements an end-to-end NDA intelligence system:

1. **Stage 1 (Segmentation & Extraction)**: Extracts text from NDA PDF documents and segments text into clause-level legal units.
2. **Stage 2 (Multi-Label Classification)**: Classifies clauses across **14 approved legal categories** using domain-adapted and disentangled-attention transformer models (`Legal-RoBERTa`, `Legal-BERT`, `DeBERTa-v3`) trained with Class-Weighted Binary Cross-Entropy (BCE) loss on 5,344 dataset clauses.

---

## 14-Label Taxonomy

All legal clauses are categorized under the following 14 standardized multi-label categories:

| Index | Legal Category Name | Target Column |
| :---: | :--- | :--- |
| `01` | **Party Identification** | `label_party_identification` |
| `02` | **Purpose** | `label_purpose` |
| `03` | **NDA Type** | `label_nda_type` |
| `04` | **Definition of Confidential Information** | `label_definition_of_confidential_information` |
| `05` | **Confidentiality Obligations** | `label_confidentiality_obligations` |
| `06` | **Authorized Disclosure** | `label_authorized_disclosure` |
| `07` | **Non-Confidential Information** | `label_non-confidential_information` |
| `08` | **Liability for Damages** | `label_liability_for_damages` |
| `09` | **Competition Rights** | `label_competition_rights` |
| `10` | **Term and Termination** | `label_term_and_termination` |
| `11` | **Intellectual Property** | `label_intellectual_property` |
| `12` | **Employees** | `label_employees` |
| `13` | **Governing Law and Jurisdiction** | `label_governing_law_and_jurisdiction` |
| `14` | **Additional Information** | `label_additional_information` |

---

## Repository Structure

```text
NDA/
├── app/                        # Web GUI and API services
├── configs/                    # Yaml configuration files and loaders
│   └── config.yaml             # Central configuration
├── data/                       # Benchmark splits & dataset manifests
│   ├── classification/         # Genuine benchmark train/val/test splits
│   └── processed/              # Extracted text & clause segmentations
├── evaluation/                 # Metrics & statistical test utilities
├── experiments/                # Controlled experiment outputs & checkpoints
│   ├── EXP-03_class_weighted/  # EXP-03 Baseline (Legal-RoBERTa)
│   ├── EXP-10_synthetic_5000/  # EXP-10 Combined training (5,344 clauses)
│   ├── EXP-11_early_stopping/ # EXP-11 Controlled early stopping
│   └── EXP-12_model_comparison/# EXP-12 Transformer Architecture Comparison
├── external_data/              # External raw datasets (Kleister-NDA)
├── models/                     # Custom model wrappers & export utilities
├── notebooks/                  # Exploratory data analysis notebooks
├── preprocessing/              # PDF text extraction & clause segmentation
├── reports/                    # Module statistics, CSV scorecards & figures
├── scripts/                    # Training, evaluation & active learning scripts
│   ├── train_exp03_baseline.py
│   ├── train_exp10_synthetic_5000.py
│   ├── train_exp11_early_stopping.py
│   └── train_exp12_model_comparison.py
├── nda_clause_dataset_5000.csv  #  NDA clause dataset (5,000 clauses)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Dataset Overview

The benchmark uses fixed, document-level non-overlapping splits to prevent data leakage across clauses from the same agreement:

- **Combined Training Set**: `5,344` clauses (344 genuine + 5,000 synthetic)
- **Validation Set**: `137` genuine clauses (untouched)
- **Test Set**: `236` genuine clauses (untouched)

---

## Benchmark Model Comparison (EXP-12)

All transformer models are evaluated under identical training conditions: **Class-Weighted BCE loss**, batch size `16`, learning rate `2e-5`, seed `42`, sequence length `256`, GPU mixed-precision (`bfloat16`/`float32`), max `10` epochs with Early Stopping patience `2` (monitoring Validation Macro F1).

Optimal classification decision thresholds are selected **exclusively via validation Macro F1 sweep (0.30 to 0.70)** and applied once to the frozen test set:

| Model | HuggingFace Model ID | Best Epoch | Best Val Macro F1 | Optimal Threshold | Test Macro F1 | Test Micro F1 | Test Weighted F1 | Minority F1 | Hamming Loss | MCC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legal-RoBERTa** | `saibo/legal-roberta-base` | 2 | `0.5589` | `0.50` | **`0.4457`** | `0.5389` | `0.5366` | **`0.3574`** | `0.1077` | `0.3884` |
| **Legal-BERT** | `nlpaueb/legal-bert-base-uncased` | 3 | `0.5545` | `0.55` | **`0.4443`** | **`0.5597`** | **`0.5537`** | `0.3382` | **`0.1038`** | `0.3938` |
| **DeBERTa-v3** | `microsoft/deberta-v3-base` | 5 | `0.5588` | `0.60` | **`0.4404`** | `0.5252` | `0.5382` | `0.3377` | `0.1111` | **`0.3954`** |

---

### Per-Category 14-Label F1 Score Comparison

| Category | Legal-RoBERTa | Legal-BERT | DeBERTa-v3 | Best Architecture |
| :--- | :---: | :---: | :---: | :---: |
| Party Identification | `0.4375` | `0.4419` | `0.3908` | **Legal-BERT** |
| Purpose | `0.1905` | `0.0000` | `0.1905` | **Legal-RoBERTa / DeBERTa-v3** |
| NDA Type | `0.0000` | `0.0000` | `0.0000` | Baseline Limit |
| Definition of Confidential Information | `0.3871` | `0.4737` | `0.4242` | **Legal-BERT** |
| Confidentiality Obligations | `0.4762` | `0.5614` | `0.5385` | **Legal-BERT** |
| Authorized Disclosure | `0.6250` | `0.6000` | `0.5625` | **Legal-RoBERTa** |
| Non-Confidential Information | `0.0000` | `0.0000` | `0.0000` | Baseline Limit |
| Liability for Damages | `0.6286` | `0.6000` | `0.4179` | **Legal-RoBERTa** |
| Competition Rights | `0.5660` | `0.5882` | `0.5357` | **Legal-BERT** |
| Term and Termination | `0.5714` | `0.6076` | **`0.6197`** | **DeBERTa-v3** |
| Intellectual Property | `0.6897` | `0.6429` | **`0.7143`** | **DeBERTa-v3** |
| Employees | `0.5152` | `0.4941` | **`0.5455`** | **DeBERTa-v3** |
| Governing Law and Jurisdiction | `0.5455` | `0.5714` | **`0.6000`** | **DeBERTa-v3** |
| Additional Information | `0.6070` | `0.6394` | `0.6256` | **Legal-BERT** |

> **Note on Ultra-Rare Categories**: `NDA Type` (test support = 1 clause) and `Non-Confidential Information` (test support = 7 clauses) exhibit 0.0000 F1 due to extreme support sparsity in the frozen 236-clause benchmark test set under global decision thresholding (0.50–0.60).

---

## Installation & Quick Start

### 1. Environment Setup

Clone the repository and install requirements:

```bash
git clone https://github.com/Tharun-C0/NDA_ANALYSIS.git
cd NDA_ANALYSIS

python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Verify Data & Setup

Run dataset pre-flight validation:

```bash
python scratch/validate_data.py
```

### 3. Run Controlled Experiments

#### Run EXP-10 (Combined 5,344 Training Clauses):
```bash
python scripts/train_exp10_synthetic_5000.py
```

#### Run EXP-11 (Early Stopping):
```bash
python scripts/train_exp11_early_stopping.py
```

#### Run EXP-12 (Transformer Model Architecture Comparison):
```bash
python scripts/train_exp12_model_comparison.py
```

---

## Methodology Highlights

1. **Document-Level Split Integrity**: All clauses from a single NDA remain in the same split to avoid data leakage.
2. **Class-Weighted BCE Loss**: Positive loss weights are computed exclusively from training clauses to handle severe class imbalance across minority categories.
3. **Validation-Driven Calibration**: Decision thresholds are tuned strictly on validation Macro F1 before evaluating on the untouched test set.
4. **Mixed-Precision Training**: CUDA mixed precision with `bfloat16` and gradient norm clipping (`max_norm=1.0`) ensures numerical stability on RTX GPUs.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
