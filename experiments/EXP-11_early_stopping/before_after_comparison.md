# EXP-11: Early Stopping Experiment Comparison Report

## Executive Summary

This report documents **EXP-11**, evaluating the impact of **Early Stopping** (Patience = 2 on Validation Macro F1) when training `saibo/legal-roberta-base` with `Class-Weighted BCE` on the 5,344 combined dataset (344 benchmark train + 5,000 synthetic clauses). Best checkpoint from Epoch 4 was restored before test evaluation.

### Key Parameters across Experiments
- **EXP-03**: Benchmark Train (344 rows) | 5 Epochs Fixed | Threshold 0.60
- **EXP-10**: Combined Train (5,344 rows) | 10 Epochs Fixed | Threshold 0.45
- **EXP-11**: Combined Train (5,344 rows) | Early Stopped at Ep 6 (Best Ep 4) | Threshold 0.40

---

## Overall 3-Experiment Metric Comparison Table

| Metric | EXP-03 Baseline | EXP-10 (Fixed 10 Ep) | EXP-11 (Early Stopped) | EXP-11 vs EXP-03 | EXP-11 vs EXP-10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Macro F1** | `0.4295` | `0.4643` | **`0.4832`** | `+0.0537` (`+12.50%`) | `+0.0189` (`+4.07%`) |
| **Minority F1** | `0.3853` | `0.3769` | **`0.4432`** | `+0.0579` | `+0.0663` |
| **Micro F1** | `0.4940` | `0.5543` | **`0.5475`** | `+0.0535` | `-0.0068` |
| **Weighted F1** | `0.5151` | `0.5568` | **`0.5532`** | `+0.0381` | `-0.0036` |
| **Hamming Loss** | `0.1283` | `0.1081` | **`0.0965`** | `-0.0318` | `-0.0116` |
| **MCC** | `0.3885` | `0.4118` | **`0.4423`** | `+0.0538` | `+0.0305` |

---

## Per-Category 14-Label F1 Comparison

| Category | Baseline EXP-03 F1 | EXP-10 F1 | EXP-11 F1 | EXP-11 vs EXP-03 | EXP-11 vs EXP-10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Party Identification | `0.4318` | `0.4950` | **`0.3291`** | `-0.1027` | `-0.1659` |
| Purpose | `0.0741` | `0.1667` | **`0.1176`** | `+0.0435` | `-0.0491` |
| NDA Type | `0.2857` | `0.0000` | **`0.5000`** | `+0.2143` | `+0.5000` |
| Definition of Confidential Information | `0.4211` | `0.4242` | **`0.4324`** | `+0.0113` | `+0.0082` |
| Confidentiality Obligations | `0.4444` | `0.5000` | **`0.4667`** | `+0.0223` | `-0.0333` |
| Authorized Disclosure | `0.3404` | `0.6286` | **`0.6154`** | `+0.2750` | `-0.0132` |
| Non-Confidential Information | `0.3200` | `0.1667` | **`0.0000`** | `-0.3200` | `-0.1667` |
| Liability for Damages | `0.5957` | `0.6667` | **`0.7000`** | `+0.1043` | `+0.0333` |
| Competition Rights | `0.4865` | `0.5455` | **`0.6667`** | `+0.1802` | `+0.1212` |
| Term and Termination | `0.4524` | `0.5714` | **`0.5965`** | `+0.1441` | `+0.0251` |
| Intellectual Property | `0.5714` | `0.6667` | **`0.6400`** | `+0.0686` | `-0.0267` |
| Employees | `0.4571` | `0.5143` | **`0.5333`** | `+0.0762` | `+0.0190` |
| Governing Law and Jurisdiction | `0.4828` | `0.5217` | **`0.5217`** | `+0.0389` | `+0.0000` |
| Additional Information | `0.6494` | `0.6332` | **`0.6449`** | `-0.0045` | `+0.0117` |

---

## Key Findings

- **Best Model Checkpoint**: Restored from **Epoch 4**.
- **Early Stopping Trigger**: Stopped training at **Epoch 6**.
- **Selected Threshold**: **`0.40`** (selected via Validation Macro F1 sweep).

## Reproducibility Command

To execute EXP-11 training and evaluation on your RTX 5060 Ti GPU, run:

```powershell
python scripts/train_exp11_early_stopping.py
```
