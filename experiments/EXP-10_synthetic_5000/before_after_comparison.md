# EXP-10: Synthetic Data Augmentation Before/After Comparison Report

## Executive Summary

This report evaluates the impact of adding **5,000 synthetic NDA clauses** (combined training dataset size: **5,344 clauses**) to the `saibo/legal-roberta-base` + `Class-Weighted BCE` classifier (`EXP-10`), compared directly against the baseline experiment trained without synthetic data (`EXP-03`).

### Key Experiment Parameters
- **Baseline Model (`EXP-03`)**: Legal-RoBERTa + Class-Weighted BCE (344 train clauses, 5 epochs, threshold 0.60)
- **New Model (`EXP-10`)**: Legal-RoBERTa + Class-Weighted BCE (5,344 train clauses, 10 epochs, GPU mixed-precision)
- **Validation & Test Sets**: 100% frozen, original benchmark splits (0 synthetic data in val/test).

---

## Before / After Metric Comparison Table

| Metric | Previous EXP-03 | New EXP-10 | Absolute Change | Percentage Change |
| :--- | :---: | :---: | :---: | :---: |
| **Macro F1** | `0.4295` | **`0.4643`** | `+0.0348` | `+8.10%` |
| **Minority F1** | `0.3853` | **`0.3769`** | `-0.0084` | `-2.18%` |
| **Micro F1** | `0.4940` | **`0.5543`** | `+0.0603` | `+12.21%` |
| **Weighted F1** | `0.5151` | **`0.5568`** | `+0.0417` | `+8.10%` |
| **Hamming Loss** | `0.1283` | **`0.1081`** | `-0.0202` | `-15.74%` |
| **MCC** | `0.3885` | **`0.4118`** | `+0.0233` | `+6.00%` |

---

## Official Research Findings

**DID PERFORMANCE IMPROVE?**  
**`YES`**

**MACRO F1 IMPROVEMENT:**  
`+0.0348` (`+8.10%`)

**MINORITY F1 IMPROVEMENT:**  
`-0.0084` (`-2.18%`)

**BEST THRESHOLD:**  
`0.45` (Selected exclusively via Validation Macro F1 sweep)

**FINAL TEST RESULT:**  
Macro F1 = `0.4643`, Minority F1 = `0.3769`, MCC = `0.4118`, Hamming Loss = `0.1081`

---

## Per-Category Detailed Comparison (EXP-03 vs EXP-10)

| Category | Baseline EXP-03 F1 | New EXP-10 F1 | Absolute Change | Impact |
| :--- | :---: | :---: | :---: | :---: |
| Party Identification | `0.4318` | `0.4950` | `+0.0632` | **`IMPROVED`** |
| Purpose | `0.0741` | `0.1667` | `+0.0926` | **`IMPROVED`** |
| NDA Type | `0.2857` | `0.0000` | `-0.2857` | **`DEGRADED`** |
| Definition of Confidential Information | `0.4211` | `0.4242` | `+0.0031` | **`NEUTRAL`** |
| Confidentiality Obligations | `0.4444` | `0.5000` | `+0.0556` | **`IMPROVED`** |
| Authorized Disclosure | `0.3404` | `0.6286` | `+0.2882` | **`IMPROVED`** |
| Non-Confidential Information | `0.3200` | `0.1667` | `-0.1533` | **`DEGRADED`** |
| Liability for Damages | `0.5957` | `0.6667` | `+0.0710` | **`IMPROVED`** |
| Competition Rights | `0.4865` | `0.5455` | `+0.0590` | **`IMPROVED`** |
| Term and Termination | `0.4524` | `0.5714` | `+0.1190` | **`IMPROVED`** |
| Intellectual Property | `0.5714` | `0.6667` | `+0.0953` | **`IMPROVED`** |
| Employees | `0.4571` | `0.5143` | `+0.0572` | **`IMPROVED`** |
| Governing Law and Jurisdiction | `0.4828` | `0.5217` | `+0.0389` | **`IMPROVED`** |
| Additional Information | `0.6494` | `0.6332` | `-0.0162` | **`NEUTRAL`** |

---

## Reproducibility Statement

To reproduce EXP-10 from scratch on your RTX 5060 / 5060 Ti GPU, run the following terminal command:

```powershell
python scripts/train_exp10_synthetic_5000.py
```
