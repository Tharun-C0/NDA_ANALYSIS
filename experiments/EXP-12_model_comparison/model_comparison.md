# EXP-12: Transformer Architecture Model Comparison Report

## Executive Summary

This experiment evaluates **3 transformer architectures** (`Legal-RoBERTa`, `Legal-BERT`, and `DeBERTa-v3`) under identical training parameters (Class-Weighted BCE, 5,344 combined clauses, max 10 epochs with Early Stopping patience=2, learning rate 2e-5, seed 42) on the frozen NDA clause benchmark.

### Benchmark Dataset Parameters
- **Combined Training Set**: 5,344 clauses (344 genuine + 5,000 synthetic)
- **Validation Set**: 137 clauses (untouched)
- **Test Set**: 236 clauses (untouched)

---

## Overall Model Comparison Table

| Model | Best Epoch | Best Val Macro F1 | Threshold | Test Macro F1 | Test Micro F1 | Test Weighted F1 | Minority F1 | Hamming Loss | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legal-RoBERTa** | 2 | `0.5589` | `0.50` | **`0.4457`** | `0.5389` | `0.5366` | `0.3574` | `0.1077` | `0.3884` |
| **Legal-BERT** | 3 | `0.5545` | `0.55` | **`0.4443`** | `0.5597` | `0.5537` | `0.3382` | `0.1038` | `0.3938` |
| **DeBERTa-v3** | 5 | `0.5588` | `0.60` | **`0.4404`** | `0.5252` | `0.5382` | `0.3377` | `0.1111` | `0.3954` |

---

## Per-Category 14-Label F1 Comparison

| Category | Legal-RoBERTa | Legal-BERT | DeBERTa-v3 | Best Architecture |
| :--- | :---: | :---: | :---: | :---: |
| Party Identification | `0.4375` | `0.4419` | `0.3908` | **`Legal-BERT`** |
| Purpose | `0.1905` | `0.0000` | `0.1905` | **`Legal-RoBERTa`** |
| NDA Type | `0.0000` | `0.0000` | `0.0000` | **`Legal-RoBERTa`** |
| Definition of Confidential Information | `0.3871` | `0.4737` | `0.4242` | **`Legal-BERT`** |
| Confidentiality Obligations | `0.4762` | `0.5614` | `0.5385` | **`Legal-BERT`** |
| Authorized Disclosure | `0.6250` | `0.6000` | `0.5625` | **`Legal-RoBERTa`** |
| Non-Confidential Information | `0.0000` | `0.0000` | `0.0000` | **`Legal-RoBERTa`** |
| Liability for Damages | `0.6286` | `0.6000` | `0.4179` | **`Legal-RoBERTa`** |
| Competition Rights | `0.5660` | `0.5882` | `0.5357` | **`Legal-BERT`** |
| Term and Termination | `0.5714` | `0.6076` | `0.6197` | **`DeBERTa-v3`** |
| Intellectual Property | `0.6897` | `0.6429` | `0.7143` | **`DeBERTa-v3`** |
| Employees | `0.5152` | `0.4941` | `0.5455` | **`DeBERTa-v3`** |
| Governing Law and Jurisdiction | `0.5455` | `0.5714` | `0.6000` | **`DeBERTa-v3`** |
| Additional Information | `0.6070` | `0.6394` | `0.6256` | **`Legal-BERT`** |

---

## Key Findings

- **Winning Architecture**: **`Legal-RoBERTa`** achieved the highest Test Macro F1 (**`0.4457`**) at decision threshold `0.50`.
- **Generalization & Calibration**: Comparative evaluation across domain-adapted (Legal-RoBERTa, Legal-BERT) vs disentangled attention (DeBERTa-v3) models on multi-label legal text.

## Reproducibility Command

To reproduce EXP-12 from scratch on your RTX 5060 Ti GPU, run:

```powershell
python scripts/train_exp12_model_comparison.py
```
