# Module 5B: Multi-Model LLM Pseudo-Labeling Completion Report

> [!IMPORTANT]
> **DISCLAIMER**: The annotations generated in this module are **LLM pseudo-labels** produced by a 3-model ensemble (Gemini 3.6 Flash, Gemini 3.5 Flash, Gemini 3.1 Flash Lite). They represent automated pre-annotations intended to accelerate downstream annotation and model training, **NOT human ground-truth labels**.

---

## 1. Executive Summary

- **Total V3 Clauses in Dataset**: 717
- **Successfully Pseudo-Labeled Clauses**: 338 / 717 (47.14%)
- **Agreement Count**: 236 (32.91%)
  - **HIGH_CONFIDENCE**: 164
  - **MEDIUM_CONFIDENCE**: 23
  - **LOW_CONFIDENCE**: 49
- **Genuine Disagreement Count**: 102 (14.23%)
- **API_ERROR Clauses**: 50
- **INVALID_RESPONSE Clauses**: 0
- **Duplicate Rows Count**: 0
- **Clauses Requiring Human Verification**: 201

---

## 2. Multi-Model Ensemble Performance

| Model Name | Model Slot | Successful Responses | Success Rate |
| :--- | :--- | :--- | :--- |
| `gemini-3.6-flash` | Model A | 209 / 393 | 53.18% |
| `gemini-3.5-flash` | Model B | 89 / 393 | 22.65% |
| `gemini-3.1-flash-lite` | Model C | 284 / 393 | 72.26% |

---

## 3. Confidence Statistics

- **Mean Ensemble Confidence**: 0.7188
- **Standard Deviation**: 0.3081
- **Minimum Confidence**: 0.0
- **Maximum Confidence**: 1.0

---

## 4. Category Distribution Across 14 Approved NDA Categories

| Category Name | Clause Frequency | Percentage |
| :--- | :--- | :--- |
| Party Identification | 66 | 12.94% |
| Purpose | 25 | 4.90% |
| NDA Type | 13 | 2.55% |
| Definition of Confidential Information | 23 | 4.51% |
| Confidentiality Obligations | 37 | 7.25% |
| Authorized Disclosure | 30 | 5.88% |
| Non-Confidential Information | 22 | 4.31% |
| Liability for Damages | 14 | 2.75% |
| Competition Rights | 12 | 2.35% |
| Term and Termination | 9 | 1.76% |
| Intellectual Property | 12 | 2.35% |
| Employees | 23 | 4.51% |
| Governing Law and Jurisdiction | 11 | 2.16% |
| Additional Information | 213 | 41.76% |

---

## 5. Artifacts Produced

1. **Primary Pseudo-Label Dataset**: `data/annotations/llm_pseudo_labels.csv` (393 rows)
2. **Resume Queue**: `data/annotations/pseudolabel_resume_queue.csv` (0 remaining)
3. **Category Distribution Summary**: `data/annotations/llm_pseudo_label_distribution.csv`
4. **Human Verification Sample**: `data/annotations/human_verification_sample.csv` (201 rows)
5. **Completion Report**: `reports/module5b_completion_report.md`

---
*Report generated automatically upon Module 5B completion.*
