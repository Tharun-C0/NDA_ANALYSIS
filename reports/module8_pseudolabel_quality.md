# Module 8 - Pseudo-Label Quality & Ensemble Agreement Analysis

> **Status**: **COMPLETE**
> Empirical analysis of LLM ensemble agreement, confidence distributions, error rates, and quality tier definitions.

---

## 1. Quality Tier Definitions (from scripts/llm_clause_classifier.py)

The 3-model Gemini ensemble (gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-flash) classifies each clause and assigns a Quality Tier based on exact consensus rules:

| Quality Tier | Ensemble Agreement Score | Average Model Confidence | Interpretation / Operational Handling |
| :--- | :---: | :---: | :--- |
| **HIGH_CONFIDENCE** | agreement_score == 1.0 | average_confidence >= 0.80 | Perfect 3-model consensus with high confidence. High distant supervision quality. |
| **MEDIUM_CONFIDENCE** | agreement_score >= 0.67 | average_confidence >= 0.60 | Majority 2/3 model consensus with moderate confidence. Included in valid dataset. |
| **LOW_CONFIDENCE** | agreement_score >= 0.50 | average_confidence < 0.60 | Majority consensus with lower individual model confidence. Included in valid dataset. |
| **DISAGREEMENT** | agreement_score < 0.50 | Any | Models produced conflicting category predictions. Included in valid dataset with union/majority label. |
| **API_ERROR** | N/A | N/A | API network error or JSON parsing failure. EXCLUDED from classification benchmark. |
| **API_QUOTA_EXHAUSTED** | N/A | N/A | Free-tier rate limit triggered (~17.68h retry). EXCLUDED from classification benchmark. |

---

## 2. Empirical Quality Tier Breakdown (254 Total Rows)

| Quality Tier | Row Count | Percentage of Corpus | Valid Evaluation Status |
| :--- | :---: | :---: | :--- |
| **DISAGREEMENT** | 100 | 39.4% | VALID (Included in benchmark) |
| **LOW_CONFIDENCE** | 49 | 19.3% | VALID (Included in benchmark) |
| **API_ERROR** | 49 | 19.3% | EXCLUDED (Requires retry) |
| **HIGH_CONFIDENCE** | 38 | 15.0% | VALID (Included in benchmark & Ablation A) |
| **MEDIUM_CONFIDENCE** | 16 | 6.3% | VALID (Included in benchmark & Ablation A) |
| **API_QUOTA_EXHAUSTED** | 2 | 0.8% | EXCLUDED (Awaiting quota reset) |
| **Total** | **254** | **100.0%** | **203 Valid / 51 Excluded** |

---

## 3. Ensemble Agreement & Confidence Summary

- **Total Valid Pseudo-Labels**: 203 clauses
- **Average Model Confidence Across Valid Set**: 0.5262
- **Average Agreement Score Across Valid Set**: 0.6396
- **High Quality Ratio (High + Medium)**: 26.6% (54 clauses)
- **API Failure / Contamination Rate**: 20.1% (51 records cleanly quarantined)

---

## 4. Category Support Breakdown Across Valid Set (203 Clauses)

| Category Name | Positive Clause Count | Prevalence in Valid Set |
| :--- | :---: | :---: |
| Additional Information | 157 | 77.3% |
| Party Identification | 15 | 7.4% |
| Confidentiality Obligations | 12 | 5.9% |
| Authorized Disclosure | 12 | 5.9% |
| Purpose | 6 | 3.0% |
| Governing Law and Jurisdiction | 5 | 2.5% |
| NDA Type | 4 | 2.0% |
| Definition of Confidential Information | 3 | 1.5% |
| Non-Confidential Information | 3 | 1.5% |
| Liability for Damages | 3 | 1.5% |
| Intellectual Property | 3 | 1.5% |
| Employees | 3 | 1.5% |
| Competition Rights | 2 | 1.0% |
| Term and Termination | 1 | 0.5% |

---

## 5. Document Distribution Across Valid Set (7 Documents)

| Document ID Prefix | Valid Clauses | Percent of Valid Set |
| :--- | :---: | :---: |
| 0a42e159b33e | 56 | 27.6% |
| 0859334b3622 | 50 | 24.6% |
| 0b59dfc4ce9b | 33 | 16.3% |
| 2268c5d1120f | 25 | 12.3% |
| 247166e02454 | 20 | 9.9% |
| 293f59373f6a | 16 | 7.9% |
| 266929af5f5b | 3 | 1.5% |

---
*Module 8 - Pseudo-Label Quality Report*
*Created: 2026-10-03*
