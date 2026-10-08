# Module 5: Multi-Model LLM Pseudo-Label Final Status Report

> [!IMPORTANT]
> **DISCLAIMER**: The annotations generated in this module are **LLM pseudo-labels** produced by a 3-model ensemble (`gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.1-flash-lite`). They represent automated pre-annotations intended to accelerate downstream annotation and active learning workflows, **NOT human ground-truth labels**.

---

## 1. Executive Summary & Processing Counts

- **Total V3 Clauses in Dataset**: `717`
- **Successfully Pseudo-Labeled (Completed Valid Records)**: `202` (28.17%)
- **Failed / API Error Records (Awaiting Retry)**: `49` (6.83%)
- **Total Processed in CSV (`data/annotations/llm_pseudo_labels.csv`)**: `251` (35.01%)
- **Unprocessed Clauses (In Resume Queue)**: `466` (64.99%)
- **Duplicate Clause IDs**: `0` (0.00%)
- **Invalid Records / Corrupted Schema**: `0` (0.00%)

---

## 2. Confidence & Quality Breakdown

Across the 251 processed records in `data/annotations/llm_pseudo_labels.csv`:

| Quality Bucket | Count | Percentage of Processed (251) | Percentage of Total (717) |
| :--- | :--- | :--- | :--- |
| **HIGH_CONFIDENCE** | 37 | 14.74% | 5.16% |
| **MEDIUM_CONFIDENCE** | 16 | 6.37% | 2.23% |
| **LOW_CONFIDENCE** | 49 | 19.52% | 6.83% |
| **DISAGREEMENT** | 100 | 39.84% | 13.95% |
| **API_ERROR** | 49 | 19.52% | 6.83% |
| **Total** | **251** | **100.00%** | **35.01%** |

- **Mean Ensemble Confidence**: `0.6085`
- **Mean Agreement Score**: `0.5133`

---

## 3. Category Distribution Across 14 Approved NDA Categories

| Category | Clause Count | Distribution % (of 288 assigned labels) |
| :--- | :--- | :--- |
| **Additional Information** | 182 | 63.19% |
| **Party Identification** | 33 | 11.46% |
| **Confidentiality Obligations** | 13 | 4.51% |
| **Authorized Disclosure** | 12 | 4.17% |
| **Purpose** | 12 | 4.17% |
| **Liability for Damages** | 8 | 2.78% |
| **Governing Law and Jurisdiction** | 6 | 2.08% |
| **Competition Rights** | 6 | 2.08% |
| **Intellectual Property** | 4 | 1.39% |
| **NDA Type** | 4 | 1.39% |
| **Non-Confidential Information** | 3 | 1.04% |
| **Definition of Confidential Information** | 3 | 1.04% |
| **Employees** | 3 | 1.04% |
| **Term and Termination** | 1 | 0.35% |
| **Total Label Assignments** | **288** | **100.00%** |

---

## 4. Multi-Model Performance & Execution Verification

- **5-Clause Test Batch**: All 5 clauses completed successfully with 0 errors.
- **20-Clause Batch**: Successfully processed and saved 10 clauses incrementally before being safely stopped. All 10 clauses were appended with zero duplicates and 100% valid schema integrity.
- **API Rate-Limiting**: Handled smoothly (2 backoff pauses totaling 40s with 0 dropped clauses).
- **Queue State**: [`data/annotations/pseudolabel_resume_queue.csv`](file:///f:/NDA/NDA/data/annotations/pseudolabel_resume_queue.csv) holds the remaining **466 unprocessed clauses** + **49 API_ERROR clauses eligible for retry**, preserving all prior work.

---

## 5. Dataset Validation Summary (`scripts/validate_pseudolabels.py`)

All checks passed with 100% compliance:
1. `Check 01: File Non-Empty`: **PASS** (251 rows found)
2. `Check 02: No Duplicate Clause IDs`: **PASS** (0 duplicates)
3. `Check 03: Required Columns Present`: **PASS** (all 15 schema columns present)
4. `Check 04: Correct Label Source`: **PASS** (all rows strictly `label_source = 'llm_pseudo_label'`)
5. `Check 05: Valid JSON Format`: **PASS** (0 malformed JSON rows)
6. `Check 06: No Invented Categories`: **PASS** (strictly 14 approved categories)
7. `Check 07: Confidence in [0,1]`: **PASS** (all confidence values valid)
8. `Check 08: Agreement Score in [0,1]`: **PASS** (all agreement scores valid)
9. `Check 09: Valid Quality Buckets`: **PASS** (`API_ERROR` recognized as valid operational status; 0 invalid bucket rows)
10. `Check 10: No Missing Clause Text`: **PASS** (0 empty text rows)

---

## 6. Downstream & Review Artifacts

- **Human Verification Sample**: [`data/annotations/human_verification_sample.csv`](file:///f:/NDA/NDA/data/annotations/human_verification_sample.csv)
- **Human Review Priority**: [`data/annotations/human_review_priority.csv`](file:///f:/NDA/NDA/data/annotations/human_review_priority.csv)
- **Classifier Training**: Not executed (per instruction).
- **Original Dataset Integrity**: `data/segmentation_v3/*.json` completely untouched.
