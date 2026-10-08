# Module 14: Comprehensive Dataset Audit Report

## Executive Summary

This audit documents the state of the NDA multi-label clause classification dataset following Module 13 targeted category discovery.

- **Total Operational Rows**: 294
- **Valid Pseudo-Labeled Clauses**: 237 (80.61%)
- **Quarantined API Error / Quota Rows**: 57 (19.39%)
- **Represented Documents**: 11 distinct NDA documents
- **Unseen Queue Documents**: 9 distinct documents (423 pending clauses)
- **Current Train / Val / Test Split**: Train = 109 clauses (3 docs), Val = 49 clauses (2 docs), Test = 79 clauses (6 docs)

---

## 1. Dataset Breakdown & Provenance Audit

| Data Bucket | Row Count | Percentage | Operational Status | Provenance Metadata |
| :--- | :---: | :---: | :--- | :--- |
| **Valid Pseudo-Labels** | 237 | 80.61% | Approved for Pipeline Testing | `label_source = "llm_pseudo_label"` |
| **Quarantined API Error / Quota** | 57 | 19.39% | Quarantined / Excluded | Flagged as `API_ERROR` or `API_QUOTA_EXHAUSTED` |
| **Total Operational Rows** | 294 | 100.0% | Audit Complete | Fully traceable in `llm_pseudo_labels.csv` |

---

## 2. Confidence Tier & Model Agreement Distribution

- **HIGH_CONFIDENCE (3/3 Model Consensus)**: 195 clauses (82.28%)
- **MEDIUM_CONFIDENCE (2/3 Model Consensus)**: 34 clauses (14.35%)
- **LOW_CONFIDENCE**: 4 clauses (1.69%)
- **DISAGREEMENT**: 4 clauses (1.69%)

### Model Pairwise Agreement Score Distribution
- **Agreement = 1.0 (Full Consensus)**: 195 clauses
- **Agreement = 0.67 (2/3 Agreement)**: 38 clauses
- **Agreement = 0.33 (1/3 Agreement)**: 4 clauses

---

## 3. Document-Level Representation & Category Redundancy

- **Total Represented Documents**: 11
  - `0859334b` (50 valid clauses, 13 categories present)
  - `0a42e159` (56 valid clauses, 3 categories present)
  - `0b59dfc4` (33 valid clauses, 10 categories present)
  - `2268c5d1` (25 valid clauses, 1 category present)
  - `247166e0` (20 valid clauses, 1 category present)
  - `266929af` (3 valid clauses, 2 categories present)
  - `293f5937` (16 valid clauses, 6 categories present)
  - `3504e06a` (10 valid clauses, 7 categories present)
  - `4fd432d8` (8 valid clauses, 8 categories present)
  - `5180f107` (10 valid clauses, 13 categories present)
  - `53c8f90c` (6 valid clauses, 10 categories present)

---

## 4. Underrepresented Category Bottleneck Audit

| Category Name | Valid Clause Count | Document Count | % Represented Docs (N=11) | Redundancy Status |
| :--- | :---: | :---: | :---: | :--- |
| **Liability for Damages** | 4 | **2** | 18.18% | **CRITICAL BOTTLENECK ($\le 2$ docs)** |
| **Competition Rights** | 3 | **3** | 27.27% | **LOW REDUNDANCY ($\le 3$ docs)** |
| **Intellectual Property** | 4 | **3** | 27.27% | **LOW REDUNDANCY ($\le 3$ docs)** |
| **Governing Law and Jurisdiction** | 6 | **3** | 27.27% | **LOW REDUNDANCY ($\le 3$ docs)** |
| **Term and Termination** | 5 | 4 | 36.36% | Moderate Coverage |
| **NDA Type** | 6 | 5 | 45.45% | Adequate Coverage |
| **Employees** | 7 | 5 | 45.45% | Adequate Coverage |
| **Definition of Confidential Info** | 7 | 6 | 54.55% | High Coverage |
| **Non-Confidential Information** | 10 | 6 | 54.55% | High Coverage |
| **Purpose** | 12 | 7 | 63.64% | High Coverage |
| **Authorized Disclosure** | 21 | 7 | 63.64% | High Coverage |
| **Confidentiality Obligations** | 22 | 7 | 63.64% | High Coverage |
| **Party Identification** | 24 | 7 | 63.64% | High Coverage |
| **Additional Information** | 164 | 9 | 81.82% | Dominant Class |

---

## 5. Current Classification Split Audit

- **Train Split**: 109 clauses across 3 documents (`0a42e159`, `0859334b`, `266929af`) — 13/14 categories present (Missing *Employees*).
- **Val Split**: 49 clauses across 2 documents (`0b59dfc4`, `293f5937`) — 12/14 categories present (Missing *Liability for Damages*, *Term and Termination*).
- **Test Split**: 79 clauses across 6 documents (`2268c5d1`, `247166e0`, `3504e06a`, `4fd432d8`, `5180f107`, `53c8f90c`) — 14/14 categories present.
