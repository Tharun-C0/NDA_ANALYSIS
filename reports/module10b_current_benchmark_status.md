# Module 10B: Current Benchmark Feasibility Check

## Executive Summary

- **Current Dataset Size**: 237 valid pseudo-labeled clauses (294 total CSV rows, excluding 57 API error/quota rows)
- **Represented Documents**: 11 distinct NDA documents
- **Current Benchmark Status**: **BLOCKED**
- **Primary Blocker**: Category coverage imbalance across document-disjoint splits. Under standard ratio-based document splitting, the Test split is missing **7 of 14 categories**. Under heuristic category maximization, Test or Train remains missing critical categories such as *Liability for Damages*, *Competition Rights*, or *Employees*.

---

## 1. Dataset & Document Representation Overview

- **Total Operational CSV Rows**: 294
- **Valid Pseudo-Labeled Clauses**: 237 (80.61%)
- **Quarantined API Error / Quota Rows**: 57 (19.39%)
- **Represented Documents**: 11 distinct documents

### Category Density Per Document

| Document ID (Prefix) | Valid Clauses | Categories Present | Dominant Categories | Diversity Tier |
| :--- | :---: | :---: | :--- | :--- |
| `0859334b` | 50 | **13 / 14** | Definition of Confidential Information (11), Confidentiality Obligations (9) | High Diversity |
| `0a42e159` | 56 | **3 / 14** | Confidentiality Obligations (28), Purpose (21) | Moderate Diversity |
| `0b59dfc4` | 33 | **10 / 14** | Definition of Confidential Information (10), Confidentiality Obligations (6) | High Diversity |
| `2268c5d1` | 25 | **1 / 14** | Additional Information (25) | Single Category |
| `247166e0` | 20 | **1 / 14** | Additional Information (20) | Single Category |
| `266929af` | 3 | **2 / 14** | Party Identification (2), Purpose (1) | Low Diversity |
| `293f5937` | 16 | **6 / 14** | Definition of Confidential Information (5), Return of Information (5) | Moderate Diversity |
| `3504e06a` | 10 | **7 / 14** | Term and Termination (3), Non-Confidential Information (2) | Moderate Diversity |
| `4fd432d8` | 8 | **8 / 14** | Authorized Disclosure (2), Definition of Confidential Information (1) | Moderate Diversity |
| `5180f107` | 10 | **13 / 14** | Confidentiality Obligations (2), IP Rights (1), Liability (1) | High Diversity |
| `53c8f90c` | 6 | **10 / 14** | Competition Rights (1), Employees (1), Remedies for Breach (1) | High Diversity |

---

## 2. Document-Disjoint Candidate Splits Evaluation

Four deterministic partition strategies were evaluated on the 11 valid documents via `scripts/analyze_document_splits.py`:

| Strategy | Train (Docs/Clauses/Cats) | Val (Docs/Clauses/Cats) | Test (Docs/Clauses/Cats) | Missing in Test | Leakage | Status |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| **Strategy A (70/15/15 Ratio)** | 5 / 184 / 14 | 1 / 16 / 6 | 1 / 10 / 7 | **7 categories missing** | 0 | **UNSUITABLE** |
| **Strategy B (60/20/20 Ratio)** | 4 / 164 / 14 | 2 / 36 / 6 | 1 / 10 / 7 | **7 categories missing** | 0 | **UNSUITABLE** |
| **Strategy C (Category Maximization)** | 8 / 144 / 11 | 2 / 43 / 14 | 1 / 50 / 13 | 1 cat missing (*Employees*); **Train missing 3 cats** | 0 | **UNSUITABLE** |
| **Strategy D (Train Category Focus)** | 1 / 50 / 13 | 2 / 16 / 14 | 8 / 171 / 13 | 1 cat missing (*Liability for Damages*) | 0 | **UNSUITABLE** |

### Detailed Missing Categories Breakdown

- **Strategy A & B**: Test split missing 7 categories: *Liability for Damages*, *Competition Rights*, *Term and Termination*, *Intellectual Property*, *Employees*, *Governing Law and Jurisdiction*, *Additional Information*.
- **Strategy C**: Test split missing *Employees*; Train split missing 3 categories: *Liability for Damages*, *Competition Rights*, *Intellectual Property*.
- **Strategy D**: Test split missing *Liability for Damages*; Train split missing *Employees*.

---

## 3. Empirical Feasibility Assessment

**Is the current 11-document dataset sufficient for a publication-grade benchmark?**

**NO.** 

Empirical analysis demonstrates that despite reaching 11 represented documents:
1. No document-disjoint split strategy achieves 14/14 category representation across all three partitions (Train, Validation, and Test) simultaneously.
2. Minority categories like *Liability for Damages*, *Competition Rights*, and *Employees* remain concentrated in a small subset of high-diversity documents (`0859334b`, `5180f107`, `53c8f90c`). Allocating these documents to one split inevitably starves another split of those categories.
3. Blindly relying on document count targets (11 or 15) is insufficient; category density per document is the true governing factor.

---

## 4. Next Priorities for Dataset Expansion

1. **Prioritize Remaining High-Volume Unseen Documents in Queue**:
   - `9a5cb310` (74 clauses remaining)
   - `f28c4f3d` (60 clauses remaining)
   - `d4566b17` (45 clauses remaining)
   - `d714d261` (43 clauses remaining)
   - `e52e4a13` (43 clauses remaining)
2. **Targeted Category Enrichment**:
   - Focus upcoming active learning batches on documents containing *Liability for Damages*, *Competition Rights*, *Intellectual Property*, and *Employees* clauses.
3. **Estimated Additional Document Requirement**:
   - Approximately **3 to 5 additional high-diversity documents** (~14 to 16 total represented documents) are required to establish multi-document redundancy for all 14 categories.

---

CURRENT BENCHMARK: BLOCKED
REASON: Insufficient category coverage across document-disjoint splits. Under standard size ratios, Test is missing 7 categories. Under category maximization, Test or Train is missing critical categories (Liability for Damages, Competition Rights, Employees).
NEXT ACTION: Await Gemini API quota reset (~17.68h retry window), then execute `python scripts/resume_runner.py` prioritizing unseen queue documents (`9a5cb310`, `f28c4f3d`, `d4566b17`) to achieve multi-document category redundancy.
