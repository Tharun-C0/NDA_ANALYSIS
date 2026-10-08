# Module 11: Category Redundancy Analysis

## Executive Summary

- **Dataset State**: 237 valid pseudo-labeled clauses across **11 represented documents**.
- **Objective**: Identify category redundancy bottlenecks by measuring document-level coverage across all 14 NDA categories.
- **Ranking Basis**: Categories are ranked strictly by empirical document coverage (lowest document count to highest). No subjective terms ("easy", "hard", "important", "unimportant") are used.

---

## 1. Category Redundancy Audit Table

| Category Name | Clause Support | Document Count | % of Represented Docs (N=11) | Represented Document IDs (Prefix) | $\le 2$ Docs? | $\le 3$ Docs? |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| **Liability for Damages** | 4 | **2** | 18.18% | `0859334b`, `5180f107` | **Yes** | **Yes** |
| **Competition Rights** | 3 | **3** | 27.27% | `0859334b`, `0b59dfc4`, `5180f107` | No | **Yes** |
| **Intellectual Property** | 4 | **3** | 27.27% | `0859334b`, `0b59dfc4`, `5180f107` | No | **Yes** |
| **Governing Law and Jurisdiction** | 6 | **3** | 27.27% | `0859334b`, `293f5937`, `5180f107` | No | **Yes** |
| **Term and Termination** | 5 | **4** | 36.36% | `0859334b`, `4fd432d8`, `5180f107`, `53c8f90c` | No | No |
| **NDA Type** | 6 | **5** | 45.45% | `0859334b`, `0a42e159`, `0b59dfc4`, `3504e06a`, `53c8f90c` | No | No |
| **Employees** | 7 | **5** | 45.45% | `0b59dfc4`, `293f5937`, `4fd432d8`, `5180f107`, `53c8f90c` | No | No |
| **Definition of Confidential Information** | 7 | **6** | 54.55% | `0859334b`, `0b59dfc4`, `3504e06a`, `4fd432d8`, `5180f107`, `53c8f90c` | No | No |
| **Non-Confidential Information** | 10 | **6** | 54.55% | `0859334b`, `0b59dfc4`, `3504e06a`, `4fd432d8`, `5180f107`, `53c8f90c` | No | No |
| **Purpose** | 12 | **7** | 63.64% | `0859334b`, `0b59dfc4`, `266929af`, `3504e06a`, `4fd432d8`, `5180f107`, `53c8f90c` | No | No |
| **Authorized Disclosure** | 21 | **7** | 63.64% | `0859334b`, `0b59dfc4`, `293f5937`, `3504e06a`, `4fd432d8`, `5180f107`, `53c8f90c` | No | No |
| **Confidentiality Obligations** | 22 | **7** | 63.64% | `0859334b`, `0b59dfc4`, `293f5937`, `3504e06a`, `4fd432d8`, `5180f107`, `53c8f90c` | No | No |
| **Party Identification** | 24 | **7** | 63.64% | `0859334b`, `0a42e159`, `293f5937`, `3504e06a`, `4fd432d8`, `5180f107`, `53c8f90c` | No | No |
| **Additional Information** | 164 | **9** | 81.82% | `0859334b`, `0a42e159`, `0b59dfc4`, `2268c5d1`, `247166e0`, `266929af`, `293f5937`, `5180f107`, `53c8f90c` | No | No |

---

## 2. Empirical Bottleneck Analysis

1. **Lowest Redundancy Category ($\le 2$ Documents)**:
   - **Liability for Damages**: Present in only 2 documents (`0859334b`, `5180f107`) with 4 total clause instances. Any 3-way document-disjoint split requires placing these 2 documents into at most 2 splits, guaranteeing that at least one split (Train, Val, or Test) has **zero support**.

2. **Low Redundancy Categories ($\le 3$ Documents)**:
   - **Competition Rights**: Present in 3 documents (`0859334b`, `0b59dfc4`, `5180f107`).
   - **Intellectual Property**: Present in 3 documents (`0859334b`, `0b59dfc4`, `5180f107`).
   - **Governing Law and Jurisdiction**: Present in 3 documents (`0859334b`, `293f5937`, `5180f107`).
   - *Impact*: In a 3-way split, having exactly 3 documents means each split can receive at most 1 document containing the category. If any of these documents are placed together in Train, Validation or Test will have zero support.

3. **High Redundancy Categories ($\ge 7$ Documents)**:
   - *Additional Information* (9 docs), *Party Identification* (7 docs), *Confidentiality Obligations* (7 docs), *Authorized Disclosure* (7 docs), *Purpose* (7 docs). These categories achieve high document-level distribution across the corpus.

---

## 3. Benchmark Implication

To construct a publication-grade document-disjoint evaluation benchmark where **every category is represented across Train, Validation, and Test splits simultaneously**, additional document expansion MUST prioritize discovering document redundancy for categories occurring in $\le 3$ documents (*Liability for Damages*, *Competition Rights*, *Intellectual Property*, *Governing Law and Jurisdiction*).
