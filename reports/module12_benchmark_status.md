# Module 12: Current Benchmark Status & Minimum Data Requirement Analysis

## Executive Summary

- **Current Dataset Size**: 237 valid pseudo-labeled clauses across **11 represented documents** (294 total operational CSV rows).
- **Benchmark Status Verdict**: **BLOCKED FOR PUBLICATION** (Conditionably usable for pipeline debugging only).
- **Primary Blocker**: Severe document-level category bottleneck. *Liability for Damages* is present in only **2 documents** dataset-wide (`0859334b`, `5180f107`), making a 3-way document-disjoint split (Train, Validation, Test) where all 3 partitions contain *Liability for Damages* **mathematically impossible**.

---

## 1. Document & Category Distribution Audit

- **Total Represented Documents**: 11
- **Total NDA Categories**: 14
- **Valid Clause Count**: 237

### Category Document Representation
- **Present in 2 Documents ($\le 2$ docs)**: *Liability for Damages* (4 clauses across `0859334b`, `5180f107`).
- **Present in 3 Documents ($\le 3$ docs)**: *Competition Rights* (3 clauses), *Intellectual Property* (4 clauses), *Governing Law and Jurisdiction* (6 clauses).
- **Present in 4-6 Documents**: *Term and Termination* (4 docs), *NDA Type* (5 docs), *Employees* (5 docs), *Definition of Confidential Information* (6 docs), *Non-Confidential Information* (6 docs).
- **Present in $\ge 7$ Documents**: *Purpose* (7 docs), *Authorized Disclosure* (7 docs), *Confidentiality Obligations* (7 docs), *Party Identification* (7 docs), *Additional Information* (9 docs).

---

## 2. Benchmark Split Strategy Simulation Summary

Four document-disjoint split strategies were simulated on the current 237 valid clauses:

| Strategy | Train (Docs/Clauses/Cats) | Val (Docs/Clauses/Cats) | Test (Docs/Clauses/Cats) | Missing in Train | Missing in Val | Missing in Test | Leakage |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- | :---: |
| **Strategy A (Current Split)** | 3 / 109 / 13 | 2 / 49 / 12 | 6 / 79 / 14 | *Employees* | *Liability*, *Term & Term.* | None | **0** |
| **Strategy B (Maximize Test)** | 6 / 122 / 10 | 2 / 22 / 11 | 3 / 93 / 14 | *Liability*, *Competition*, *IP*, *Gov. Law* | *Liability*, *Competition*, *IP* | None | **0** |
| **Strategy C (Balanced Multi-Split)** | 5 / 154 / 13 | 3 / 59 / 12 | 3 / 24 / 14 | *Employees* | *Liability*, *Term & Term.* | None | **0** |
| **Strategy D (Holdout Equalization)** | 5 / 154 / 13 | 3 / 55 / 13 | 3 / 28 / 14 | *Employees* | *Liability* | None | **0** |

### Simulation Insights
1. **Test Split Coverage**: Strategies A, C, and D achieve **14/14 category coverage in Test**.
2. **Train/Val Starvation**: Achieving 14/14 in Test leaves *Employees* missing from Train (13/14) and *Liability for Damages* missing from Val (12-13/14).
3. **The 2-Document Mathematical Wall**: Because *Liability for Damages* exists in only 2 documents, no partition can assign it to Train, Validation, AND Test simultaneously without violating document-disjointness.

---

## 3. Exact Minimum Data Requirement Calculation

To convert the current dataset into a publication-grade research benchmark, the following empirical additions are strictly required:

| Category Name | Current Document Count | Required Document Count | Additional Documents Needed | Priority |
| :--- | :---: | :---: | :---: | :--- |
| **Liability for Damages** | 2 | $\ge 4$ | **+2 documents** | **URGENT / CRITICAL** |
| **Competition Rights** | 3 | $\ge 4$ | **+1 document** | **HIGH** |
| **Intellectual Property** | 3 | $\ge 4$ | **+1 document** | **HIGH** |
| **Governing Law and Jurisdiction** | 3 | $\ge 4$ | **+1 document** | **HIGH** |
| **Term and Termination** | 4 | $\ge 5$ | **+1 document** | **MEDIUM** |

### Target Benchmark Thresholds
- **Minimum Represented Documents Target**: **$\ge 14$ well-distributed multi-category documents** (currently 11).
- **Target Clause Batch**: Process 45 clauses from the 9 Tier 1 unseen documents (`586c367e`, `64303e5a`, `9a5cb310`, `b82a10c4`, `c58882f7`, `d4566b17`, `d714d261`, `e52e4a13`, `f28c4f3d`).

---

## 4. Benchmark Status Verdict

**STATUS: BLOCKED**

- **Reason**: *Liability for Damages* exists in only 2 documents, preventing complete 14-category presence across Train, Validation, and Test splits simultaneously. Human verification is pending.
- **Next Action**: Await Gemini API quota reset, then execute `python scripts/resume_runner.py --max-clauses 45`.
