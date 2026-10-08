# Module 9 - Document/Category Distribution Matrix

> **Status**: **COMPLETE**
> Granular document-level analysis of valid pseudo-labeled clauses across the 14 NDA categories.

---

## 1. Document-Category Matrix Summary

| Document ID Prefix | Valid Clauses | Categories Present | HIGH Conf | MED Conf | LOW Conf | DISAGREEMENT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0859334b7622 | 50 | **13** | 13 | 7 | 8 | 22 |
| 0b59dfc4ce9b | 33 | **10** | 3 | 5 | 6 | 19 |
| 293f59373f6a | 16 | **6** | 10 | 3 | 0 | 3 |
| 0a42e159b33e | 56 | **3** | 10 | 0 | 17 | 29 |
| 266929af5f5b | 3 | **2** | 0 | 1 | 0 | 2 |
| 247166e02454 | 20 | **1** | 0 | 0 | 7 | 13 |
| 2268c5d1120f | 25 | **1** | 2 | 0 | 11 | 12 |

---

## 2. Category Prevalence Across Documents

| Category Name | Total Support | % of Valid Set (203) | Document Representation |
| :--- | :---: | :---: | :---: |
| Party Identification | 15 | 7.4% | 3 / 7 docs |
| Purpose | 6 | 3.0% | 3 / 7 docs |
| NDA Type | 4 | 2.0% | 3 / 7 docs |
| Definition of Confidential Information | 3 | 1.5% | 2 / 7 docs |
| Confidentiality Obligations | 12 | 5.9% | 3 / 7 docs |
| Authorized Disclosure | 12 | 5.9% | 3 / 7 docs |
| Non-Confidential Information | 3 | 1.5% | 2 / 7 docs |
| Liability for Damages | 3 | 1.5% | 1 / 7 docs |
| Competition Rights | 2 | 1.0% | 2 / 7 docs |
| Term and Termination | 1 | 0.5% | 1 / 7 docs |
| Intellectual Property | 3 | 1.5% | 2 / 7 docs |
| Employees | 3 | 1.5% | 2 / 7 docs |
| Governing Law and Jurisdiction | 5 | 2.5% | 2 / 7 docs |
| Additional Information | 157 | 77.3% | 7 / 7 docs |

---

## 3. Key Document Diversity Findings

### 3.1 Document Categorization by Diversity Tiers
- Single-Category Documents: 2 documents (2268c5d1120f: 25 clauses, 247166e02454: 20 clauses)
- Moderate-Diversity Documents (>=3 Categories): 4 documents (0859334b3622: 13, 0b59dfc4ce9b: 10, 293f59373f6a: 6, 0a42e159b33e: 3)
- High-Diversity Documents (>=5 Categories): 3 documents (0859334b3622: 13, 0b59dfc4ce9b: 10, 293f59373f6a: 6)
- Maximum Diversity Document: 0859334b3622 containing 13 of 14 categories across 50 valid clauses.

### 3.2 Concentration Risk Assessment
1. Single Point of Failure: Document 0859334b3622 is the ONLY document containing Liability for Damages (3 clauses) and Term and Termination (1 clause).
2. Minority Category Concentration: Employees (3 clauses) exists only in 0b59dfc4ce9b (2) and 293f59373f6a (1).
3. Partitioning Bottleneck: Because 0859334b3622 contains 13 categories, assigning it to any single split deprives remaining splits of multiple categories.

---
*Module 9 - Document Category Matrix Report*
*Created: 2026-10-03*
