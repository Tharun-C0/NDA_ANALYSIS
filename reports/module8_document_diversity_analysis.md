# Module 8 - Document Diversity Analysis & Dataset Expansion Target

> **Status**: **COMPLETE**
> Defines empirical document diversity requirements, document-disjoint split mechanics, candidate expansion scenarios, and target stopping conditions.

---

## 1. Current State Summary

| Metric | Empirical Value |
| :--- | :--- |
| **Total Valid Pseudo-Labels** | 203 clauses |
| **Represented NDA Documents** | 7 documents |
| **Remaining Active Queue Clauses** | 465 clauses |
| **Completely Unseen Documents in Queue** | 13 documents |
| **Total Corpus Potential** | 668 clauses across 20 documents |
| **Dataset-Wide Category Representation**| 14 / 14 categories |

---

## 2. Document-Level Category Distribution (Current 7 Documents)

Analyzing the category richness per document reveals severe concentration:

| Document ID Prefix | Valid Clauses | Categories Present | Dominant / Present Categories |
| :--- | :---: | :---: | :--- |
| 0859334b3622 | 50 | **13** | All categories except Employees |
| 0b59dfc4ce9b | 33 | **10** | Purpose, NDA Type, Def of Conf Info, Obligations, Disclosure, Non-Conf, IP, Employees, Competition, Additional Info |
| 293f59373f6a | 16 | **6** | Party ID, Obligations, Disclosure, Governing Law, Employees, Additional Info |
| 0a42e159b33e | 56 | **3** | Party ID, NDA Type, Additional Info |
| 266929af5f5b | 3 | **2** | Purpose, Additional Info |
| 2268c5d1120f | 25 | **1** | Additional Info only (Boilerplate / SEC Header heavy) |
| 247166e02454 | 20 | **1** | Additional Info only |

---

## 3. The Document-Disjoint Split Problem

A core requirement of legal NLP benchmark evaluation is **Strict Document Disjointness**:
No clause from document D_i may appear in more than one split (Train, Validation, or Test).

### Why the Current 7-Document Dataset Fails Benchmark Requirements
When assigning 7 documents across Train (4), Validation (2), and Test (1):
1. **Category Concentration**: Categories such as Liability for Damages, Term and Termination, Intellectual Property, and Competition Rights exist almost exclusively in document 0859334b3622 (13 categories) and 0b59dfc4ce9b (10 categories).
2. **Test Set Deficiency**: Placing 0859334b in Test leaves Train/Val lacking key categories, or placing it in Train/Val leaves Test with 0 positive instances for minority categories.
3. **Macro F1 Invalidation**: Reporting Macro F1 on a Test set where categories have 0 positive support is scientifically invalid.

---

## 4. Candidate Expansion Scenarios

To resolve the test-set category deficiency, we evaluate 3 dataset expansion scenarios:

| Scenario | Total Documents | Total Clauses | Est. Test Categories | Benchmark Feasibility |
| :--- | :---: | :---: | :---: | :--- |
| **Scenario A: Current State** | **7** | **203** | **1 / 14** | **FAIL** (Severe test-category deficiency) |
| **Scenario B: Partial Expansion** | **12** (+5 unseen) | **~380** | **8-10 / 14** | **SUB-OPTIMAL** (Minority categories missing in Test) |
| **Scenario C: Full Corpus Expansion** | **20** (+13 unseen) | **668** (Full Queue) | **14 / 14** | **OPTIMAL** (Defensible 14-category document-disjoint benchmark) |

---

## 5. Recommended Stopping Condition & Diversity Priority

### Target Stopping Condition
Pseudo-label expansion via the Gemini ensemble will continue until:
1. **Primary Criteria**: All **20 documents** in the corpus are pseudo-labeled (668 total clauses).
2. **Minimum Threshold**: At least **15 diverse documents** are pseudo-labeled, achieving **14/14 category presence** in Train, Validation, and Test splits simultaneously.

### Queue Execution Strategy (scripts/resume_runner.py)
Queue selection uses strict priority tiers:
- **Tier 1 (T1)**: Documents with **0** valid pseudo-labels (13 unseen documents prioritized first).
- **Tier 2 (T2)**: Documents with **1-4** valid pseudo-labels.
- **Tier 3 (T3)**: Documents already sufficiently represented (>= 5 valid pseudo-labels).

---

## 6. Limitations & Risk Mitigation

1. **API Quota Bottleneck**: Free-tier rate limits (~17.68h retry window) require batch execution across multiple days. The resumable runner handles lock cleanups and partial batch persistence safely.
2. **Distant Supervision Noise**: LLM pseudo-labels contain ensemble variance. Downstream classifier experiments (Module 6C) explicitly compare training on high_confidence vs all_valid subsets.

---
*Module 8 - Document Diversity Analysis*
*Created: 2026-10-03*
