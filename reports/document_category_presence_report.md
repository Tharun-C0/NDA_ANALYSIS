# Module 6 — Document Category Presence & Evaluation Design Report

> [!IMPORTANT]
> **RESEARCH & EVALUATION DESIGN FINDING**:
> Analysis of all 7 currently pseudo-labeled NDA documents reveals that 13 out of 14 categories are concentrated in just 2 documents (`0859334b` and `0b59dfc4`).
> As a result, under strict document-disjoint splitting, it is **mathematically impossible** for all 3 splits (Train, Val, Test) to simultaneously contain 12–14 categories until additional documents are pseudo-labeled.

---

## 1. Document Category Breakdown (203 Usable Pseudo-Labels)

The table below details the clause count, category coverage, and exact present/absent categories for all 7 currently processed documents:

| Document ID | Clause Count | Categories Present | Categories Absent | Category Details |
| :--- | :---: | :---: | :---: | :--- |
| `0859334b76224ff82c1312ae7b2b5da1` | 50 | **13** / 14 | 1 / 14 | **Present**: Party Id, Purpose, NDA Type, Def of Conf Info, Conf Oblig, Auth Discl, Non-Conf Info, Liab for Dam, Comp Rights, Term & Term, IP, Gov Law.<br>**Absent**: Employees |
| `0b59dfc4ce9b40b0c39759dc1ade14bc` | 33 | **10** / 14 | 4 / 14 | **Present**: Purpose, NDA Type, Def of Conf Info, Conf Oblig, Auth Discl, Non-Conf Info, Comp Rights, IP, Employees, Add Info.<br>**Absent**: Party Id, Liab for Dam, Term & Term, Gov Law |
| `293f59373f6a966b13cd7463b4617a6f` | 16 | **6** / 14 | 8 / 14 | **Present**: Party Id, Conf Oblig, Auth Discl, Employees, Gov Law, Add Info.<br>**Absent**: Purpose, NDA Type, Def of Conf Info, Non-Conf Info, Liab for Dam, Comp Rights, Term & Term, IP |
| `0a42e159b33ed521c4157d8babfaf3c1` | 56 | **3** / 14 | 11 / 14 | **Present**: Party Id, NDA Type, Add Info.<br>**Absent**: All other 11 categories |
| `266929af5f5b1ddb4018f2633cb96e24` | 3 | **2** / 14 | 12 / 14 | **Present**: Purpose, Add Info.<br>**Absent**: All other 12 categories |
| `2268c5d1120f1abd57170d689f496418` | 25 | **1** / 14 | 13 / 14 | **Present**: Add Info.<br>**Absent**: All other 13 categories |
| `247166e0245431dcf97ee884f1f07e35` | 20 | **1** / 14 | 13 / 14 | **Present**: Add Info.<br>**Absent**: All other 13 categories |

---

## 2. Evaluation Bottleneck Analysis

In the initial default split:
- **Train** received `0859334b` + `0a42e159` + `266929af` (109 clauses, 13 categories)
- **Validation** received `0b59dfc4` + `293f5937` (49 clauses, 12 categories)
- **Test** received `2268c5d1` + `247166e0` (45 clauses, **ONLY 1 category**: *Additional Information*)

### Why This Occurred:
Because documents `2268c5d1` (25 clauses) and `247166e0` (20 clauses) happen to contain exclusively *Additional Information* clauses. Placing them together in Test resulted in zero positive samples for 13 of the 14 categories in Test.

---

## 3. Exhaustive Document Allocation Analysis

An exhaustive search over all $3^7 = 2,187$ possible document assignments revealed:

| Strategy | Train Split | Validation Split | Test Split | Evaluation Trade-off |
| :--- | :--- | :--- | :--- | :--- |
| **Initial Default** | 109 c / 13 cats | 49 c / 12 cats | 45 c / **1 cat** | Test cannot evaluate 13/14 categories. |
| **Re-Allocated (Best Test Coverage)** | 78 c / 10 cats | 19 c / 7 cats | 106 c / **13 cats** | Test evaluates 13/14 categories; Val has 7 cats. |
| **Balanced CV** | 100 c / 8 cats | 53 c / 10 cats | 50 c / **13 cats** | Maximize min categories across all 3 splits simultaneously (8 cats). |

---

## 4. Scientifically Defensible Recommendations

### Recommendation A: Immediate Re-Allocation for Pipeline Verification
If preliminary/debug training is desired now, re-allocate the 7 documents as:
- **Train**: `0b59dfc4` + `2268c5d1` + `247166e0` (78 clauses, 10 categories)
- **Validation**: `293f5937` + `266929af` (19 clauses, 7 categories)
- **Test**: `0859334b` + `0a42e159` (106 clauses, **13 categories**)

This ensures the Test set has positive examples for 13 out of 14 categories.

### Recommendation B: Resume Gemini Pseudo-Labeling First (STRONGLY RECOMMENDED)
- The current 203 pseudo-labels represent only **28.3% of the total dataset** (203 / 717 V3 clauses).
- Remaining queue: **465 clauses** across ~15 additional NDA documents.
- When the Gemini free-tier quota resets in ~18 hours, completing pseudo-labeling will provide **20+ documents**.
- With 20+ documents, a document-disjoint split can naturally achieve 14/14 category coverage across Train, Validation, AND Test without forcing skewed document splits.

---

## 5. Strategic Decision for User

1. **Option 1 (Preliminary Debug Run)**: Re-allocate the current 7 documents so Test gets 13 categories, then run a 1-epoch debug test of EXP-01 to verify training code.
2. **Option 2 (Recommended Main Benchmarking Path)**: **WAIT** ~18 hours for the Gemini quota reset, resume pseudo-labeling to process all 717 clauses across 20+ documents, and build the definitive 14-category document-disjoint benchmark split.
