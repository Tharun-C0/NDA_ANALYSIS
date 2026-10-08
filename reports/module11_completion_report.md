# Module 11: Targeted Document Expansion — Completion Report

## Executive Summary

Module 11 (Targeted Document Expansion) is complete. This module analyzed category redundancy bottlenecks across the 237 valid pseudo-labeled clauses, inventoried all remaining unseen queue documents, established a diversity-first targeted expansion policy, generated a deterministic round-robin queue strategy script (`scripts/targeted_expansion_plan.py`), and defined empirical benchmark stopping criteria.

The simulation re-check confirms that **EXPANSION IS REQUIRED** before a publication-grade document-disjoint benchmark can be established.

---

## 1. Current Dataset State

- **Total Operational CSV Rows**: 294
- **Valid Pseudo-Labeled Clauses**: 237 (80.61%)
- **Quarantined API Error / Quota Rows**: 57 (19.39%)
- **Represented Documents**: 11 distinct documents
- **Pipeline Infrastructure**: READY
- **Classifier Experiments**: NOT RUN

---

## 2. Category Redundancy Analysis Summary

- **Report**: [`reports/module11_category_redundancy_analysis.md`](file:///f:/NDA/NDA/reports/module11_category_redundancy_analysis.md)
- **CSV Data**: [`reports/module11_category_redundancy_analysis.csv`](file:///f:/NDA/NDA/reports/module11_category_redundancy_analysis.csv)
- **Bottlenecks Identified**:
  - **$\le 2$ Documents**: *Liability for Damages* (present in only 2 docs: `0859334b`, `5180f107`).
  - **$\le 3$ Documents**: *Competition Rights* (3 docs), *Intellectual Property* (3 docs), *Governing Law and Jurisdiction* (3 docs).
  - *Impact*: Any 3-way split starves Train, Validation, or Test of these categories.

---

## 3. Unseen Document Inventory

- **CSV Data**: [`reports/module11_unseen_document_inventory.csv`](file:///f:/NDA/NDA/reports/module11_unseen_document_inventory.csv)
- **Queue Summary**: 423 total pending clauses remaining across 13 documents.
- **Unseen Tier 1 Documents (0 Valid Labels)**: 9 documents (`9a5cb310`, `586c367e`, `f28c4f3d`, `d4566b17`, `d714d261`, `e52e4a13`, `64303e5a`, `b82a10c4`, `c58882f7`). Total pending clauses = 382.
- **Represented Tier 3 Documents ($\ge 5$ Valid Labels)**: 4 documents (`53c8f90c`, `4fd432d8`, `293f5937`, `3504e06a`). Total pending clauses = 41.

---

## 4. Targeted Expansion Policy & Batch Strategy

- **Policy Report**: [`reports/module11_expansion_policy.md`](file:///f:/NDA/NDA/reports/module11_expansion_policy.md)
- **Batch Script**: [`scripts/targeted_expansion_plan.py`](file:///f:/NDA/NDA/scripts/targeted_expansion_plan.py)
- **Round-Robin Sampling Strategy**:
  - Sample **5 clauses per unseen document** in the initial round across all 9 Tier 1 unseen documents.
  - Recommended Next Expansion Batch: **45 clauses across 9 unseen documents** (`586c367e`, `64303e5a`, `9a5cb310`, `b82a10c4`, `c58882f7`, `d4566b17`, `d714d261`, `e52e4a13`, `f28c4f3d`).
  - Preserves original queue order within each document and excludes completed/quarantined clauses.

---

## 5. Benchmark Stopping Criteria

- **Report**: [`reports/module11_stopping_criteria.md`](file:///f:/NDA/NDA/reports/module11_stopping_criteria.md)
- **Composite 5-Point Rule**:
  1. $N_{\text{docs}} \ge 14$ represented documents.
  2. Every one of the 14 NDA categories present in $\ge 3$ distinct documents.
  3. Severe bottleneck categories (*Liability for Damages*, *Competition Rights*, *IP*) present in $\ge 3$ distinct documents.
  4. At least one document-disjoint split achieves $\text{Train} \ge 13$, $\text{Val} \ge 10$, $\text{Test} \ge 10$ categories.
  5. Zero document or clause leakage ($\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$).

---

## 6. Benchmark Recheck Simulation Results

- **Simulation Executed**: `python scripts/analyze_document_splits.py`
- **Result**: Standard ratios (Strategy A/B) leave Test missing **7 categories**. Heuristic maximization (Strategy C/D) leaves Test or Train missing critical categories (*Liability for Damages* or *Employees*).
- **Estimated Expansion Required**: ~3 to 5 additional high-diversity documents (sampling 45 clauses across the 9 Tier 1 unseen documents).

---

## 7. Current Benchmark Status & Final Status

- **Status Verdict**: **EXPANSION REQUIRED** (Current benchmark is **BLOCKED**)
- **Exact Blocker**: Lack of multi-document category redundancy for *Liability for Damages* (2 docs), *Competition Rights* (3 docs), and *Intellectual Property* (3 docs).
- **Exact Next Action After Gemini Quota Resets**: Execute `python scripts/resume_runner.py --max-clauses 45` to process the 45-clause diversity-first batch across all 9 unseen documents.

---

MODULE 11 STATUS:
Category redundancy analysis: PASS
Unseen document analysis: PASS
Expansion policy: PASS
Benchmark simulation: PASS
Current benchmark: BLOCKED
Next action: Await Gemini API quota reset (~17.68h retry window), then execute `python scripts/resume_runner.py --max-clauses 45` to process the 45-clause diversity-first batch across all 9 unseen documents.
