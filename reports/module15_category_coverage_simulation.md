# Module 15: Category Coverage Simulation

## Executive Summary

This report evaluates how executing Stage 1 targeted pseudo-labeling (45 clauses across 12 documents) could impact category coverage and document-level redundancy.

---

## 1. Current Facts vs. Hypothetical Future States

> [!IMPORTANT]
> - **CURRENT FACTS**: Confirmed empirical values from the **237 valid pseudo-labeled clauses** currently in `llm_pseudo_labels.csv`.
> - **HYPOTHETICAL FUTURE STATES**: Projections based on candidate clauses in `data/annotations/module14_final_targeted_queue.csv`. Textual signals are used strictly for queue prioritization and MUST NOT be treated as actual pseudo-labels or final metrics.

| Evaluation Metric | Current Facts (N=11 Docs, 237 Clauses) | Hypothetical Future State (N=20 Docs, ~282 Clauses) | Redundancy Impact |
| :--- | :---: | :---: | :--- |
| **Represented Documents ($N_{\text{docs}}$)** | **11 documents** | **20 documents** | Exceeds minimum target $N_{\text{docs}} \ge 14$. |
| **Liability for Damages Coverage** | **2 documents** (`0859334b`, `5180f107`) | **Projected $\ge 4$ documents** | Resolves the 2-doc mathematical bottleneck. |
| **Competition Rights Coverage** | **3 documents** | **Projected $\ge 5$ documents** | Establishes 3-way split representation. |
| **Intellectual Property Coverage** | **3 documents** | **Projected $\ge 5$ documents** | Establishes 3-way split representation. |
| **Governing Law Coverage** | **3 documents** | **Projected $\ge 5$ documents** | Establishes 3-way split representation. |
| **3-Way Disjoint Split Coverage** | **BLOCKED** (*Liability* missing from Val in Strategy A/B/D) | **HYPOTHETICALLY READY** ($\ge 10$ cats in Train, Val, Test) | Enables balanced 3-way partition without category starvation. |

---

## 2. Projected Category Redundancy Matrix

If processing candidate clauses from top-ranked unseen documents (`f28c4f3d`, `d714d261`, `586c367e`, `9a5cb310`) confirms valid pseudo-labels for underrepresented categories in 2 additional documents:

- **Liability for Damages**: Expands from 2 docs $\to$ 4+ docs.
- **Competition Rights**: Expands from 3 docs $\to$ 5+ docs.
- **Intellectual Property**: Expands from 3 docs $\to$ 5+ docs.
- **Governing Law and Jurisdiction**: Expands from 3 docs $\to$ 5+ docs.

---

## 3. Methodological Safeguard

All future state projections remain **HYPOTHETICAL** until actual Gemini API pseudo-labeling is executed and post-batch validation completes.
