# Module 14: Document Diversity Policy

## Executive Summary

This policy governs the document selection strategy for future pseudo-labeling and active learning expansion. It establishes that **document diversity is strictly more critical to benchmark validity than clause count volume**.

---

## 1. Why Document Diversity Supersedes Clause Volume

In legal contract NLP research:
1. **Document-Level Drafting Styles**: Individual NDA contracts are drafted using specific boilerplates and firm-specific terminology patterns. A model trained on 1,000 clauses from only 3 documents will overfit to the drafting idiosyncrasies of those 3 firms.
2. **Document-Disjoint Generalization**: A publication-grade research benchmark MUST evaluate model generalization on **completely unseen legal contracts**. Evaluating test clauses originating from training documents causes severe data leakage.
3. **Category Redundancy Wall**: If a category (e.g., *Liability for Damages*) exists in only 2 documents, it cannot be distributed across Train, Validation, and Test splits simultaneously. Adding 100 extra clauses from those same 2 documents does NOT resolve the split bottleneck.

---

## 2. Preferred Future Labeling Priority Order

Future active learning batches MUST select clauses according to the following 5-tier priority hierarchy:

| Priority Rank | Document Category / Status | Operational Target | Selection Rationale |
| :---: | :--- | :--- | :--- |
| **Tier 1 (Highest)** | **Completely Unseen Documents (0 Valid Labels)** | `d714d261`, `9a5cb310`, `586c367e`, `f28c4f3d`, `e52e4a13`, `d4566b17`, `b82a10c4`, `c58882f7`, `64303e5a` | Directly increases represented document count ($N_{\text{docs}}$) towards the target ($\ge 14$). |
| **Tier 2** | **Documents Useful for Underrepresented Categories** | Clauses matching signals for *Liability for Damages*, *Competition Rights*, *Intellectual Property*, *Governing Law* | Builds document-level category redundancy for categories present in $\le 3$ documents. |
| **Tier 3** | **Multi-Target Signal Documents** | Unseen documents matching 3 or 4 underrepresented target category signals simultaneously | Maximizes category discovery efficiency per API call. |
| **Tier 4** | **Sparse Documents ($<5$ Valid Labels)** | Documents with 1 to 4 existing valid labels | Completes document characterization to confirm multi-category richness. |
| **Tier 5 (Lowest)** | **Already Well-Represented Documents ($\ge 5$ Valid Labels)** | `0859334b`, `0a42e159`, `0b59dfc4`, `293f5937` | Yields diminishing returns for document-disjoint split construction. |

---

## 3. Operational Rules

1. **Round-Robin Sampling**: Process initial 5-clause sampling batches across all Tier 1 unseen documents before increasing clause counts per document.
2. **Zero Modification Rule**: Existing successful pseudo-labels MUST NOT be modified or deleted.
3. **Exclusion of Quarantined Rows**: `API_ERROR` and `API_QUOTA_EXHAUSTED` rows MUST remain quarantined and excluded from selection.
