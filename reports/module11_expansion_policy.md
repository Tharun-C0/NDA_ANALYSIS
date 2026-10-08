# Module 11: Targeted Document Expansion Policy

## Executive Summary

This policy governs the active learning queue selection strategy for expanding the NDA clause dataset. Its primary goal is to **maximize document diversity and category redundancy** across the document pool so that a defensible 3-way document-disjoint evaluation split can be established.

---

## 1. Distinction Between KNOWN vs. UNKNOWN Information

> [!IMPORTANT]
> - **KNOWN**: Categories and clause counts derived from the **237 valid pseudo-labeled clauses** currently in `llm_pseudo_labels.csv`.
> - **UNKNOWN**: Category labels for clauses remaining in `pseudolabel_resume_queue.csv`. We do NOT assume or fabricate category identities for unseen clauses prior to LLM pseudo-labeling.

---

## 2. Priority Ordering & Rationale

### Tier 1 Priority: Completely Unseen Documents (0 Valid Labels)
- **Target Documents**: `9a5cb310`, `586c367e`, `f28c4f3d`, `d4566b17`, `d714d261`, `e52e4a13`, `64303e5a`, `b82a10c4`, `c58882f7` (9 documents total).
- **Rationale**: Adding new represented documents directly addresses the core benchmark blocker (Document Diversity). Each new document introduces a distinct legal drafting style and potential multi-category representation.

### Tier 2 Priority: Sparse / Low-Coverage Documents ($<5$ Valid Labels)
- **Target Documents**: Documents with fewer than 5 valid pseudo-labeled clauses.
- **Rationale**: Small initial samples that need slight expansion to confirm document-level category richness.

### Tier 3 Priority: Well-Represented Documents ($\ge 5$ Valid Labels)
- **Target Documents**: `0859334b`, `0a42e159`, `0b59dfc4`, `2268c5d1`, `247166e0`, `293f5937`, `3504e06a`, `5180f107`, `53c8f90c`.
- **Rationale**: Exhausting clauses in already well-represented documents yields diminishing returns for document-level split diversity.

---

## 3. Targeted Category Redundancy Goals

Based on the empirical category redundancy audit, dataset expansion must build multi-document redundancy for categories present in $\le 3$ documents:

1. **Liability for Damages**: Currently present in only **2 documents** (`0859334b`, `5180f107`). Needs at least 2 additional documents.
2. **Competition Rights**: Currently present in only **3 documents** (`0859334b`, `0b59dfc4`, `5180f107`). Needs at least 1 additional document.
3. **Intellectual Property**: Currently present in only **3 documents** (`0859334b`, `0b59dfc4`, `5180f107`). Needs at least 1 additional document.
4. **Governing Law and Jurisdiction**: Currently present in only **3 documents** (`0859334b`, `293f5937`, `5180f107`). Needs at least 1 additional document.

---

## 4. Initial Sampling Batch Size & Stopping Criteria Per Document

1. **Initial Sampling Batch**:
   - Process exactly **5 clauses per unseen document** in the initial round.
   - *Purpose*: Sample the document's category distribution without consuming API quota on a single document.

2. **Per-Document Round-Robin Rule**:
   - Do NOT process a single unseen document to completion (e.g., all 74 clauses of `9a5cb310`) while other unseen documents remain untouched.
   - Cycle through all 9 unseen documents with 5-clause sampling batches first.

3. **Document Switching Condition**:
   - After 5 valid pseudo-labels are obtained for a new document, pause that document and proceed to the next unseen document in queue order.

4. **Batch Execution Order**:
   - Preserve original queue clause order within each document.
   - Exclude already completed, errored, or quarantined clause IDs.
