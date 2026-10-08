# Module 9: Benchmark Construction Decision Checklist

This checklist evaluates whether the current dataset state supports constructing a defensible, publication-grade document-disjoint evaluation benchmark.

---

## Decision Checklist

| # | Requirement / Question | Status | Empirical Evidence / Rationale |
| :-: | :--- | :-: | :--- |
| 1 | **Are enough documents represented?** | **NO** | Only 7 documents are currently represented in the valid pseudo-label dataset (Target >= 15 for 3-way disjoint split). |
| 2 | **Is train document-disjoint from validation/test?** | **YES** | 0 document overlap across train, validation, and test partitions across all tested strategies. |
| 3 | **Is validation category coverage adequate?** | **PARTIAL** | Validation coverage ranges from 6/14 (Strategies A & B) to 12/14 (Strategy C) and 11/14 (Strategy D). |
| 4 | **Is test category coverage adequate?** | **NO** | Test coverage fails across all strategies: Strategy A (2/14), Strategy B (2/14), Strategy C (13/14 but strips Train to 4/14), Strategy D (7/14). |
| 5 | **Are minority categories represented in test?** | **NO** | Minority categories like *No Assignment of Rights* (2 clauses) and *Remedies for Breach* (3 clauses) are completely absent in test under balanced ratio splits. |
| 6 | **Are pseudo-label provenance fields preserved?** | **YES** | 100% of records retain `label_source = "llm_pseudo_label"`, model ID, confidence tier, agreement score, and prompt version. |
| 7 | **Are API error rows excluded?** | **YES** | All 49 `API_ERROR` rows and 2 `API_QUOTA_EXHAUSTED` rows are strictly filtered out from all evaluation splits. |
| 8 | **Are duplicate clauses excluded?** | **YES** | Zero duplicate clause IDs or text strings exist within the 203 valid pseudo-labeled clauses dataset. |
| 9 | **Is human verification evidence available?** | **YES** | Stratified human verification sample of 30 clauses (`data/annotations/human_verification_sample.csv`) is created and ready for human audit. |
| 10 | **Is the benchmark defensible for a research paper?** | **NO** | The current 7-document pool cannot produce a defensible 14-category document-disjoint split. Benchmark is **BLOCKED**. |

---

## Detailed Item Audits

### 1. Document Representation
- **Current Count**: 7 documents
- **Threshold**: >= 15 documents (empirically evaluated based on category density)
- **Verdict**: **FAIL**. 13 unseen documents remain in the queue awaiting pseudo-label processing.

### 2. Document Disjointness
- **Verification**: Evaluated via strict document ID set intersection checks.
- **Overlap**: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
- **Verdict**: **PASS**.

### 3. Validation Category Coverage
- **Best Observed**: Strategy C (12/14 categories present in Val).
- **Worst Observed**: Strategy A & B (6/14 categories present in Val).
- **Verdict**: **WARN / PARTIAL**.

### 4. Test Category Coverage
- **Required**: At least 12 of 14 categories present in Test without degrading Train or Val.
- **Best Defensible Split**: Strategy D yields only 7 of 14 categories in Test.
- **Verdict**: **FAIL**.

### 5. Minority Category Distribution
- *No Assignment of Rights*: 2 clauses (100% in Doc `0859334b`).
- *Remedies for Breach*: 3 clauses (100% in Doc `0859334b`).
- *Governing Law & Jurisdiction*: 5 clauses (spread across 2 docs).
- **Verdict**: **FAIL**. Single-document concentration prevents balanced test evaluation.

### 6. Provenance Preservation
- Metadata columns verified: `label_source`, `llm_model`, `llm_confidence_tier`, `llm_agreement_score`, `prompt_version`.
- **Verdict**: **PASS**.

### 7. API Error Exclusion
- 49 `API_ERROR` and 2 `API_QUOTA_EXHAUSTED` records are flagged and quarantined in `data/annotations/llm_pseudo_labels.csv`.
- None of these records are loaded into `classifier_dataset_all_valid.csv` or split matrices.
- **Verdict**: **PASS**.

### 8. Duplicate Exclusion
- Verified via `scripts/check_benchmark_readiness.py`.
- **Verdict**: **PASS**.

### 9. Human Verification Evidence
- Sample file: `data/annotations/human_verification_sample.csv` (30 clauses).
- Clearly distinguishes LLM pseudo-labels from human ground-truth verification.
- **Verdict**: **PASS**.

### 10. Overall Research Defensibility
- **Verdict**: **BLOCKED**. No final research benchmark dataset will be written or declared ready until dataset expansion resolves the document diversity bottleneck.

---

## Final Readiness Verdict

**STATUS: BLOCKED**

- **Reason**: Insufficient document diversity (7 docs available vs. ~12-15 required).
- **Next Action**: Await Gemini API quota reset, then execute `scripts/resume_runner.py` on the 13 unseen documents in the queue.
