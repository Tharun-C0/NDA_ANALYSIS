# Baseline Dataset Readiness — Module 4

## Decision
Module 5 classifier training cannot begin. No verified clause-level assignments to the paper's categories were found.

| Metric | Value |
| --- | --- |
| total_documents | 20 |
| total_v3_segments | 717 |
| total_priority_segments | 215 |
| successful_llm_reviewed_segments | 18 |
| unique_segments_with_llm_or_recorded_human_review | 18 |
| recorded_human_review_segments | 7 |
| priority_without_llm_success | 197 |
| high_confidence | 17 |
| medium_confidence | 1 |
| low_confidence | 0 |
| api_failure_events | 1 |
| cleaned_candidate_clauses | 707 |
| removed_segments | 10 |
| conflicting_review_cases | 5 |
| unresolved_merge_cases | 5 |
| unresolved_split_cases | 0 |
| unresolved_retained_cases | 707 |
| unresolved_priority_cases | 205 |
| human_review_cases_including_excluded_audits | 711 |
| existing_human_review_queue | 9 |
| genuinely_labeled_clauses | 0 |
| multi_labeled_clauses | 0 |
| module3a_saved_status | PAUSED_DAILY_QUOTA |
| module5_training_possible | False |

## Meaning of clean and unresolved
The cleaned CSV is a **provisional clause candidate dataset**, not a fully human-validated legal-clause corpus. Unreviewed V3 segments are retained with blank review_action and UNREVIEWED/PRIORITY_UNREVIEWED status; no implicit KEEP decisions are invented. API failures have API_FAILURE status. Non-conflicting REMOVE_NON_LEGAL decisions exclude rows only from this derived CSV. KEEP is retained; merge, split, REVIEW, and conflicting evidence are retained and flagged. Duplicate text is reported, not dropped across documents. All clause text, IDs, and offsets remain those of V3.

Unresolved retained cases include unreviewed non-priority segments. Unresolved priority cases are a narrower subset. Merge-involved counts include recorded human merge requests in conflicts as well as LLM merge suggestions; these overlap conflict counts and must not be summed. Human-review cases include audits on excluded rows. No physical merge/split is performed: segmenter heuristics are not a validated mechanism for applying review decisions.

| Retained review status | Count |
| --- | --- |
| API_FAILURE | 1 |
| CONFLICT_REVIEW | 5 |
| PENDING_MERGE | 3 |
| PRIORITY_UNREVIEWED | 196 |
| UNREVIEWED | 502 |

## Category and multi-label evidence
Available genuine categories: 0. The following is the requested taxonomy with **observed annotation counts**, not assigned labels. No category column or artificial category assignments were added to the clean dataset.

| Requested category | Verified clauses |
| --- | --- |
| Party Identification | 0 |
| Purpose | 0 |
| NDA Type | 0 |
| Definition of Confidential Information | 0 |
| Confidentiality Obligations | 0 |
| Authorized Disclosure | 0 |
| Non-Confidential Information | 0 |
| Liability for Damages | 0 |
| Competition Rights | 0 |
| Term and Termination | 0 |
| Intellectual Property | 0 |
| Employees | 0 |
| Governing Law and Jurisdiction | 0 |
| Additional Information | 0 |

Local evidence:
- V3 JSON and the review CSVs contain structural metadata/actions, not category annotations.
- `data/raw`, `data/splits`, and `training` have no annotated training corpus in this snapshot.
- `tests/test_module1.py` and `README.md` contain synthetic/example class tags; they are not research annotations.
- `external_data/kleister-nda/README.md` describes document-level extraction, not clause-level classification.
- Multiple `party=` values in Kleister-NDA are not multiple clause-category labels.
- Local structured/text data were scanned; test/cache/dependency artifacts and generated cleaned data were excluded. Potential label assignments or unsupported data formats block a build pending provenance inspection.

```json
{
  "scanned_formats": {
    ".txt": 57,
    ".json": 78,
    ".csv": 18,
    ".tsv": 4
  },
  "candidate_files": [],
  "unsupported_files": [],
  "document_level_keys": [
    "effective_date",
    "jurisdiction",
    "party",
    "term"
  ],
  "expected_document_rows": {
    "external_data/kleister-nda/dev-0/expected.tsv": 83,
    "external_data/kleister-nda/train/expected-original.tsv": 254,
    "external_data/kleister-nda/train/expected.tsv": 254
  },
  "compressed_document_rows": {
    "external_data/kleister-nda/dev-0/in.tsv.xz": 83,
    "external_data/kleister-nda/test-A/in.tsv.xz": 203,
    "external_data/kleister-nda/train/in.tsv.xz": 254
  },
  "external_pdf_count": 726
}
```

## Missing data required before Module 5
1. Genuine, provenance-backed human clause-category annotations: document_id, stable clause_id (or exact source span), clause text, and one or more explicitly assigned categories from the 14-category taxonomy. Obtain the authors' annotated data separately or have qualified annotators label this corpus; no download is performed here.
2. Explicit multi-label sets for clauses where multiple categories apply, plus annotation guidelines, reviewer identity, and adjudication/version records. Single-label examples or parser support do not supply multi-label data.
3. Complete/adjudicate structural review, including unreviewed segments, conflicting test-like records, merge/split decisions, and exclusion audits, before treating clause boundaries as validated.
4. Verify adequate genuine examples per target category and create document-disjoint train/validation/test partitions after annotations exist. No unsupported minimum sample size is assumed.

## Validation and preservation
| Check | Status | Count |
| --- | --- | --- |
| required_columns | PASS | 0 |
| missing_clause_ids | PASS | 0 |
| duplicate_clause_ids | PASS | 0 |
| missing_clause_text | PASS | 0 |
| empty_clauses | PASS | 0 |
| duplicate_text | WARN | 25 |
| invalid_review_actions | PASS | 0 |
| missing_review_status | PASS | 0 |
| invalid_review_status | PASS | 0 |
| action_status_consistency | PASS | 0 |
| inconsistent_document_ids | PASS | 0 |
| source_text_integrity | PASS | 0 |
| source_provenance | PASS | 0 |
| cleaning_coverage | PASS | 0 |
| confidence_range | PASS | 0 |
| multi_label_format | PASS | 0 |
| label_provenance | PASS | 0 |
| label_availability | BLOCKED | 0 |
| multi_label_availability | BLOCKED | 0 |

Protected source snapshot: 827 files; SHA-256 manifest digest `2b9565da72c13f3dd991ef434883fb377f5d4a7b011669a234a16aab65d7ce11`. The build verifies identical before/after hashes for original segmentation, human-review outputs, Kleister-NDA files, and human-review reports. Validation warnings/blockers are not a training approval.

Module 4 ends here. No Gemini requests, downloads, classifier training, or Module 5 execution.
