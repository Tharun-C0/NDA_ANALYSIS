# Annotation Validation Report — Module 5B

*Generated: 2026-09-30 04:20 UTC*

## Annotation Queue (`annotation_queue.csv`) — 699 rows

| Check | Severity | Status | Count |
|---|---|---|---|
| [OK] annotated_but_empty_labels | WARN | PASS | 0 |
| [OK] annotated_missing_reviewer | WARN | PASS | 0 |
| [OK] duplicate_clause_ids | ERROR | PASS | 0 |
| [OK] duplicate_labels_in_array | ERROR | PASS | 0 |
| [OK] empty_clause_text | ERROR | PASS | 0 |
| [OK] invalid_annotation_status | ERROR | PASS | 0 |
| [OK] invalid_category_labels_json | ERROR | PASS | 0 |
| [OK] invalid_label_source | BLOCKED | PASS | 0 |
| [OK] invalid_verified_format | ERROR | PASS | 0 |
| [OK] missing_document_ids | ERROR | PASS | 0 |
| [OK] pseudolabel_source_detected | BLOCKED | PASS | 0 |
| [OK] required_columns_queue | ERROR | PASS | 0 |
| [OK] unknown_category_names | ERROR | PASS | 0 |
| [OK] verified_missing_provenance | BLOCKED | PASS | 0 |

## Annotation Database (`annotations.csv`) — 0 rows

| Check | Severity | Status | Count |
|---|---|---|---|
| [OK] db_empty | PASS | PASS | 0 |

## Annotation Summary

| Metric | Queue | DB |
|---|---|---|
| total_clauses | 699 | 0 |
| labeled_clauses | 1 | 0 |
| verified_labeled_clauses | 0 | 0 |
| multi_labeled_clauses | 0 | 0 |
| pending | 699 | 0 |
| annotated | 0 | 0 |
| disputed | 0 | 0 |
| skipped | 0 | 0 |

## Category Distribution (from annotation DB)

| Category | Count | % | Warning |
|---|---|---|---|
| Party Identification | 0 | 0.0 | ZERO |
| Purpose | 0 | 0.0 | ZERO |
| NDA Type | 0 | 0.0 | ZERO |
| Definition of Confidential Information | 0 | 0.0 | ZERO |
| Confidentiality Obligations | 0 | 0.0 | ZERO |
| Authorized Disclosure | 0 | 0.0 | ZERO |
| Non-Confidential Information | 0 | 0.0 | ZERO |
| Liability for Damages | 0 | 0.0 | ZERO |
| Competition Rights | 0 | 0.0 | ZERO |
| Term and Termination | 0 | 0.0 | ZERO |
| Intellectual Property | 0 | 0.0 | ZERO |
| Employees | 0 | 0.0 | ZERO |
| Governing Law and Jurisdiction | 0 | 0.0 | ZERO |
| Additional Information | 0 | 0.0 | ZERO |

## Training Readiness

> Classifier training ready: **NO**

> Reason: Genuine 14-category clause-level annotations are still required.

**Errors:** 0 | **Blocked:** 0 | **Warnings:** 0
