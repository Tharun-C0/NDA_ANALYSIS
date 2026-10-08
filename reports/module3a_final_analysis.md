# Module 3A Final Snapshot Analysis

The user has closed Module 3A. Its persisted status is `PAUSED_DAILY_QUOTA`, not a fully reviewed queue. Module 4 uses only saved evidence; no Gemini calls or reruns of Modules 1, 2, or 3A.

| Metric | Value |
| --- | --- |
| total_documents | 20 |
| total_v3_segments | 717 |
| total_priority_segments | 215 |
| successful_llm_reviewed_segments | 18 |
| unique_segments_with_llm_or_recorded_human_review | 18 |
| recorded_human_review_segments | 7 |
| priority_without_llm_success | 197 |
| KEEP | 0 |
| REMOVE_NON_LEGAL | 14 |
| MERGE_WITH_NEXT | 4 |
| MERGE_WITH_PREVIOUS | 0 |
| SPLIT | 0 |
| REVIEW | 0 |
| high_confidence | 17 |
| medium_confidence | 1 |
| low_confidence | 0 |
| api_failure_events | 1 |
| cleaned_candidate_clauses | 707 |
| removed_segments | 10 |

## Counting and confidence definitions
Counts use unique SUCCESS clause IDs across the log, reconciled with both pseudo-label and human-review CSVs. Human review records are reported separately and do not turn suggestions into category labels. Confidence: high >= 0.90; medium >= 0.70 and < 0.90; low < 0.70. API failures count persisted terminal ERROR/CONNECTIVITY_ERROR events, not recovered retry attempts. API errors and unreviewed rows are never counted as AI REVIEW.

## Input reconciliation
| File | Rows |
| --- | --- |
| review_dataset.csv | 717 |
| priority_review.csv | 215 |
| llm_pseudo_labels.csv | 10 |
| human_review_required.csv | 9 |

## Saved statistics (historical run, not recalculated coverage)
| Field | Saved value |
| --- | --- |
| total_segments | 215 |
| high_confidence | 17 |
| medium_confidence | 1 |
| low_confidence | 0 |
| ai_review | 0 |
| merge_cases | 4 |
| split_cases | 0 |
| random_audit_cases | 4 |
| human_review_required | 9 |
| api_failures | 1 |
| already_completed_before_run | 1 |
| newly_processed | 18 |
| successful | 18 |
| successful_this_run | 17 |
| remaining | 197 |
| keep | 0 |
| remove_non_legal | 14 |
| merge_with_next | 4 |
| merge_with_previous | 0 |
| api_failures_this_run | 1 |
| connectivity_failures_this_run | 0 |
| total_processing_time_seconds | 264.54 |
| run_status | PAUSED_DAILY_QUOTA |
| timestamp | 2026-09-30T08:35:21.755268 |
| percentage_requiring_human_review | 4.19 |

## Conservative resolution of conflicting actions
Older priority-queue records include reviewer `tester` and reason `test reason`. Their research provenance is not established. They are not silently promoted to trusted human annotations or discarded. Any conflicting recorded human/LLM action is retained with effective action REVIEW, original actions, and provenance. `test_save.csv` is a test artifact, not an authority. This is why the effective removal count can be smaller than the LLM REMOVE_NON_LEGAL count.

| Clause ID | LLM action | Recorded human action |
| --- | --- | --- |
| 0859334b76224ff82c1312ae7b2b5da1_clause_002 | REMOVE_NON_LEGAL | KEEP |
| 0859334b76224ff82c1312ae7b2b5da1_clause_042 | REMOVE_NON_LEGAL | MERGE_WITH_NEXT |
| 0a42e159b33ed521c4157d8babfaf3c1_clause_000 | REMOVE_NON_LEGAL | REVIEW |
| 0a42e159b33ed521c4157d8babfaf3c1_clause_001 | REMOVE_NON_LEGAL | KEEP |
| 0a42e159b33ed521c4157d8babfaf3c1_clause_003 | MERGE_WITH_NEXT | KEEP |

## Provisional exclusions (reversible using unchanged V3)
| Clause ID | Review action | Still queued for human audit |
| --- | --- | --- |
| 0859334b76224ff82c1312ae7b2b5da1_clause_000 | REMOVE_NON_LEGAL | False |
| 0859334b76224ff82c1312ae7b2b5da1_clause_001 | REMOVE_NON_LEGAL | True |
| 0a42e159b33ed521c4157d8babfaf3c1_clause_034 | REMOVE_NON_LEGAL | True |
| 0a42e159b33ed521c4157d8babfaf3c1_clause_041 | REMOVE_NON_LEGAL | True |
| 0a42e159b33ed521c4157d8babfaf3c1_clause_042 | REMOVE_NON_LEGAL | False |
| 0a42e159b33ed521c4157d8babfaf3c1_clause_055 | REMOVE_NON_LEGAL | True |
| 0b59dfc4ce9b40b0c39759dc1ade14bc_clause_000 | REMOVE_NON_LEGAL | False |
| 0b59dfc4ce9b40b0c39759dc1ade14bc_clause_032 | REMOVE_NON_LEGAL | False |
| 2268c5d1120f1abd57170d689f496418_clause_000 | REMOVE_NON_LEGAL | False |
| 2268c5d1120f1abd57170d689f496418_clause_001 | REMOVE_NON_LEGAL | False |
