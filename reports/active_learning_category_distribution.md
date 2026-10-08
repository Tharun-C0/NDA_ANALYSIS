# Active Learning Category Distribution — Module 5B

> Updated by `scripts/run_active_learning_round.py` after each annotation round.

## Current Status: Round 0 — IN PROGRESS (0 annotations)

| Category | Annotations | % | Docs | R0 | R1 | R2 | R3 | Warning |
|---|---|---|---|---|---|---|---|---|
| Party Identification | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | ⚠ ZERO |
| Purpose | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | ⚠ ZERO |
| NDA Type | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | ⚠ ZERO |
| Definition of Confidential Information | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | ⚠ ZERO |
| Confidentiality Obligations | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | ⚠ ZERO |
| Authorized Disclosure | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | ⚠ ZERO |
| Non-Confidential Information | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | ⚠ ZERO |
| Liability for Damages | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | ⚠ ZERO |
| Competition Rights | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | ⚠ ZERO |
| Term and Termination | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | ⚠ ZERO |
| Intellectual Property | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | ⚠ ZERO |
| Employees | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | ⚠ ZERO |
| Governing Law and Jurisdiction | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | ⚠ ZERO |
| Additional Information | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | ⚠ ZERO |

## Imbalance Warnings

> [!WARNING]
> All 14 categories currently have 0 examples. No baseline model can be trained until
> human annotations are completed for the seed set.

**Thresholds used:**
- `ZERO_EXAMPLES` — 0 annotated clauses for this category
- `SPARSE` — fewer than 5 annotated clauses (training unreliable)
- `UNDERREPRESENTED` — fewer than 10% of the most frequent category

Labels are **never** artificially balanced. Imbalance is reported so that
annotators can prioritise underrepresented categories during later rounds.
