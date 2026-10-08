# Active Learning Seed Selection — Module 5B

## Summary

| Metric | Value |
|---|---|
| **Total seed clauses** | 100 |
| **Documents covered** | 20 / 20 (all documents) |
| **Clauses per document** | 5 (uniform) |
| **Selection date** | 2026-09-30 |
| **Source queue** | `data/annotations/annotation_queue.csv` (699 clauses) |
| **Seed file** | `data/annotations/active_learning_seed.csv` |

---

## Selection Goal

The goal of the seed set is to give human annotators a **maximally diverse starting
sample** across all 20 NDA source documents before any model-assisted selection
is performed.  Good seed diversity ensures that:

1. Initial human annotations span all documents, preventing document-level bias in the
   baseline model trained after Round 0.
2. The baseline model sees examples of short boilerplate text, medium definitions, and
   long operative clauses, making its uncertainty estimates more calibrated.
3. No document is over- or under-represented in Round 0 annotations.

---

## Selection Strategy

### Step 1 — Group by document
All 699 queue clauses were grouped into their 20 source documents.

### Step 2 — Assign length buckets
Each clause was assigned to one of three length buckets based on character count
of stripped clause text:

| Bucket | Character range |
|---|---|
| `short` | < 100 characters |
| `medium` | 100 – 399 characters |
| `long` | ≥ 400 characters |

### Step 3 — Per-document diversity quota
For each document, exactly **5 clauses** were selected using the following target
bucket sequence: `[short, medium, long, medium, long]`.

- Within each bucket, clauses with `PRIORITY_UNREVIEWED` status were preferred
  over `UNREVIEWED` clauses.
- Within the same status tier, clauses were ordered by a hash-spread score derived
  from the MD5 of the clause ID, ensuring that no document always contributes
  only its first or last clauses.
- If the preferred bucket was exhausted, a fallback order was applied:
  `medium → long → short`.

### Step 4 — Deduplication
A set of already-selected clause IDs was maintained to prevent any clause from
being selected twice.

---

## Resulting Distribution

### Length bucket breakdown

| Bucket | Count | Percentage |
|---|---|---|
| short | 20 | 20% |
| medium | 40 | 40% |
| long | 40 | 40% |

### Segment type breakdown

| Segment type | Count |
|---|---|
| legal_clause | 87 |
| fragment | 5 |
| header | 4 |
| section_heading | 4 |

### Review status breakdown

| Review status | Count |
|---|---|
| PRIORITY_UNREVIEWED | 41 |
| UNREVIEWED | 59 |

### Text length statistics

| Metric | Value (characters) |
|---|---|
| Minimum | 1 |
| Maximum | 5,621 |
| Average | 597 |

---

## What was NOT done

- Category labels were **not** assigned to any seed clause.
- No LLM was called to pre-score or pre-classify seed clauses.
- No existing Module 3A pseudo-labels were used to influence selection.
- The first-100-rows strategy was explicitly avoided.

---

## Reproducibility

The selection is deterministic given the same input queue, because:

- Document grouping is performed by sorted document ID.
- Within-bucket ordering uses MD5 hash of clause ID (no random seed needed).
- The quota and fallback sequence are fixed constants.

To regenerate: `python scripts/_build_seed.py`

---

## Next Step

A human annotator should use the annotation interface to label all 100 seed clauses:

```powershell
.\.venv\Scripts\python.exe app\annotation_app.py --mode seed
```

This will restrict the interface to only the 100 seed clauses until they are all
annotated, then unlock the full active learning workflow.
