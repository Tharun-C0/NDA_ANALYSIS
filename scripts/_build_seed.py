"""Build active_learning_seed.csv — run once, internal script."""
import csv, json, math, re, hashlib
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
QUEUE  = ROOT / "data/annotations/annotation_queue.csv"
SEED_OUT = ROOT / "data/annotations/active_learning_seed.csv"
TARGET = 100

# ── load queue ──────────────────────────────────────────────────────────────
rows = []
with QUEUE.open(encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

# ── length buckets ───────────────────────────────────────────────────────────
def bucket(text):
    n = len(text.strip())
    if n < 100:  return "short"
    if n < 400:  return "medium"
    return "long"

# ── group by document ────────────────────────────────────────────────────────
by_doc = defaultdict(list)
for r in rows:
    by_doc[r["document_id"]].append(r)

docs = sorted(by_doc.keys())
n_docs = len(docs)   # 20 documents

# ── per-document quota: try to take 5 clauses per document = 100 total ───────
# Strategy:
#  1. From each document, pick 1 short, 1 medium, 1 long, and 2 more by hash
#     spread to avoid always taking the first matching clause.
#  2. Within each length bucket, prefer PRIORITY_UNREVIEWED first.
#  3. Fallback to UNREVIEWED if bucket exhausted.
#  4. After per-doc pass, top-up to 100 using global diversity (length spread).

QUOTA_PER_DOC = 5   # 20 docs × 5 = 100 exactly

selected_ids = set()
seed_rows = []

def score_row(r, prefer_status="PRIORITY_UNREVIEWED"):
    """Lower score → higher priority for selection."""
    status_score = 0 if r["review_status"] == prefer_status else 1
    # Spread within document by clause index (hash of clause_id)
    spread = int(hashlib.md5(r["clause_id"].encode()).hexdigest()[:4], 16)
    return (status_score, spread)

for doc_id in docs:
    doc_rows = by_doc[doc_id]
    buckets = defaultdict(list)
    for r in doc_rows:
        b = bucket(r["clause_text"])
        buckets[b].append(r)

    # Sort each bucket by priority then hash-spread
    for b in buckets:
        buckets[b].sort(key=lambda r: score_row(r))

    chosen = []
    target_buckets = ["short", "medium", "long", "medium", "long"]  # 5 picks
    for tb in target_buckets:
        # try preferred bucket, then fallback order
        fallback = [tb] + [x for x in ["medium", "long", "short"] if x != tb]
        for fb in fallback:
            pool = [r for r in buckets[fb] if r["clause_id"] not in selected_ids
                    and r["clause_id"] not in {c["clause_id"] for c in chosen}]
            if pool:
                pick = pool[0]
                chosen.append(pick)
                break
        if len(chosen) >= QUOTA_PER_DOC:
            break

    # Emit chosen
    for r in chosen:
        selected_ids.add(r["clause_id"])
        b = bucket(r["clause_text"])
        reason = (
            f"document_spread; length_bucket={b}; "
            f"review_status={r['review_status']}; "
            f"segment_type={r.get('segment_type','unknown')}"
        )
        seed_rows.append({
            "clause_id":       r["clause_id"],
            "document_id":     r["document_id"],
            "clause_text":     r["clause_text"],
            "segment_type":    r.get("segment_type", ""),
            "review_status":   r.get("review_status", ""),
            "length_bucket":   b,
            "text_length":     len(r["clause_text"].strip()),
            "selection_reason": reason,
            "annotation_status": "PENDING",
        })

print(f"Selected {len(seed_rows)} seed clauses from {n_docs} documents.")

# Verify coverage
doc_coverage = defaultdict(int)
bucket_coverage = defaultdict(int)
for r in seed_rows:
    doc_coverage[r["document_id"]] += 1
    bucket_coverage[r["length_bucket"]] += 1

print("Per-document counts:", {d[:8]: c for d,c in sorted(doc_coverage.items())})
print("Length bucket counts:", dict(bucket_coverage))

# ── write seed CSV ────────────────────────────────────────────────────────────
COLS = ["clause_id", "document_id", "clause_text", "segment_type",
        "review_status", "length_bucket", "text_length",
        "selection_reason", "annotation_status"]
with SEED_OUT.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(seed_rows)
print(f"Saved: {SEED_OUT}")
