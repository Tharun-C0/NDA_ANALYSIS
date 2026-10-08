"""Active Learning Status — Module 5B.

Displays a current snapshot of the annotation pipeline without modifying any files.

Usage:
    python scripts/active_learning_status.py
"""
from __future__ import annotations
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = (
    "Party Identification", "Purpose", "NDA Type",
    "Definition of Confidential Information", "Confidentiality Obligations",
    "Authorized Disclosure", "Non-Confidential Information",
    "Liability for Damages", "Competition Rights", "Term and Termination",
    "Intellectual Property", "Employees",
    "Governing Law and Jurisdiction", "Additional Information",
)

def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]

def bool_parse(v: str) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}

def main() -> None:
    queue   = load_csv(ROOT / "data/annotations/annotation_queue.csv")
    seed    = load_csv(ROOT / "data/annotations/active_learning_seed.csv")
    anndb   = load_csv(ROOT / "data/annotations/annotations.csv")
    next_al = load_csv(ROOT / "data/annotations/next_active_learning_batch.csv")

    # ── annotation queue stats ───────────────────────────────────────────────
    total_queue = len(queue)

    # ── seed stats ────────────────────────────────────────────────────────────
    seed_ids = {r["clause_id"] for r in seed}
    seed_total = len(seed_ids)

    # Determine seed annotation progress from the queue CSV (source of truth)
    queue_by_id = {r["clause_id"]: r for r in queue}
    seed_annotated = sum(
        1 for cid in seed_ids
        if cid in queue_by_id and
           queue_by_id[cid].get("annotation_status", "PENDING") not in ("PENDING", "")
           and json.loads(queue_by_id[cid].get("category_labels") or "[]")
    )

    # ── annotations.csv stats ────────────────────────────────────────────────
    db_annotated = [r for r in anndb if r.get("annotation_status") == "ANNOTATED"]
    db_verified  = [r for r in anndb if bool_parse(r.get("verified", "false"))]
    db_disputed  = [r for r in anndb if r.get("annotation_status") == "DISPUTED"]
    db_skipped   = [r for r in anndb if r.get("annotation_status") == "SKIPPED"]

    # ── category distribution (from annotations.csv only) ───────────────────
    cat_dist: Counter = Counter()
    multi_label = 0
    for r in anndb:
        try:
            labels = json.loads(r.get("category_labels") or "[]")
        except (json.JSONDecodeError, TypeError):
            labels = []
        if labels:
            cat_dist.update(labels)
            if len(labels) > 1:
                multi_label += 1

    # ── round detection ──────────────────────────────────────────────────────
    rounds = set(r.get("annotation_round", "0") for r in anndb if r.get("annotation_round"))
    current_round = max((int(r) for r in rounds if str(r).isdigit()), default=0)

    # ── next batch ───────────────────────────────────────────────────────────
    next_batch_pending = sum(1 for r in next_al
                             if r.get("annotation_status", "PENDING") == "PENDING")

    # ── baseline model ────────────────────────────────────────────────────────
    baseline_path = ROOT / "models/active_learning_baseline"
    baseline_exists = baseline_path.exists() and any(baseline_path.iterdir())

    # ── next action logic ─────────────────────────────────────────────────────
    total_annotated = len(anndb)
    if total_annotated == 0:
        next_action = "Complete initial seed annotation (100 clauses) in app/annotation_app.py --mode seed"
    elif seed_annotated < seed_total:
        next_action = f"Continue seed annotation — {seed_total - seed_annotated} seed clauses remaining"
    elif not baseline_exists:
        next_action = "Run: python scripts/run_active_learning_round.py --round 1"
    elif next_batch_pending > 0:
        next_action = f"Annotate {next_batch_pending} uncertain clauses in the annotation interface"
    else:
        next_action = "Run: python scripts/run_active_learning_round.py --round (next)"

    # ── print ────────────────────────────────────────────────────────────────
    SEP = "-" * 55
    print()
    print("=" * 55)
    print("  ACTIVE LEARNING STATUS — NDA Clause Annotation")
    print("=" * 55)
    print()
    print(f"  Total queue clauses:         {total_queue}")
    print(f"  Seed set size:               {seed_total}")
    print(f"  Seed annotated:              {seed_annotated} / {seed_total}")
    print()
    print(f"  Human annotated (total):     {total_annotated}")
    print(f"  Verified:                    {len(db_verified)}")
    print(f"  Disputed:                    {len(db_disputed)}")
    print(f"  Skipped:                     {len(db_skipped)}")
    print(f"  Multi-label clauses:         {multi_label}")
    print(f"  Remaining unlabeled:         {total_queue - total_annotated}")
    print()
    print(f"  Current round:               {current_round}")
    print(f"  Baseline model trained:      {'YES' if baseline_exists else 'NO'}")
    print(f"  Next AL batch pending:       {next_batch_pending}")
    print()
    print(SEP)
    print("  CATEGORIES COVERED")
    print(SEP)
    max_count = max(cat_dist.values()) if cat_dist else 1
    for cat in CATEGORIES:
        count = cat_dist.get(cat, 0)
        bar = "|" * count
        warn = ""
        if count == 0:    warn = "  [ZERO]"
        elif count < 5:   warn = "  [SPARSE]"
        print(f"  {cat:<42} {count:>4}{warn}")
    print()
    print(SEP)
    print("  NEXT ACTION")
    print(SEP)
    print(f"  {next_action}")
    print()

    if total_annotated == 0:
        print("  NOTE: No classifier training until genuine human")
        print("        annotations exist.")
    print("=" * 55)
    print()

if __name__ == "__main__":
    main()
