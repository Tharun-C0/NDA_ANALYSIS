"""
scripts/targeted_expansion_plan.py

Module 11 — Diversity-First Active Learning Batch Selection Strategy.

Generates a deterministic priority plan for pseudo-label queue execution prioritizing document diversity.

Rules:
  1. Prioritize documents with ZERO valid labels first (Tier 1).
  2. Prefer broad document coverage by taking a small initial sampling batch (e.g. 5 clauses) per document.
  3. Do NOT exhaust a single document while other unseen documents remain untouched.
  4. Preserve original queue ordering within each document.
  5. Never modify existing valid labels or re-select completed clauses.
  6. Do NOT attempt to predict categories for unseen clauses (Diversity First, Discovery Second).

Usage:
  python scripts/targeted_expansion_plan.py --dry-run
  python scripts/targeted_expansion_plan.py --batch-size 5
"""

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build_targeted_expansion_plan(
    queue_path: Path,
    labels_path: Path,
    clauses_per_doc: int = 5,
    max_total_clauses: int = 45,
    dry_run: bool = True
):
    print("=======================================================")
    print("  MODULE 11 — DIVERSITY-FIRST EXPANSION PLAN")
    print("=======================================================")
    print(f"Queue File: {queue_path}")
    print(f"Labels File: {labels_path}")
    print(f"Sampling Target Per Document: {clauses_per_doc} clauses")
    print(f"Max Total Batch Clauses: {max_total_clauses}")
    print(f"Dry Run Mode: {dry_run}\n")

    if not queue_path.exists() or not labels_path.exists():
        raise FileNotFoundError("Missing required queue or pseudo-labels CSV files!")

    df_queue = pd.read_csv(queue_path)
    df_labels = pd.read_csv(labels_path)

    # Valid label IDs
    valid_mask = df_labels["pseudo_label_quality"].isin([
        "HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE", "DISAGREEMENT"
    ])
    df_valid = df_labels[valid_mask]
    completed_clause_ids = set(df_labels["clause_id"].unique())

    # Count valid labels per document
    valid_counts_by_doc = df_valid["document_id"].value_counts().to_dict()

    # Filter queue to un-completed clauses
    df_pending = df_queue[~df_queue["clause_id"].isin(completed_clause_ids)].copy()
    print(f"Total pending clauses in queue: {len(df_pending)}")

    # Categorize document tiers in queue
    unique_pending_docs = df_pending["document_id"].unique()

    tier1_unseen = []
    tier2_sparse = []
    tier3_represented = []

    for doc_id in unique_pending_docs:
        valid_cnt = valid_counts_by_doc.get(doc_id, 0)
        available_cnt = len(df_pending[df_pending["document_id"] == doc_id])
        if valid_cnt == 0:
            tier1_unseen.append((doc_id, valid_cnt, available_cnt))
        elif valid_cnt < 5:
            tier2_sparse.append((doc_id, valid_cnt, available_cnt))
        else:
            tier3_represented.append((doc_id, valid_cnt, available_cnt))

    print("\n-------------------------------------------------------")
    print("DOCUMENT DIVERSITY TIER SUMMARY:")
    print(f"  Tier 1 — Completely Unseen Documents (0 valid labels): {len(tier1_unseen)}")
    print(f"  Tier 2 — Sparse Documents (<5 valid labels)         : {len(tier2_sparse)}")
    print(f"  Tier 3 — Represented Documents (>=5 valid labels)   : {len(tier3_represented)}")
    print("-------------------------------------------------------\n")

    print(f"{'Document ID':<12} | {'Tier':<8} | {'Current Valid':<13} | {'Clauses Avail':<13} | {'Recommended Batch'}")
    print("-" * 75)

    recommended_batches = []
    total_selected = 0

    # Round-robin selection order across Tier 1, then Tier 2, then Tier 3
    priority_docs = tier1_unseen + tier2_sparse + tier3_represented

    for doc_id, valid_cnt, avail_cnt in priority_docs:
        if total_selected >= max_total_clauses:
            break

        doc_pending = df_pending[df_pending["document_id"] == doc_id]
        take_count = min(clauses_per_doc, len(doc_pending), max_total_clauses - total_selected)
        if take_count <= 0:
            continue

        selected_clauses = doc_pending.head(take_count)
        recommended_batches.append((doc_id, valid_cnt, avail_cnt, take_count, selected_clauses))
        total_selected += take_count

        tier_label = "Tier 1" if valid_cnt == 0 else ("Tier 2" if valid_cnt < 5 else "Tier 3")
        print(f"{doc_id[:8]:<12} | {tier_label:<8} | {valid_cnt:<13} | {avail_cnt:<13} | {take_count} clause(s)")

    print("-" * 75)
    print(f"TOTAL RECOMMENDED EXPANSION BATCH: {total_selected} clauses across {len(recommended_batches)} documents\n")

    if dry_run:
        print("[DRY RUN COMPLETE] No queue files were modified.")
    return recommended_batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diversity-first targeted expansion plan generator")
    parser.add_argument("--queue-file", type=Path, default=ROOT / "data/annotations/pseudolabel_resume_queue.csv")
    parser.add_argument("--labels-file", type=Path, default=ROOT / "data/annotations/llm_pseudo_labels.csv")
    parser.add_argument("--per-doc", type=int, default=5, help="Clauses to select per unseen document")
    parser.add_argument("--max-total", type=int, default=45, help="Total clauses for next expansion batch")
    parser.add_argument("--dry-run", action="store_true", help="Print recommended batch without modifying files")
    args = parser.parse_args()

    build_targeted_expansion_plan(
        queue_path=args.queue_file,
        labels_path=args.labels_file,
        clauses_per_doc=args.per_doc,
        max_total_clauses=args.max_total,
        dry_run=args.dry_run
    )
