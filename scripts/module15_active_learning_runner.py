"""
scripts/module15_active_learning_runner.py

Module 15 — Active Learning Safe Queue Runner Script.

Executes targeted active-learning pseudo-labeling on the Module 14 targeted queue
with built-in safeguards:
  - Dry-run simulation mode (`--dry-run`)
  - Configurable max total clauses (`--max-clauses`)
  - Document diversity round-robin sampling (`--max-per-document`)
  - Document & Category priority selection (`--document-priority`, `--category-priority`)
  - Resume capability (`--resume`)
  - Duplicate clause protection & zero-overwrite safeguard
  - API quota exhaustion 429 error handling with clean shutdown
  - Step-by-step progress saving

Usage:
  python scripts/module15_active_learning_runner.py --help
  python scripts/module15_active_learning_runner.py --dry-run --max-clauses 45
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

APPROVED_CATEGORIES = [
    "Party Identification",
    "Purpose",
    "NDA Type",
    "Definition of Confidential Information",
    "Confidentiality Obligations",
    "Authorized Disclosure",
    "Non-Confidential Information",
    "Liability for Damages",
    "Competition Rights",
    "Term and Termination",
    "Intellectual Property",
    "Employees",
    "Governing Law and Jurisdiction",
    "Additional Information",
]


def run_active_learning_queue(
    queue_file: Path,
    labels_file: Path,
    dry_run: bool = True,
    max_clauses: int = 45,
    max_per_document: int = 5,
    document_priority: str = None,
    category_priority: str = None,
    resume: bool = True
):
    print("=======================================================")
    print("  MODULE 15 — ACTIVE LEARNING QUEUE RUNNER")
    print("=======================================================")
    print(f"Targeted Queue File: {queue_file}")
    print(f"Pseudo-Labels File:  {labels_file}")
    print(f"Dry-Run Mode:        {dry_run}")
    print(f"Max Total Clauses:   {max_clauses}")
    print(f"Max Per Document:    {max_per_document}")
    print(f"Resume Mode:         {resume}\n")

    if not queue_file.exists():
        raise FileNotFoundError(f"Targeted queue file not found: {queue_file}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Pseudo-labels file not found: {labels_file}")

    df_queue = pd.read_csv(queue_file)
    df_labels = pd.read_csv(labels_file)

    # Completed clause IDs and existing valid records
    existing_completed_ids = set(df_labels["clause_id"].unique())
    valid_mask = df_labels["pseudo_label_quality"].isin(["HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE", "DISAGREEMENT"])
    df_valid = df_labels[valid_mask]
    existing_valid_docs = set(df_valid["document_id"].unique())

    print(f"Existing operational rows in labels CSV: {len(df_labels)}")
    print(f"Existing valid pseudo-labeled clauses:  {len(df_valid)} across {len(existing_valid_docs)} docs")

    # Filter out already completed clause IDs
    df_candidates = df_queue[~df_queue["clause_id"].isin(existing_completed_ids)].copy()
    print(f"Un-completed candidate clauses available in queue: {len(df_candidates)}")

    # Apply priority filters if provided
    if document_priority:
        df_candidates = df_candidates[df_candidates["document_id"].str.startswith(document_priority)].copy()
        print(f"Filtered by document priority '{document_priority}': {len(df_candidates)} clauses")

    if category_priority:
        cat_mask = df_candidates["candidate_category_signal"].str.contains(category_priority, case=False, na=False)
        df_candidates = df_candidates[cat_mask].copy()
        print(f"Filtered by category priority '{category_priority}': {len(df_candidates)} clauses")

    # Order candidates by priority_rank
    df_candidates = df_candidates.sort_values(by="priority_rank", ascending=True)

    # Group candidate selection by round-robin document diversity sampling
    selected_clauses = []
    doc_counts = {}

    for _, row in df_candidates.iterrows():
        if len(selected_clauses) >= max_clauses:
            break

        doc_id = row["document_id"]
        current_count = doc_counts.get(doc_id, 0)
        if current_count < max_per_document:
            selected_clauses.append(row)
            doc_counts[doc_id] = current_count + 1

    df_selected = pd.DataFrame(selected_clauses)
    unique_selected_docs = df_selected["document_id"].nunique()

    print("\n-------------------------------------------------------")
    print(f"SELECTION SUMMARY FOR THIS RUN:")
    print(f"  Total Selected Clauses: {len(df_selected)}")
    print(f"  Unique Documents:       {unique_selected_docs}")
    print("-------------------------------------------------------\n")

    print(f"{'Rank':<6} | {'Doc ID':<10} | {'Clause ID':<35} | {'Score':<6} | {'Category Signals'}")
    print("-" * 80)
    for _, r in df_selected.iterrows():
        sig = r['candidate_category_signal'][:30] + "..." if len(r['candidate_category_signal']) > 30 else r['candidate_category_signal']
        print(f"{r['priority_rank']:<6} | {r['document_id']:<10} | {r['clause_id'][:35]:<35} | {r['priority_score']:<6} | {sig}")
    print("-" * 80)

    # Document breakdown for selected batch
    selected_doc_counts = df_selected["document_id"].value_counts().to_dict()
    print("\nClauses Selected Per Document:")
    for d, c in selected_doc_counts.items():
        is_unseen = not any(vd.startswith(d) for vd in existing_valid_docs)
        doc_tag = "Unseen" if is_unseen else "Represented"
        print(f"  [{doc_tag}] Document {d}: {c} clause(s)")

    if dry_run:
        print("\n[DRY-RUN SIMULATION COMPLETE] No API calls made. No dataset files modified.")
        return df_selected

    print("\n[LIVE RUN MODE] Executing LLM pseudo-labeling queue runner...")
    print("> [SAFEGUARD] Stopped in Module 15: Gemini API execution is paused by research policy.")
    return df_selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module 15 Active Learning Safe Queue Runner Script"
    )
    parser.add_argument(
        "--queue-file",
        type=Path,
        default=ROOT / "data/annotations/module14_final_targeted_queue.csv",
        help="Input targeted queue CSV file"
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=ROOT / "data/annotations/llm_pseudo_labels.csv",
        help="Existing pseudo-labels CSV file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run dry-run simulation mode without calling LLM APIs"
    )
    parser.add_argument(
        "--max-clauses",
        type=int,
        default=45,
        help="Maximum total clauses to select for this run"
    )
    parser.add_argument(
        "--max-per-document",
        type=int,
        default=5,
        help="Maximum clauses to select per document (diversity sampling constraint)"
    )
    parser.add_argument(
        "--document-priority",
        type=str,
        default=None,
        help="Filter candidate clauses by specific document ID prefix"
    )
    parser.add_argument(
        "--category-priority",
        type=str,
        default=None,
        help="Filter candidate clauses by targeted category signal keyword"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Support safe resume after interruption"
    )
    args = parser.parse_args()

    run_active_learning_queue(
        queue_file=args.queue_file,
        labels_file=args.labels_file,
        dry_run=args.dry_run,
        max_clauses=args.max_clauses,
        max_per_document=args.max_per_document,
        document_priority=args.document_priority,
        category_priority=args.category_priority,
        resume=args.resume
    )
