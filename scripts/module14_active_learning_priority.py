"""
scripts/module14_active_learning_priority.py

Module 14 — Heuristic Active-Learning Priority Scoring Script.

Ranks unlabeled candidate clauses using a deterministic heuristic priority score
based on:
  1. Underrepresented category textual signals (Liability, Competition, IP, Governing Law)
  2. Document diversity & unseen document status
  3. Category redundancy need
  4. Clause informativeness / length (word count)
  5. Absence of duplicate clause IDs

NOTE: This script computes a 'heuristic active-learning priority' score.
It does NOT use model uncertainty because no final classifier training has occurred.

Usage:
  python scripts/module14_active_learning_priority.py --help
  python scripts/module14_active_learning_priority.py
"""

import argparse
import json
import re
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

UNDERREPRESENTED_SIGNALS = {
    "Liability for Damages": [r"\bdamage", r"\bliabil", r"\bindemn", r"\bloss", r"\bbreach", r"\bpenalty", r"\bremedy", r"\bremedies"],
    "Competition Rights": [r"\bcompet", r"\bsolicit", r"\brival", r"\bexclusive", r"\bentic"],
    "Intellectual Property": [r"\bpatent", r"\bcopyright", r"\btrademark", r"\btrade secret", r"\binvention", r"\bintellectual property", r"\bproprietary", r"\bownership", r"\blicense"],
    "Governing Law and Jurisdiction": [r"\bgovern", r"\bjurisdiction", r"\bvenue", r"\barbitrat", r"\bcourt", r"\blaws of", r"\bstate of"]
}


def compute_active_learning_priorities(
    queue_file: Path,
    labels_file: Path,
    output_queue_file: Path
) -> pd.DataFrame:
    print("=======================================================")
    print("  MODULE 14 — HEURISTIC ACTIVE-LEARNING PRIORITY RANKING")
    print("=======================================================")

    df_labels = pd.read_csv(labels_file)
    completed_clause_ids = set(df_labels["clause_id"].unique())

    df_valid = df_labels[df_labels["pseudo_label_quality"].isin([
        "HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE", "DISAGREEMENT"
    ])].copy()
    valid_doc_ids = set(df_valid["document_id"].unique())

    df_queue = pd.read_csv(queue_file)
    print(f"Total rows in source queue: {len(df_queue)}")

    # Deduplicate queue by clause_id and filter out already completed clauses
    df_pending = df_queue.drop_duplicates(subset=["clause_id"]).copy()
    df_pending = df_pending[~df_pending["clause_id"].isin(completed_clause_ids)].copy()
    print(f"Total unique pending clauses: {len(df_pending)}")

    all_queue_docs = set(df_pending["document_id"].unique())
    unseen_doc_ids = set(all_queue_docs) - valid_doc_ids

    print(f"Represented documents in queue: {len(all_queue_docs.intersection(valid_doc_ids))}")
    print(f"Completely unseen documents in queue: {len(unseen_doc_ids)}")

    scored_clauses = []

    for _, row in df_pending.iterrows():
        doc_id = row["document_id"]
        clause_id = row["clause_id"]
        txt = str(row["clause_text"])
        txt_lower = txt.lower()

        words = txt.split()
        word_count = len(words)

        is_unseen = doc_id in unseen_doc_ids
        doc_status = "Completely Unseen" if is_unseen else "Represented"

        suspected_cats = []
        matched_terms = []

        for cat, patterns in UNDERREPRESENTED_SIGNALS.items():
            cat_matches = []
            for pat in patterns:
                found = re.findall(pat, txt_lower)
                if found:
                    cat_matches.extend(found)
            if cat_matches:
                suspected_cats.append(cat)
                matched_terms.extend(cat_matches)

        distinct_signals = len(suspected_cats)
        total_term_matches = len(matched_terms)

        # Heuristic Active Learning Score calculation:
        # 1. Unseen Document Bonus: +30 pts
        # 2. Distinct Underrepresented Category Signals: +15 pts per distinct category
        # 3. Term Match Intensity: +2 pts per term match (max 20 pts)
        # 4. Clause Informativeness / Word Length: +0.05 pts per word (max 10 pts)
        unseen_bonus = 30.0 if is_unseen else 0.0
        signal_bonus = 15.0 * distinct_signals
        term_bonus = min(20.0, 2.0 * total_term_matches)
        len_bonus = min(10.0, 0.05 * word_count)

        priority_score = round(unseen_bonus + signal_bonus + term_bonus + len_bonus, 2)

        unique_matched_terms = sorted(list(set(matched_terms)))

        scored_clauses.append({
            "document_id": doc_id[:8],
            "full_document_id": doc_id,
            "clause_id": clause_id,
            "clause_text": txt,
            "priority_score": priority_score,
            "candidate_category_signal": "; ".join(suspected_cats) if suspected_cats else "General Context",
            "signal_terms": "; ".join(unique_matched_terms) if unique_matched_terms else "None",
            "document_status": doc_status,
            "label_source": "heuristic_candidate_only"
        })

    df_scored = pd.DataFrame(scored_clauses).sort_values(by="priority_score", ascending=False)
    df_scored["priority_rank"] = range(1, len(df_scored) + 1)

    # Reorder output columns according to spec
    output_cols = [
        "priority_rank",
        "document_id",
        "clause_id",
        "clause_text",
        "priority_score",
        "candidate_category_signal",
        "signal_terms",
        "document_status",
        "label_source"
    ]
    df_output = df_scored[output_cols]

    output_queue_file.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_queue_file, index=False)

    print(f"\nSuccessfully scored and saved targeted clause queue ({len(df_output)} clauses) to:")
    print(f"  {output_queue_file}\n")

    print("Top 5 Highest Priority Candidates:")
    print(df_output[["priority_rank", "document_id", "priority_score", "candidate_category_signal", "document_status"]].head(5).to_string(index=False))

    return df_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute heuristic active-learning priority scores for unlabeled NDA candidate clauses."
    )
    parser.add_argument(
        "--queue-file",
        type=Path,
        default=ROOT / "data/annotations/pseudolabel_resume_queue.csv",
        help="Input source queue CSV file"
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=ROOT / "data/annotations/llm_pseudo_labels.csv",
        help="Existing pseudo-labels CSV file"
    )
    parser.add_argument(
        "--output-queue",
        type=Path,
        default=ROOT / "data/annotations/module14_final_targeted_queue.csv",
        help="Output final targeted queue CSV file"
    )
    args = parser.parse_args()

    compute_active_learning_priorities(
        queue_file=args.queue_file,
        labels_file=args.labels_file,
        output_queue_file=args.output_queue
    )
