"""Module 5B — Uncertainty Sampling for Active Learning.

Uses the trained baseline model to predict labels for remaining UNLABELED clauses,
computes an uncertainty score, and selects the top-N most uncertain clauses as
candidates for the next human annotation batch.

Usage:
    python scripts/select_uncertain_clauses.py [--round N] [--n-select N] [--model-dir PATH]

Outputs:
    data/annotations/next_active_learning_batch.csv

IMPORTANT:
    Predicted labels are NEVER written to the annotation database.
    They are displayed as suggestions ONLY to assist the human annotator.
    The human annotator retains full authority to accept, reject, or modify.
"""
from __future__ import annotations

import argparse, csv, json, pickle, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = [
    "Party Identification", "Purpose", "NDA Type",
    "Definition of Confidential Information", "Confidentiality Obligations",
    "Authorized Disclosure", "Non-Confidential Information",
    "Liability for Damages", "Competition Rights", "Term and Termination",
    "Intellectual Property", "Employees",
    "Governing Law and Jurisdiction", "Additional Information",
]

# ---------------------------------------------------------------------------
def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]

def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    import os; os.replace(tmp, path)

def compute_uncertainty(proba_row: list[float]) -> float:
    """Mean margin from 0.5 across all labels — higher = more uncertain."""
    import math
    margins = [abs(p - 0.5) for p in proba_row]
    # Lower average margin = more uncertain (all predictions close to 0.5)
    # We invert so higher score = more uncertain
    avg_margin = sum(margins) / len(margins) if margins else 1.0
    return round(1.0 - avg_margin, 6)   # 1.0 = maximally uncertain, 0.0 = very certain

# ---------------------------------------------------------------------------
def select(round_n: int, n_select: int, model_dir: Path | None) -> int:
    # ── find model ────────────────────────────────────────────────────────────
    if model_dir is None:
        # Try latest round
        base = ROOT / "models/active_learning_baseline"
        if not base.exists():
            print("[BLOCKED] No trained model found.", file=sys.stderr)
            print("  Run: python scripts/train_active_learning_baseline.py --round 1", file=sys.stderr)
            return 1
        round_dirs = sorted(base.glob("round_*"), key=lambda p: int(p.name.split("_")[1]))
        if not round_dirs:
            print("[BLOCKED] No round directory found in models/active_learning_baseline/", file=sys.stderr)
            return 1
        model_dir = round_dirs[-1]
    model_path = model_dir / "model.pkl"
    meta_path  = model_dir / "metadata.json"
    if not model_path.exists():
        print(f"[BLOCKED] Model not found: {model_path}", file=sys.stderr)
        return 1
    print(f"Loading model: {model_path}")
    with model_path.open("rb") as f:
        model = pickle.load(f)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    # ── load already-annotated clause IDs ─────────────────────────────────────
    ann_db = load_csv(ROOT / "data/annotations/annotations.csv")
    annotated_ids = {r["clause_id"] for r in ann_db
                     if r.get("annotation_status") in ("ANNOTATED", "VERIFIED")}
    print(f"Already annotated: {len(annotated_ids)} clauses")

    # ── load queue — filter out annotated ─────────────────────────────────────
    queue = load_csv(ROOT / "data/annotations/annotation_queue.csv")
    unlabeled = [r for r in queue if r["clause_id"] not in annotated_ids]
    print(f"Unlabeled remaining: {len(unlabeled)} clauses")
    if not unlabeled:
        print("All clauses have been annotated — no uncertainty sampling needed.")
        return 0

    # ── predict ────────────────────────────────────────────────────────────────
    texts = [r["clause_text"] for r in unlabeled]
    print(f"Predicting {len(texts)} clauses...")
    try:
        proba_matrix = model.predict_proba(texts)   # shape: (n_samples, n_labels)
    except AttributeError:
        # Some sklearn pipelines need predict_proba on the OvR estimator
        proba_matrix = model.named_steps["clf"].predict_proba(
            model.named_steps["tfidf"].transform(texts)
        )
    print("Predictions complete.")

    # ── compute uncertainty ────────────────────────────────────────────────────
    import numpy as np
    scored = []
    for i, row in enumerate(unlabeled):
        proba_row = proba_matrix[i].tolist()
        unc = compute_uncertainty(proba_row)
        # predicted labels at threshold 0.5
        pred_labels = [CATEGORIES[j] for j, p in enumerate(proba_row) if p >= 0.5]
        scored.append({
            **row,
            "uncertainty_score": unc,
            "predicted_labels":  json.dumps(pred_labels, ensure_ascii=False),
            "selection_reason":  f"uncertainty_sampling; round={round_n}; score={unc}",
            "annotation_status": "PENDING",
        })

    # Sort by uncertainty descending
    scored.sort(key=lambda r: r["uncertainty_score"], reverse=True)

    # De-prioritise documents already well-represented in annotated set
    # (spread selection across documents for diversity within uncertainty tier)
    ann_doc_counts = defaultdict(int)
    for r in ann_db:
        ann_doc_counts[r["document_id"]] += 1

    top = scored[:n_select * 3]   # candidate pool
    selected = []
    seen_docs: dict[str, int] = defaultdict(int)
    # First pass: pick highest-uncertainty avoiding doc over-representation
    for r in top:
        if len(selected) >= n_select:
            break
        doc_id = r["document_id"]
        if seen_docs[doc_id] < 3:   # max 3 per doc in a single batch
            selected.append(r)
            seen_docs[doc_id] += 1
    # Fill remaining slots if needed
    if len(selected) < n_select:
        for r in scored[n_select * 3:]:
            if len(selected) >= n_select:
                break
            selected.append(r)

    print(f"Selected {len(selected)} uncertain clauses for annotation.")

    # ── write output ───────────────────────────────────────────────────────────
    COLS = ["clause_id", "document_id", "clause_text", "segment_type",
            "review_status", "uncertainty_score", "predicted_labels",
            "selection_reason", "annotation_status"]
    out_path = ROOT / "data/annotations/next_active_learning_batch.csv"
    write_csv(out_path, COLS, selected)
    print(f"Saved: {out_path}")
    print()
    print("IMPORTANT: predicted_labels are model suggestions ONLY.")
    print("           Do NOT use them as ground truth.")
    print("           Human annotator must review each clause independently.")
    return 0

# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, default=1, dest="round_n")
    parser.add_argument("--n-select", type=int, default=50)
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()
    sys.exit(select(args.round_n, args.n_select, args.model_dir))

if __name__ == "__main__":
    main()
