"""Module 5B — Active Learning Baseline Trainer.

Trains a TF-IDF + One-vs-Rest Logistic Regression multi-label classifier on
GENUINE human annotations only.  Never uses LLM pseudo-labels or unannotated
clauses as training examples.

Usage:
    python scripts/train_active_learning_baseline.py [--round N] [--min-examples N]

Outputs:
    models/active_learning_baseline/round_<N>/  (model artefacts)
    reports/active_learning_baseline.md          (training report)

Requirements:
    - data/annotations/annotations.csv must have at least --min-examples rows
      with annotation_status=ANNOTATED or VERIFIED and non-empty category_labels.
    - scikit-learn >= 1.0
"""
from __future__ import annotations

import argparse, csv, json, os, pickle, sys
from collections import Counter
from datetime import datetime, timezone
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
CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}
N_CATS = len(CATEGORIES)

MIN_EXAMPLES_DEFAULT = 10
MIN_CATS_DEFAULT = 3

# ---------------------------------------------------------------------------
def load_annotations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def parse_labels(raw: str) -> list[str]:
    try:
        labels = json.loads(raw) if str(raw).strip() else []
    except (json.JSONDecodeError, TypeError):
        labels = []
    return [lbl for lbl in labels if isinstance(lbl, str) and lbl in CAT_INDEX]

def bool_parse(v: str) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}

def encode_labels(labels: list[str]) -> list[int]:
    vec = [0] * N_CATS
    for lbl in labels:
        if lbl in CAT_INDEX:
            vec[CAT_INDEX[lbl]] = 1
    return vec

# ---------------------------------------------------------------------------
def check_training_preconditions(rows: list[dict], min_examples: int, min_cats: int) -> tuple[bool, str]:
    valid = [r for r in rows
             if r.get("annotation_status") in ("ANNOTATED", "VERIFIED")
             and r.get("label_source", "").strip() == "human_annotation"
             and parse_labels(r.get("category_labels", "[]"))]
    if not valid:
        return False, (f"No valid human annotations found. Need at least {min_examples} rows "
                       f"with annotation_status=ANNOTATED/VERIFIED and label_source=human_annotation.")
    if len(valid) < min_examples:
        return False, (f"Only {len(valid)} valid annotations found. Need at least {min_examples}. "
                       f"Complete more seed annotation first.")
    cats_seen = set(lbl for r in valid for lbl in parse_labels(r.get("category_labels","[]")))
    if len(cats_seen) < min_cats:
        return False, (f"Only {len(cats_seen)} categories have at least one example. "
                       f"Need at least {min_cats}. Annotate clauses from more categories.")
    return True, f"{len(valid)} valid annotations across {len(cats_seen)} categories."

# ---------------------------------------------------------------------------
def document_disjoint_split(rows: list[dict], val_fraction: float = 0.2) -> tuple[list, list]:
    """Split by document ID so no document appears in both train and validation."""
    from collections import defaultdict
    by_doc: dict[str, list] = defaultdict(list)
    for r in rows:
        by_doc[r["document_id"]].append(r)
    docs = sorted(by_doc.keys())
    # Sort docs by size (smallest go to val to use quota efficiently)
    docs_by_size = sorted(docs, key=lambda d: len(by_doc[d]))
    val_target = int(len(rows) * val_fraction)
    val_docs, val_count = set(), 0
    for d in docs_by_size:
        if val_count >= val_target:
            break
        val_docs.add(d)
        val_count += len(by_doc[d])
    train_rows = [r for d, rs in by_doc.items() for r in rs if d not in val_docs]
    val_rows   = [r for d, rs in by_doc.items() for r in rs if d in val_docs]
    return train_rows, val_rows

# ---------------------------------------------------------------------------
def train(round_n: int, min_examples: int, min_cats: int) -> int:
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.metrics import (
            f1_score, hamming_loss, classification_report
        )
        import numpy as np
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("Install with: pip install scikit-learn", file=sys.stderr)
        return 1

    ann_path = ROOT / "data/annotations/annotations.csv"
    rows = load_annotations(ann_path)
    ok, msg = check_training_preconditions(rows, min_examples, min_cats)
    if not ok:
        print(f"\n[BLOCKED] Cannot train baseline:\n  {msg}", file=sys.stderr)
        print("\nClassifier training ready: NO", file=sys.stderr)
        print("Reason: Genuine 14-category clause-level annotations are still required.", file=sys.stderr)
        return 1

    print(f"\n[OK] Preconditions met: {msg}")

    # Filter valid annotated rows
    valid = [r for r in rows
             if r.get("annotation_status") in ("ANNOTATED", "VERIFIED")
             and r.get("label_source", "").strip() == "human_annotation"
             and parse_labels(r.get("category_labels", "[]"))]

    # Document-disjoint split
    train_rows, val_rows = document_disjoint_split(valid, val_fraction=0.2)
    if not val_rows:
        # Not enough docs for a split — use all for training, report val as empty
        train_rows, val_rows = valid, []
        print("  WARNING: Too few documents for a document-disjoint validation split.")
        print("           Using all annotations for training. No validation metrics.")

    print(f"  Train: {len(train_rows)} clauses | Val: {len(val_rows)} clauses")
    train_docs = {r["document_id"] for r in train_rows}
    val_docs   = {r["document_id"] for r in val_rows}
    print(f"  Train docs: {len(train_docs)} | Val docs: {len(val_docs)}")

    X_train = [r["clause_text"] for r in train_rows]
    y_train = np.array([encode_labels(parse_labels(r["category_labels"])) for r in train_rows])

    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_features=20000,
            sublinear_tf=True, min_df=1,
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
            n_jobs=-1,
        )),
    ])

    print("  Training...")
    pipeline.fit(X_train, y_train)
    print("  Training complete.")

    # Save model
    model_dir = ROOT / f"models/active_learning_baseline/round_{round_n}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(pipeline, f)
    print(f"  Model saved: {model_path}")

    # Save category list for consistent prediction
    meta_path = model_dir / "metadata.json"
    meta = {
        "round": round_n, "categories": CATEGORIES,
        "n_train": len(train_rows), "n_val": len(val_rows),
        "train_docs": sorted(train_docs), "val_docs": sorted(val_docs),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Evaluate on validation set
    val_metrics: dict = {}
    cat_report_text = ""
    if val_rows:
        X_val = [r["clause_text"] for r in val_rows]
        y_val = np.array([encode_labels(parse_labels(r["category_labels"])) for r in val_rows])
        y_pred = pipeline.predict(X_val)
        val_metrics = {
            "macro_f1":    round(float(f1_score(y_val, y_pred, average="macro",  zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y_val, y_pred, average="weighted", zero_division=0)), 4),
            "micro_f1":    round(float(f1_score(y_val, y_pred, average="micro",  zero_division=0)), 4),
            "hamming_loss": round(float(hamming_loss(y_val, y_pred)), 4),
        }
        cat_report_text = classification_report(
            y_val, y_pred, target_names=CATEGORIES, zero_division=0
        )
        print(f"  Macro F1: {val_metrics['macro_f1']}  |  Micro F1: {val_metrics['micro_f1']}  |  Hamming loss: {val_metrics['hamming_loss']}")
    else:
        print("  No validation set — metrics not computed.")
        val_metrics = {"note": "No validation split possible at this sample size."}

    # ── category distribution in training data ────────────────────────────────
    cat_dist: Counter = Counter()
    for r in valid:
        cat_dist.update(parse_labels(r.get("category_labels", "[]")))
    sparse_cats = [c for c in CATEGORIES if cat_dist[c] < 5]
    zero_cats   = [c for c in CATEGORIES if cat_dist[c] == 0]

    # ── write report ──────────────────────────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"# Active Learning Baseline Report — Round {round_n}\n\n"
    report += f"*Generated: {now}*\n\n"
    report += "## Training Overview\n\n"
    report += f"| Metric | Value |\n|---|---|\n"
    report += f"| Round | {round_n} |\n"
    report += f"| Human-labeled clauses (total) | {len(valid)} |\n"
    report += f"| Training clauses | {len(train_rows)} |\n"
    report += f"| Validation clauses | {len(val_rows)} |\n"
    report += f"| Training documents | {len(train_docs)} |\n"
    report += f"| Validation documents | {len(val_docs)} |\n"
    report += f"| Feature extraction | TF-IDF word unigrams+bigrams (max 20k features) |\n"
    report += f"| Classifier | One-vs-Rest Logistic Regression (C=1.0, balanced) |\n\n"
    report += "## Validation Metrics\n\n"
    if val_rows:
        report += f"| Metric | Value |\n|---|---|\n"
        for k, v in val_metrics.items():
            report += f"| {k} | {v} |\n"
        report += "\n### Per-category report\n\n```\n" + cat_report_text + "\n```\n\n"
    else:
        report += "> No validation split available at this sample size.\n\n"
    report += "## Category Distribution (Training Set)\n\n"
    report += "| Category | Count | Warning |\n|---|---|---|\n"
    for cat in CATEGORIES:
        c = cat_dist[cat]
        warn = "ZERO" if c == 0 else ("SPARSE" if c < 5 else "")
        report += f"| {cat} | {c} | {warn} |\n"
    if zero_cats:
        report += f"\n> **Zero-example categories:** {', '.join(zero_cats)}\n"
        report += "> These categories were not predictable by the model.\n"
    if sparse_cats:
        report += f"\n> **Sparse categories (< 5 examples):** {', '.join(sparse_cats)}\n"
        report += "> Predictions for these categories are unreliable.\n"
    report += "\n## Important Notes\n\n"
    report += "- This model is used **only for uncertainty estimation** (active learning selection).\n"
    report += "- It is **not** the final research classifier.\n"
    report += "- Macro F1 at this sample size is expected to be low.\n"
    report += "- Do NOT interpret these metrics as the paper's final results.\n"

    report_path = ROOT / "reports/active_learning_baseline.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report written: {report_path}")
    return 0

# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, default=1, dest="round_n",
                        help="Active learning round number (default: 1)")
    parser.add_argument("--min-examples", type=int, default=MIN_EXAMPLES_DEFAULT,
                        help=f"Minimum annotated clauses required (default: {MIN_EXAMPLES_DEFAULT})")
    parser.add_argument("--min-cats", type=int, default=MIN_CATS_DEFAULT,
                        help=f"Minimum categories with at least one example (default: {MIN_CATS_DEFAULT})")
    args = parser.parse_args()
    sys.exit(train(args.round_n, args.min_examples, args.min_cats))

if __name__ == "__main__":
    main()
