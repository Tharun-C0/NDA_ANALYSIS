"""Module 5B — Active Learning Round Manager.

Orchestrates the full active learning cycle for one round:
    1. Validate annotation database
    2. Update category distribution report
    3. Train baseline model (if enough data)
    4. Run uncertainty sampling to produce next annotation batch
    5. Update research log

Usage:
    python scripts/run_active_learning_round.py --round 1
    python scripts/run_active_learning_round.py --round 2

Does NOT annotate anything automatically.
Every label must come from the human annotation interface.
"""
from __future__ import annotations

import argparse, csv, json, os, subprocess, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

CATEGORIES = [
    "Party Identification", "Purpose", "NDA Type",
    "Definition of Confidential Information", "Confidentiality Obligations",
    "Authorized Disclosure", "Non-Confidential Information",
    "Liability for Damages", "Competition Rights", "Term and Termination",
    "Intellectual Property", "Employees",
    "Governing Law and Jurisdiction", "Additional Information",
]

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
    os.replace(tmp, path)

def run_script(script: str, extra_args: list[str] = []) -> int:
    cmd = [PYTHON, str(ROOT / "scripts" / script)] + extra_args
    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode

# ---------------------------------------------------------------------------
def update_category_distribution(round_n: int, ann_db: list[dict]) -> None:
    """Rewrite the category distribution CSV and MD report."""
    # Load existing CSV to get previous round columns
    dist_path = ROOT / "reports/active_learning_category_distribution.csv"
    existing = load_csv(dist_path)
    # Build per-round column name
    round_col = f"round_{round_n}"

    # Compute current totals
    cat_dist: Counter = Counter()
    cat_docs: dict[str, set] = {c: set() for c in CATEGORIES}
    for r in ann_db:
        try:
            labels = json.loads(r.get("category_labels") or "[]")
        except (json.JSONDecodeError, TypeError):
            labels = []
        for lbl in labels:
            if lbl in cat_dist or lbl in [c for c in CATEGORIES]:
                cat_dist[lbl] += 1
                cat_docs.get(lbl, set()).add(r.get("document_id", ""))

    total = sum(cat_dist.values()) or 1

    # Build new rows preserving old round columns
    old_cols = [c for c in (existing[0].keys() if existing else [])
                if c.startswith("round_") and c != round_col]
    old_round_data = {r["category"]: r for r in existing}

    new_rows = []
    for cat in CATEGORIES:
        count = cat_dist.get(cat, 0)
        warn = ("ZERO_EXAMPLES" if count == 0 else
                "SPARSE" if count < 5 else
                "UNDERREPRESENTED" if count < total * 0.1 else "")
        row = {
            "category": cat,
            "total_annotations": count,
            "percentage": round(count / total * 100, 2),
            "documents_present": len(cat_docs.get(cat, set())),
        }
        # preserve previous round columns
        for col in old_cols:
            row[col] = old_round_data.get(cat, {}).get(col, 0)
        row[round_col] = count
        row["warning"] = warn
        new_rows.append(row)

    all_cols = ["category", "total_annotations", "percentage", "documents_present"] + \
               old_cols + [round_col, "warning"]
    write_csv(dist_path, all_cols, new_rows)

    # Update MD report
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = f"# Active Learning Category Distribution — Module 5B\n\n"
    md += f"*Last updated: {now} (Round {round_n})*\n\n"
    md += f"> Updated by `scripts/run_active_learning_round.py --round {round_n}`\n\n"
    md += "## Current Distribution\n\n"
    header = "| Category | Annotations | % | Docs |" + "".join(f" R{i} |" for i in range(1, round_n+1)) + " Warning |\n"
    sep    = "|---|---|---|---|" + "".join("---|" for _ in range(round_n)) + "---|\n"
    md += header + sep
    for row in new_rows:
        r_cols = "".join(f" {row.get(f'round_{i}', 0)} |" for i in range(1, round_n+1))
        md += f"| {row['category']} | {row['total_annotations']} | {row['percentage']} | {row['documents_present']} |{r_cols} {row['warning']} |\n"
    md += "\n## Imbalance Warnings\n\n"
    zeros   = [r["category"] for r in new_rows if r["total_annotations"] == 0]
    sparse  = [r["category"] for r in new_rows if 0 < r["total_annotations"] < 5]
    if zeros:
        md += f"> **ZERO_EXAMPLES** — {', '.join(zeros)}\n\n"
    if sparse:
        md += f"> **SPARSE (< 5)** — {', '.join(sparse)}\n\n"
    if not zeros and not sparse:
        md += "> No critical imbalance detected at current annotation count.\n\n"
    md += "Labels are never artificially balanced.\n"
    (ROOT / "reports/active_learning_category_distribution.md").write_text(md, encoding="utf-8")

# ---------------------------------------------------------------------------
def update_research_log(round_n: int, ann_db: list[dict], model_trained: bool,
                        n_selected: int, uncertainty_method: str) -> None:
    log_path = ROOT / "reports/active_learning_research_log.md"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    cat_dist: Counter = Counter()
    for r in ann_db:
        try:
            labels = json.loads(r.get("category_labels") or "[]")
        except (json.JSONDecodeError, TypeError):
            labels = []
        cat_dist.update(labels)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry  = f"\n---\n\n## Round {round_n} — {now}\n\n"
    entry += f"| Metric | Value |\n|---|---|\n"
    entry += f"| Annotation round | {round_n} |\n"
    entry += f"| Total annotated clauses | {len(ann_db)} |\n"
    entry += f"| Categories with examples | {sum(1 for c in CATEGORIES if cat_dist[c] > 0)} |\n"
    entry += f"| Baseline model trained | {'YES' if model_trained else 'NO'} |\n"
    entry += f"| Uncertainty method | {uncertainty_method} |\n"
    entry += f"| New batch selected | {n_selected} clauses |\n\n"
    entry += "**Category distribution this round:**\n\n"
    for cat in CATEGORIES:
        entry += f"- {cat}: {cat_dist.get(cat, 0)}\n"

    # Append to log
    if "---" in existing:
        updated = existing + entry
    else:
        updated = existing + "\n" + entry
    log_path.write_text(updated, encoding="utf-8")

# ---------------------------------------------------------------------------
def run_round(round_n: int, min_examples: int, n_select: int) -> int:
    print(f"\n{'='*55}")
    print(f"  ACTIVE LEARNING ROUND {round_n}")
    print(f"{'='*55}")

    ann_db = load_csv(ROOT / "data/annotations/annotations.csv")
    valid  = [r for r in ann_db
              if r.get("annotation_status") in ("ANNOTATED", "VERIFIED")
              and r.get("label_source", "").strip() == "human_annotation"]

    print(f"\n  Valid human annotations: {len(valid)}")
    if len(valid) < min_examples:
        print(f"\n[BLOCKED] Not enough annotations to start Round {round_n}.")
        print(f"  Need at least {min_examples} annotated clauses.")
        print(f"  Current: {len(valid)}")
        print(f"\n  Next action: complete seed annotation in annotation interface.")
        print(f"  Command: .venv\\Scripts\\python.exe app\\annotation_app.py --mode seed")
        return 1

    # Step 1: validate
    print("\n  [1/4] Running annotation validation...")
    rc = run_script("validate_annotations.py")
    if rc != 0:
        print("  Validation failed. Fix errors before proceeding.", file=sys.stderr)
        return 1

    # Step 2: update category distribution
    print("\n  [2/4] Updating category distribution...")
    update_category_distribution(round_n, valid)

    # Step 3: train baseline
    print("\n  [3/4] Training baseline model...")
    rc = run_script("train_active_learning_baseline.py",
                    ["--round", str(round_n), "--min-examples", str(min_examples)])
    model_trained = (rc == 0)
    if not model_trained:
        print("  Baseline training failed or skipped (insufficient data).")

    # Step 4: uncertainty sampling
    n_selected = 0
    if model_trained:
        print("\n  [4/4] Running uncertainty sampling...")
        rc = run_script("select_uncertain_clauses.py",
                        ["--round", str(round_n), "--n-select", str(n_select)])
        if rc == 0:
            batch = load_csv(ROOT / "data/annotations/next_active_learning_batch.csv")
            n_selected = len(batch)

    # Step 5: update research log
    uncertainty_method = "mean_margin_from_0.5" if model_trained else "N/A"
    update_research_log(round_n, valid, model_trained, n_selected, uncertainty_method)

    print(f"\n{'='*55}")
    print(f"  ROUND {round_n} COMPLETE")
    print(f"{'='*55}")
    print(f"  Annotations used:     {len(valid)}")
    print(f"  Model trained:        {'YES' if model_trained else 'NO'}")
    print(f"  New batch selected:   {n_selected} clauses")
    if n_selected > 0:
        print(f"\n  NEXT STEP: Annotate {n_selected} uncertain clauses:")
        print(f"  .venv\\Scripts\\python.exe app\\annotation_app.py --mode batch")
    return 0

# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True, dest="round_n",
                        help="Active learning round number (1, 2, 3, …)")
    parser.add_argument("--min-examples", type=int, default=10)
    parser.add_argument("--n-select", type=int, default=50,
                        help="Number of uncertain clauses to select (default: 50)")
    args = parser.parse_args()
    sys.exit(run_round(args.round_n, args.min_examples, args.n_select))

if __name__ == "__main__":
    main()
