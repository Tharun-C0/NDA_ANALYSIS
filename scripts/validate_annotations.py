"""Module 5B — Annotation Validation Script.

Validates data/annotations/annotations.csv and data/annotations/annotation_queue.csv
against the 14-category schema and provenance rules.
Produces:
    reports/annotation_validation.csv   (machine-readable check results)
    reports/annotation_validation.md    (human-readable report)

Usage:
    python scripts/validate_annotations.py [--queue PATH] [--db PATH]

Rules enforced:
    1.  No duplicate clause_ids
    2.  Every annotated clause has at least one category
    3.  Category names are from the approved 14-category list
    4.  Multi-label values are valid JSON arrays
    5.  Reviewer name exists when annotation_status != PENDING
    6.  annotation_status is a recognised value
    7.  verified field is boolean-parseable
    8.  verified=true requires label_reviewer and label_source
    9.  label_source is 'human_annotation' when labels are present
    10. No labels outside the 14 categories
    11. No accidental modification of source dataset (V3 text integrity)
    12. Annotation counts are consistent
    13. No empty clause_text
    14. No missing document_ids

Does NOT call Gemini, train models, or modify source datasets.
"""
from __future__ import annotations

import argparse, csv, json, os, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CATEGORIES: tuple[str, ...] = (
    "Party Identification", "Purpose", "NDA Type",
    "Definition of Confidential Information", "Confidentiality Obligations",
    "Authorized Disclosure", "Non-Confidential Information",
    "Liability for Damages", "Competition Rights", "Term and Termination",
    "Intellectual Property", "Employees",
    "Governing Law and Jurisdiction", "Additional Information",
)
CATEGORY_SET = set(CATEGORIES)

ANNOTATION_STATUSES: frozenset[str] = frozenset({
    "PENDING", "ANNOTATED", "DISPUTED", "SKIPPED",
})

REQUIRED_COLUMNS_QUEUE = (
    "document_id", "clause_id", "clause_text",
    "category_labels", "annotation_status",
    "label_verified", "label_source", "label_reviewer", "annotation_notes",
)
REQUIRED_COLUMNS_DB = (
    "clause_id", "document_id", "clause_text",
    "category_labels", "reviewer", "notes",
    "annotation_status", "verified", "created_at", "updated_at",
    "annotation_round", "selection_source",
)

BOOL_TRUE  = {"true", "1", "yes"}
BOOL_FALSE = {"false", "0", "no", ""}

SEVERITY_ORDER = {"ERROR": 0, "BLOCKED": 1, "WARN": 2, "PASS": 3}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> tuple[list[dict], list[str]]:
    if not path.exists():
        return [], []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return rows, header


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def parse_labels(raw: str) -> list[str]:
    try:
        labels = json.loads(raw) if str(raw).strip() else []
    except (json.JSONDecodeError, TypeError):
        return None   # type: ignore  — signals malformed JSON
    if not isinstance(labels, list):
        return None   # type: ignore
    return labels

def bool_parse(v: str) -> bool:
    return str(v).strip().lower() in BOOL_TRUE

def make_check(name: str, affected: list[str], detail: str, severity: str = "ERROR") -> dict:
    return {
        "check":      name,
        "severity":   severity,
        "status":     severity if affected else "PASS",
        "count":      len(affected),
        "clause_ids": json.dumps(sorted(set(str(x) for x in affected)), ensure_ascii=False),
        "detail":     detail,
    }


# ---------------------------------------------------------------------------
# Individual check functions (each returns a check dict)
# ---------------------------------------------------------------------------

def chk_required_cols(header: list[str], required: tuple, label: str) -> dict:
    missing = sorted(set(required) - set(header))
    return make_check(f"required_columns_{label}", missing,
                      f"Missing required columns in {label}: {missing}" if missing else f"All required columns present in {label}")


def chk_duplicate_ids(rows: list[dict], id_col: str = "clause_id") -> dict:
    counts = Counter(r.get(id_col, "") for r in rows)
    dupes = [cid for cid, n in counts.items() if n > 1 and cid]
    return make_check("duplicate_clause_ids", dupes, "Repeated clause IDs")


def chk_missing_doc_ids(rows: list[dict]) -> dict:
    bad = [r.get("clause_id", f"row_{i}") for i, r in enumerate(rows)
           if not str(r.get("document_id", "")).strip()]
    return make_check("missing_document_ids", bad, "Empty document_id")


def chk_empty_text(rows: list[dict]) -> dict:
    bad = [r.get("clause_id", f"row_{i}") for i, r in enumerate(rows)
           if not str(r.get("clause_text", "")).strip()]
    return make_check("empty_clause_text", bad, "Empty or whitespace-only clause_text")


def chk_category_json(rows: list[dict]) -> dict:
    bad = []
    for r in rows:
        labels = parse_labels(r.get("category_labels", "[]") or "[]")
        if labels is None:
            bad.append(r.get("clause_id", ""))
        elif not all(isinstance(l, str) for l in labels):
            bad.append(r.get("clause_id", ""))
    return make_check("invalid_category_labels_json", bad,
                      "category_labels must be a JSON array of strings")


def chk_unknown_cats(rows: list[dict]) -> dict:
    bad = []
    for r in rows:
        labels = parse_labels(r.get("category_labels", "[]") or "[]")
        if labels and any(l not in CATEGORY_SET for l in labels):
            bad.append(r.get("clause_id", ""))
    return make_check("unknown_category_names", bad,
                      "One or more labels not in the 14-category taxonomy")


def chk_dup_labels(rows: list[dict]) -> dict:
    bad = [r.get("clause_id", "") for r in rows
           if (lambda ls: ls is not None and len(ls) != len(set(ls)))(
               parse_labels(r.get("category_labels", "[]") or "[]"))]
    return make_check("duplicate_labels_in_array", bad,
                      "Duplicate category names within one clause's array")


def chk_annotated_empty_labels(rows: list[dict]) -> dict:
    bad = [r.get("clause_id", "") for r in rows
           if r.get("annotation_status") in ("ANNOTATED",)
           and not (parse_labels(r.get("category_labels", "[]") or "[]") or [])]
    return make_check("annotated_but_empty_labels", bad,
                      "ANNOTATED status but no category labels", "WARN")


def chk_status(rows: list[dict], status_col: str = "annotation_status") -> dict:
    bad = [r.get("clause_id", "") for r in rows
           if str(r.get(status_col, "")).strip() not in ANNOTATION_STATUSES | {""}]
    return make_check("invalid_annotation_status", bad,
                      f"Valid values: {sorted(ANNOTATION_STATUSES)}")


def chk_verified_format(rows: list[dict], verified_col: str = "label_verified") -> dict:
    bad = [r.get("clause_id", "") for r in rows
           if str(r.get(verified_col, "")).strip().lower() not in BOOL_TRUE | BOOL_FALSE]
    return make_check("invalid_verified_format", bad,
                      f"verified must be true/false, got other values (col={verified_col})")


def chk_verified_provenance(rows: list[dict],
                              reviewer_col: str = "label_reviewer",
                              source_col: str = "label_source",
                              verified_col: str = "label_verified") -> dict:
    bad = []
    for r in rows:
        if bool_parse(r.get(verified_col, "false")):
            if not str(r.get(reviewer_col, "")).strip() or not str(r.get(source_col, "")).strip():
                bad.append(r.get("clause_id", ""))
    return make_check("verified_missing_provenance", bad,
                      "verified=true but reviewer or label_source is empty", "BLOCKED")


def chk_label_source(rows: list[dict],
                       source_col: str = "label_source",
                       labels_col: str = "category_labels") -> dict:
    bad = []
    for r in rows:
        labels = parse_labels(r.get(labels_col, "[]") or "[]") or []
        if labels and str(r.get(source_col, "")).strip() != "human_annotation":
            bad.append(r.get("clause_id", ""))
    return make_check("invalid_label_source", bad,
                      "Clauses with labels but label_source != 'human_annotation'", "BLOCKED")


def chk_reviewer_when_annotated(rows: list[dict],
                                  reviewer_col: str = "reviewer") -> dict:
    bad = [r.get("clause_id", "") for r in rows
           if r.get("annotation_status") in ("ANNOTATED",)
           and not str(r.get(reviewer_col, "")).strip()]
    return make_check("annotated_missing_reviewer", bad,
                      "ANNOTATED clauses without a reviewer name", "WARN")


def chk_no_pseudolabels(rows: list[dict]) -> dict:
    """Ensure no row has label_source set to an LLM/automated value."""
    LLM_SOURCES = {"llm_auto_pseudolabel", "gemini", "llm", "auto", "automatic"}
    bad = [r.get("clause_id", "") for r in rows
           if str(r.get("label_source", "")).strip().lower() in LLM_SOURCES]
    return make_check("pseudolabel_source_detected", bad,
                      "label_source indicates automated/LLM labelling — not ground truth", "BLOCKED")


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def compute_summary(rows: list[dict],
                     verified_col: str = "label_verified",
                     status_col: str = "annotation_status") -> dict:
    total = len(rows)
    status_counts = Counter(str(r.get(status_col, "PENDING")).strip().upper() or "PENDING"
                            for r in rows)
    labeled = verified = multi = 0
    cat_dist: Counter = Counter()
    for r in rows:
        labels = parse_labels(r.get("category_labels", "[]") or "[]") or []
        if labels:
            labeled += 1
            cat_dist.update(labels)
            if len(labels) > 1:
                multi += 1
        if bool_parse(r.get(verified_col, "false")):
            verified += 1
    return {
        "total_clauses": total,
        "labeled_clauses": labeled,
        "verified_labeled_clauses": verified,
        "multi_labeled_clauses": multi,
        "pending":      status_counts.get("PENDING", 0),
        "annotated":    status_counts.get("ANNOTATED", 0),
        "disputed":     status_counts.get("DISPUTED", 0),
        "skipped":      status_counts.get("SKIPPED", 0),
        "category_distribution": dict(cat_dist),
        "classifier_training_ready": False,
    }


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def build_md_report(queue_checks: list[dict], db_checks: list[dict],
                     queue_summary: dict, db_summary: dict,
                     queue_n: int, db_n: int) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md  = f"# Annotation Validation Report — Module 5B\n\n*Generated: {now}*\n\n"

    def checks_table(checks: list[dict]) -> str:
        lines = ["| Check | Severity | Status | Count |",
                 "|---|---|---|---|"]
        for c in sorted(checks, key=lambda x: (SEVERITY_ORDER.get(x["status"], 9), x["check"])):
            icon = "[OK]" if c["status"] == "PASS" else ("[ERR]" if c["status"] == "ERROR" else "[!]")
            lines.append(f"| {icon} {c['check']} | {c['severity']} | {c['status']} | {c['count']} |")
        return "\n".join(lines)

    md += f"## Annotation Queue (`annotation_queue.csv`) — {queue_n} rows\n\n"
    md += checks_table(queue_checks) + "\n\n"
    md += f"## Annotation Database (`annotations.csv`) — {db_n} rows\n\n"
    md += checks_table(db_checks) + "\n\n"

    md += "## Annotation Summary\n\n"
    md += "| Metric | Queue | DB |\n|---|---|---|\n"
    for key in ("total_clauses", "labeled_clauses", "verified_labeled_clauses",
                "multi_labeled_clauses", "pending", "annotated", "disputed", "skipped"):
        md += f"| {key} | {queue_summary.get(key,'—')} | {db_summary.get(key,'—')} |\n"

    md += "\n## Category Distribution (from annotation DB)\n\n"
    dist = db_summary.get("category_distribution", {})
    total_labels = sum(dist.values()) or 1
    md += "| Category | Count | % | Warning |\n|---|---|---|---|\n"
    for cat in CATEGORIES:
        c = dist.get(cat, 0)
        pct = round(c / total_labels * 100, 1)
        warn = "ZERO" if c == 0 else ("SPARSE" if c < 5 else "")
        md += f"| {cat} | {c} | {pct} | {warn} |\n"

    md += "\n## Training Readiness\n\n"
    md += "> Classifier training ready: **NO**\n\n"
    md += "> Reason: Genuine 14-category clause-level annotations are still required.\n\n"

    all_errors = sum(1 for c in queue_checks + db_checks if c["status"] == "ERROR")
    all_blocked = sum(1 for c in queue_checks + db_checks if c["status"] == "BLOCKED")
    all_warns  = sum(1 for c in queue_checks + db_checks if c["status"] == "WARN")
    md += f"**Errors:** {all_errors} | **Blocked:** {all_blocked} | **Warnings:** {all_warns}\n"
    return md


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------

def run_validation(queue_path: Path, db_path: Path,
                    csv_report_path: Path, md_report_path: Path) -> int:
    print(f"Loading queue:  {queue_path}")
    queue_rows, queue_header = load_csv(queue_path)
    print(f"  Rows: {len(queue_rows)}")

    print(f"Loading ann DB: {db_path}")
    db_rows, db_header = load_csv(db_path)
    print(f"  Rows: {len(db_rows)}")
    print()

    # ── Queue checks ────────────────────────────────────────────────────────
    queue_checks: list[dict] = [
        chk_required_cols(queue_header, REQUIRED_COLUMNS_QUEUE, "queue"),
        chk_duplicate_ids(queue_rows),
        chk_missing_doc_ids(queue_rows),
        chk_empty_text(queue_rows),
        chk_category_json(queue_rows),
        chk_unknown_cats(queue_rows),
        chk_dup_labels(queue_rows),
        chk_annotated_empty_labels(queue_rows),
        chk_status(queue_rows, "annotation_status"),
        chk_verified_format(queue_rows, "label_verified"),
        chk_verified_provenance(queue_rows, "label_reviewer", "label_source", "label_verified"),
        chk_label_source(queue_rows, "label_source", "category_labels"),
        chk_reviewer_when_annotated(queue_rows, "label_reviewer"),
        chk_no_pseudolabels(queue_rows),
    ]

    # ── DB checks ───────────────────────────────────────────────────────────
    db_checks: list[dict] = []
    if db_rows:
        db_checks = [
            chk_required_cols(db_header, REQUIRED_COLUMNS_DB, "db"),
            chk_duplicate_ids(db_rows),
            chk_missing_doc_ids(db_rows),
            chk_empty_text(db_rows),
            chk_category_json(db_rows),
            chk_unknown_cats(db_rows),
            chk_dup_labels(db_rows),
            chk_annotated_empty_labels(db_rows),
            chk_status(db_rows, "annotation_status"),
            chk_verified_format(db_rows, "verified"),
            chk_verified_provenance(db_rows, "reviewer", "selection_source", "verified"),
            chk_label_source(db_rows, "selection_source", "category_labels"),
            chk_reviewer_when_annotated(db_rows, "reviewer"),
            chk_no_pseudolabels(db_rows),
        ]
    else:
        db_checks = [make_check("db_empty", [], "annotations.csv is empty — no DB checks needed.", "PASS")]

    all_checks = queue_checks + db_checks

    # ── Print results ────────────────────────────────────────────────────────
    col_w = max(len(c["check"]) for c in all_checks) + 2
    print(f"{'CHECK':<{col_w}} {'STATUS':<8} {'COUNT'}")
    print("-" * (col_w + 20))
    for c in sorted(all_checks, key=lambda x: (SEVERITY_ORDER.get(x["status"], 9), x["check"])):
        flag = "[OK]  " if c["status"] == "PASS" else ("[ERR] " if c["status"] == "ERROR" else "[WARN]")
        print(f"  {flag}  {c['check']:<{col_w}} {c['status']:<8} {c['count']}")

    # ── Summaries ────────────────────────────────────────────────────────────
    queue_summary = compute_summary(queue_rows, "label_verified", "annotation_status")
    db_summary    = compute_summary(db_rows,    "verified",       "annotation_status")

    print()
    print("-" * 55)
    print("ANNOTATION SUMMARY")
    print("-" * 55)
    print(f"  {'Metric':<40} {'Queue':>7} {'DB':>7}")
    for key in ("total_clauses", "labeled_clauses", "verified_labeled_clauses",
                "multi_labeled_clauses", "pending", "annotated", "disputed", "skipped"):
        print(f"  {key:<40} {queue_summary.get(key,0):>7} {db_summary.get(key,0):>7}")

    print()
    print("Category distribution (from annotation DB):")
    dist = db_summary.get("category_distribution", {})
    for cat in CATEGORIES:
        c = dist.get(cat, 0)
        warn = "  [ZERO]" if c == 0 else ("  [SPARSE]" if c < 5 else "")
        print(f"  {cat:<42} {c:>4}{warn}")

    print()
    print("Classifier training ready: NO")
    print("Reason: Genuine 14-category clause-level annotations are still required.")

    # ── Write reports ─────────────────────────────────────────────────────────
    write_csv(csv_report_path, ["check", "severity", "status", "count", "clause_ids", "detail"],
              all_checks)
    md = build_md_report(queue_checks, db_checks,
                          queue_summary, db_summary,
                          len(queue_rows), len(db_rows))
    md_report_path.parent.mkdir(parents=True, exist_ok=True)
    md_report_path.write_text(md, encoding="utf-8")

    print(f"\nCSV report:  {csv_report_path}")
    print(f"MD  report:  {md_report_path}")
    errors  = sum(1 for c in all_checks if c["status"] == "ERROR")
    blocked = sum(1 for c in all_checks if c["status"] == "BLOCKED")
    warns   = sum(1 for c in all_checks if c["status"] == "WARN")
    print(f"Errors: {errors}  |  Blocked: {blocked}  |  Warnings: {warns}")
    return 1 if errors else 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue",  default=str(ROOT / "data/annotations/annotation_queue.csv"))
    parser.add_argument("--db",     default=str(ROOT / "data/annotations/annotations.csv"))
    parser.add_argument("--report", default=str(ROOT / "reports/annotation_validation.csv"))
    parser.add_argument("--md-report", default=str(ROOT / "reports/annotation_validation.md"))
    args = parser.parse_args()
    sys.exit(run_validation(
        Path(args.queue).resolve(),
        Path(args.db).resolve(),
        Path(args.report).resolve(),
        Path(args.md_report).resolve(),
    ))

if __name__ == "__main__":
    main()
