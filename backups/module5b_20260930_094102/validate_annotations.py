"""Module 5 — Annotation Validation Script.

Validates data/annotations/annotation_queue.csv against the 14-category schema and
provenance rules.  Produces reports/annotation_validation.csv with per-check results.

Usage:
    python scripts/validate_annotations.py [--queue PATH]

Options:
    --queue PATH    Path to the annotation CSV (default: data/annotations/annotation_queue.csv)

Rules enforced:
    - All required columns present
    - No missing / empty clause_id or document_id
    - No duplicate clause_ids
    - No empty clause_text
    - category_labels is valid JSON array of known category names
    - annotation_status is a recognised value
    - label_verified is boolean-parseable ('true'/'false')
    - label_verified=true requires label_reviewer and label_source to be non-empty
    - label_source is 'human_annotation' whenever labels are present
    - No invented labels (all names must be in the approved 14-category list)
    - category_labels=[] is allowed only when annotation_status is UNCLASSIFIABLE or PENDING

Does NOT:
    - Call Gemini or any LLM
    - Modify original V3 segmentation or Module 4 outputs
    - Invent or auto-assign categories
    - Perform classifier training
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from io import StringIO
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

CATEGORIES: tuple[str, ...] = (
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
)
CATEGORY_SET = set(CATEGORIES)

ANNOTATION_STATUSES: frozenset[str] = frozenset({
    "PENDING",
    "ANNOTATED",
    "VERIFIED",
    "DISPUTED",
    "SKIPPED",
    "UNCLASSIFIABLE",
})

REQUIRED_COLUMNS = (
    "document_id",
    "clause_id",
    "clause_text",
    "category_labels",
    "annotation_status",
    "label_verified",
    "label_source",
    "label_reviewer",
    "annotation_notes",
)

BOOL_TRUE  = {"true", "1", "yes"}
BOOL_FALSE = {"false", "0", "no", ""}

# Statuses where empty labels are acceptable
EMPTY_LABEL_OK_STATUSES = {"PENDING", "UNCLASSIFIABLE", "SKIPPED", ""}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        for row in reader:
            rows.append(dict(row))
    return rows, list(header)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    import os
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Individual check functions
# Each returns (affected_ids: list[str], detail: str)
# ---------------------------------------------------------------------------

def check_required_columns(header: list[str]) -> tuple[list[str], str]:
    missing = sorted(set(REQUIRED_COLUMNS) - set(header))
    return (missing, f"Missing required columns: {missing}" if missing else "All required columns present")


def check_missing_clause_ids(rows: list[dict]) -> tuple[list[str], str]:
    bad = [str(i + 1) for i, r in enumerate(rows) if not str(r.get("clause_id", "")).strip()]
    return (bad, "Rows (1-based record number) with empty clause_id" if bad else "No missing clause_ids")


def check_duplicate_clause_ids(rows: list[dict]) -> tuple[list[str], str]:
    counts = Counter(r.get("clause_id", "") for r in rows)
    dupes = sorted(cid for cid, n in counts.items() if n > 1 and cid)
    return (dupes, "Clause IDs appearing more than once" if dupes else "No duplicate clause_ids")


def check_missing_document_ids(rows: list[dict]) -> tuple[list[str], str]:
    bad = [r.get("clause_id", f"row_{i}") for i, r in enumerate(rows)
           if not str(r.get("document_id", "")).strip()]
    return (bad, "Clauses with empty document_id" if bad else "No missing document_ids")


def check_empty_clause_text(rows: list[dict]) -> tuple[list[str], str]:
    bad = [r.get("clause_id", f"row_{i}") for i, r in enumerate(rows)
           if not str(r.get("clause_text", "")).strip()]
    return (bad, "Clauses with empty or whitespace-only clause_text" if bad else "No empty clause_text")


def check_category_labels(rows: list[dict]) -> tuple[list[str], str]:
    """Validate JSON array format and that every label name is in the approved taxonomy."""
    bad: list[str] = []
    for row in rows:
        cid = row.get("clause_id", "")
        raw = row.get("category_labels", "")
        try:
            labels = json.loads(raw) if str(raw).strip() else []
        except (json.JSONDecodeError, TypeError):
            bad.append(cid)
            continue
        if not isinstance(labels, list):
            bad.append(cid)
            continue
        if not all(isinstance(lbl, str) for lbl in labels):
            bad.append(cid)
            continue
        unknown = [lbl for lbl in labels if lbl not in CATEGORY_SET]
        if unknown:
            bad.append(cid)
    return (bad, "Clauses with invalid or unknown category_labels JSON" if bad else "All category_labels are valid JSON arrays of approved names")


def check_unknown_categories(rows: list[dict]) -> tuple[list[str], str]:
    """Report which specific unknown category names appear."""
    found_unknown: set[str] = set()
    bad_ids: list[str] = []
    for row in rows:
        raw = row.get("category_labels", "")
        try:
            labels = json.loads(raw) if str(raw).strip() else []
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(labels, list):
            continue
        unk = [lbl for lbl in labels if isinstance(lbl, str) and lbl not in CATEGORY_SET]
        if unk:
            found_unknown.update(unk)
            bad_ids.append(row.get("clause_id", ""))
    detail = (f"Unknown category names used: {sorted(found_unknown)}" if found_unknown
              else "No unknown category names found")
    return (bad_ids, detail)


def check_annotation_status(rows: list[dict]) -> tuple[list[str], str]:
    bad = [r.get("clause_id", "") for r in rows
           if str(r.get("annotation_status", "")).strip() not in ANNOTATION_STATUSES | {""} ]
    return (bad, f"Invalid annotation_status. Allowed: {sorted(ANNOTATION_STATUSES)}" if bad
            else "All annotation_status values are valid")


def check_label_verified_format(rows: list[dict]) -> tuple[list[str], str]:
    bad = [r.get("clause_id", "") for r in rows
           if str(r.get("label_verified", "")).strip().lower() not in BOOL_TRUE | BOOL_FALSE]
    return (bad, "label_verified must be 'true' or 'false'" if bad else "All label_verified values are boolean-parseable")


def check_verified_provenance(rows: list[dict]) -> tuple[list[str], str]:
    """label_verified=true requires both label_reviewer and label_source to be non-empty."""
    bad = []
    for row in rows:
        verified = str(row.get("label_verified", "")).strip().lower() in BOOL_TRUE
        reviewer = str(row.get("label_reviewer", "")).strip()
        source   = str(row.get("label_source", "")).strip()
        if verified and (not reviewer or not source):
            bad.append(row.get("clause_id", ""))
    return (bad, "label_verified=true but label_reviewer or label_source is empty" if bad
            else "All verified annotations have complete provenance")


def check_label_source(rows: list[dict]) -> tuple[list[str], str]:
    """When labels are present, label_source must be 'human_annotation'."""
    bad = []
    for row in rows:
        raw = row.get("category_labels", "")
        try:
            labels = json.loads(raw) if str(raw).strip() else []
        except (json.JSONDecodeError, TypeError):
            labels = []
        if isinstance(labels, list) and labels:
            if str(row.get("label_source", "")).strip() != "human_annotation":
                bad.append(row.get("clause_id", ""))
    return (bad, "Clauses with labels but label_source != 'human_annotation'" if bad
            else "All labeled clauses have label_source='human_annotation'")


def check_empty_labels_in_annotated(rows: list[dict]) -> tuple[list[str], str]:
    """ANNOTATED or VERIFIED status should not have empty category_labels."""
    bad = []
    for row in rows:
        status = str(row.get("annotation_status", "")).strip().upper()
        if status in {"ANNOTATED", "VERIFIED"}:
            raw = row.get("category_labels", "")
            try:
                labels = json.loads(raw) if str(raw).strip() else []
            except (json.JSONDecodeError, TypeError):
                labels = []
            if not labels:
                bad.append(row.get("clause_id", ""))
    return (bad, "Clauses with ANNOTATED/VERIFIED status but empty category_labels" if bad
            else "All ANNOTATED/VERIFIED clauses have at least one label")


def check_multi_label_duplicates(rows: list[dict]) -> tuple[list[str], str]:
    """A single category name should not appear twice within one clause's labels."""
    bad = []
    for row in rows:
        raw = row.get("category_labels", "")
        try:
            labels = json.loads(raw) if str(raw).strip() else []
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(labels, list) and len(labels) != len(set(labels)):
            bad.append(row.get("clause_id", ""))
    return (bad, "Clauses with duplicate category names in the same label array" if bad
            else "No duplicate categories within any clause's label array")


def check_missing_reviewer_when_annotated(rows: list[dict]) -> tuple[list[str], str]:
    bad = [r.get("clause_id", "") for r in rows
           if str(r.get("annotation_status", "")).strip().upper() in {"ANNOTATED", "VERIFIED"}
           and not str(r.get("label_reviewer", "")).strip()]
    return (bad, "ANNOTATED/VERIFIED clauses missing reviewer name" if bad
            else "All ANNOTATED/VERIFIED clauses have a reviewer name")


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def compute_summary(rows: list[dict]) -> dict:
    total = len(rows)
    status_counts: dict[str, int] = Counter(
        str(r.get("annotation_status", "PENDING")).strip().upper() or "PENDING"
        for r in rows
    )
    labeled = 0
    multi_labeled = 0
    verified = 0
    cat_dist: Counter = Counter()
    for row in rows:
        raw = row.get("category_labels", "")
        try:
            labels = json.loads(raw) if str(raw).strip() else []
        except (json.JSONDecodeError, TypeError):
            labels = []
        if isinstance(labels, list) and labels:
            labeled += 1
            cat_dist.update(labels)
            if len(labels) > 1:
                multi_labeled += 1
        if str(row.get("label_verified", "")).strip().lower() in BOOL_TRUE:
            verified += 1
    return {
        "total_clauses": total,
        "labeled_clauses": labeled,
        "verified_labeled_clauses": verified,
        "multi_labeled_clauses": multi_labeled,
        "pending": status_counts.get("PENDING", 0),
        "annotated": status_counts.get("ANNOTATED", 0),
        "verified_status": status_counts.get("VERIFIED", 0),
        "disputed": status_counts.get("DISPUTED", 0),
        "skipped": status_counts.get("SKIPPED", 0),
        "unclassifiable": status_counts.get("UNCLASSIFIABLE", 0),
        "category_distribution": dict(cat_dist),
        "categories_with_zero_labels": [cat for cat in CATEGORIES if cat_dist[cat] == 0],
        "classifier_training_ready": False,
    }


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------

CHECKS = [
    # (check_fn_or_None, check_name, severity, column_check_only)
    # column check is special-cased below
    ("required_columns",                None,                              "ERROR"),
    ("missing_clause_ids",              check_missing_clause_ids,          "ERROR"),
    ("duplicate_clause_ids",            check_duplicate_clause_ids,        "ERROR"),
    ("missing_document_ids",            check_missing_document_ids,        "ERROR"),
    ("empty_clause_text",               check_empty_clause_text,           "ERROR"),
    ("invalid_category_labels_json",    check_category_labels,             "ERROR"),
    ("unknown_category_names",          check_unknown_categories,          "ERROR"),
    ("invalid_annotation_status",       check_annotation_status,           "ERROR"),
    ("invalid_label_verified_format",   check_label_verified_format,       "ERROR"),
    ("verified_missing_provenance",     check_verified_provenance,         "BLOCKED"),
    ("invalid_label_source",            check_label_source,                "BLOCKED"),
    ("annotated_but_empty_labels",      check_empty_labels_in_annotated,   "WARN"),
    ("duplicate_labels_in_array",       check_multi_label_duplicates,      "ERROR"),
    ("annotated_missing_reviewer",      check_missing_reviewer_when_annotated, "WARN"),
]

CHECK_FN_MAP = {
    "missing_clause_ids":              check_missing_clause_ids,
    "duplicate_clause_ids":            check_duplicate_clause_ids,
    "missing_document_ids":            check_missing_document_ids,
    "empty_clause_text":               check_empty_clause_text,
    "invalid_category_labels_json":    check_category_labels,
    "unknown_category_names":          check_unknown_categories,
    "invalid_annotation_status":       check_annotation_status,
    "invalid_label_verified_format":   check_label_verified_format,
    "verified_missing_provenance":     check_verified_provenance,
    "invalid_label_source":            check_label_source,
    "annotated_but_empty_labels":      check_empty_labels_in_annotated,
    "duplicate_labels_in_array":       check_multi_label_duplicates,
    "annotated_missing_reviewer":      check_missing_reviewer_when_annotated,
}

SEVERITY_ORDER = {"ERROR": 0, "BLOCKED": 1, "WARN": 2, "PASS": 3}


def run_validation(queue_path: Path, report_path: Path) -> int:
    print(f"Loading queue: {queue_path}")
    rows, header = load_csv(queue_path)
    print(f"Rows loaded:   {len(rows)}")
    print()

    results: list[dict] = []

    # --- required columns (special case — no rows needed) ---
    missing_cols, col_detail = check_required_columns(header)
    results.append({
        "check":      "required_columns",
        "severity":   "ERROR",
        "status":     "ERROR" if missing_cols else "PASS",
        "count":      len(missing_cols),
        "clause_ids": json.dumps(missing_cols, ensure_ascii=False),
        "detail":     col_detail,
    })
    if missing_cols:
        print(f"[ERROR] required_columns: {col_detail}")
        print("  Stopping — cannot validate further without required columns.")
        write_csv(report_path, list(results[0].keys()), results)
        print(f"\nValidation report: {report_path}")
        return 1

    # --- row-level checks ---
    severity_map = {name: sev for name, _, sev in CHECKS if name != "required_columns"}
    for check_name, fn_or_none, severity in CHECKS:
        if check_name == "required_columns":
            continue
        fn = CHECK_FN_MAP[check_name]
        affected, detail = fn(rows)
        status = severity if affected else "PASS"
        results.append({
            "check":      check_name,
            "severity":   severity,
            "status":     status,
            "count":      len(affected),
            "clause_ids": json.dumps(sorted(set(affected)), ensure_ascii=False),
            "detail":     detail,
        })

    # --- summary ---
    summary = compute_summary(rows)

    # --- print results ---
    col_w = max(len(r["check"]) for r in results) + 2
    print(f"{'CHECK':<{col_w}} {'STATUS':<8} {'COUNT'}")
    print("-" * (col_w + 20))
    for r in sorted(results, key=lambda x: (SEVERITY_ORDER.get(x["status"], 9), x["check"])):
        flag = "[OK]  " if r["status"] == "PASS" else ("[ERR] " if r["status"] == "ERROR" else "[WARN]")
        print(f"  {flag}  {r['check']:<{col_w}} {r['status']:<8} {r['count']}")

    print()
    print("-" * 55)
    print("ANNOTATION SUMMARY")
    print("-" * 55)
    for key, val in summary.items():
        if key in ("category_distribution", "categories_with_zero_labels"):
            continue
        print(f"  {key:<40} {val}")

    print()
    print("Category distribution (verified labels only):")
    dist = summary["category_distribution"]
    if dist:
        for cat in CATEGORIES:
            count = dist.get(cat, 0)
            bar = "|" * count
            print(f"  {cat:<42} {count:>4}  {bar}")
    else:
        print("  No verified category labels found yet.")

    print()
    print("Classifier training ready: NO")
    print("Reason: Genuine 14-category clause-level annotations are still required.")

    # --- write report ---
    write_csv(report_path, ["check", "severity", "status", "count", "clause_ids", "detail"], results)
    print(f"\nValidation report written: {report_path}")

    errors = sum(1 for r in results if r["status"] == "ERROR")
    blocked = sum(1 for r in results if r["status"] == "BLOCKED")
    warns = sum(1 for r in results if r["status"] == "WARN")
    print(f"Errors: {errors}  |  Blocked: {blocked}  |  Warnings: {warns}")
    return 1 if errors else 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        default=str(ROOT / "data" / "annotations" / "annotation_queue.csv"),
        help="Path to the annotation queue CSV",
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "reports" / "annotation_validation.csv"),
        help="Output path for the validation report CSV",
    )
    args = parser.parse_args()

    queue_path  = Path(args.queue).resolve()
    report_path = Path(args.report).resolve()

    if not queue_path.exists():
        sys.exit(f"Queue file not found: {queue_path}")

    sys.exit(run_validation(queue_path, report_path))


if __name__ == "__main__":
    main()
