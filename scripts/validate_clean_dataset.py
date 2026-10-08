"""Offline Module 4: build a provisional clean dataset, audit labels, and validate.

Run --build to create the requested Module 4 artifacts; without it, validate the
existing clean CSV. --self-test runs in-memory checks only. No Module 3A imports,
network clients, physical merges/splits, category inference, or training.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import lzma
import math
import os
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ("KEEP", "REMOVE_NON_LEGAL", "MERGE_WITH_NEXT", "MERGE_WITH_PREVIOUS", "SPLIT", "REVIEW")
CATEGORIES = (
    "Party Identification", "Purpose", "NDA Type", "Definition of Confidential Information",
    "Confidentiality Obligations", "Authorized Disclosure", "Non-Confidential Information",
    "Liability for Damages", "Competition Rights", "Term and Termination",
    "Intellectual Property", "Employees", "Governing Law and Jurisdiction", "Additional Information",
)
LABEL_FIELDS = {"label", "labels", "category", "categories", "class", "classes", "class_labels",
                "clause_labels", "clause_category", "clause_categories", "category_labels", "label_ids"}
REQUIRED = ("document_id", "clause_id", "clause_text", "source", "review_status", "review_action",
            "confidence", "review_reason")
STATUSES = {"UNREVIEWED", "PRIORITY_UNREVIEWED", "API_FAILURE", "CONFLICT_REVIEW",
            "PENDING_MERGE", "PENDING_SPLIT", "PENDING_REVIEW", "PSEUDO_KEEP", "HUMAN_RECORDED_KEEP"}
INPUT_NAMES = ("review_dataset.csv", "priority_review.csv", "llm_pseudo_labels.csv", "human_review_required.csv")
PROTECTED = ("data/segmentation", "data/segmentation_v2", "data/segmentation_v3", "data/human_review",
             "external_data/kleister-nda", "reports/human_review")
IGNORED_DIRS = {".git", ".venv", "venv", "env", ".pytest_cache", "__pycache__", "node_modules",
                ".qoder", ".idea", ".vscode"}


def truth(value):
    return str(value).strip().lower() in {"true", "1"}


def csv_rows(path):
    """Preserve quoted multiline text, including blank lines inside a clause."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next((row for row in reader if row), [])
        if len(header) != len(set(header)):
            raise ValueError(f"Duplicate CSV columns: {path}")
        result = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV width mismatch: {path}, record {len(result) + 1}")
            result.append(dict(zip(header, row)))
    return result, header


def csv_text(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    os.replace(temporary, path)


def local_files(root):
    for folder, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS and not Path(folder, d).is_symlink())
        for name in sorted(names):
            path = Path(folder, name)
            if not path.is_symlink():
                yield path


def fingerprint(root):
    result = {}
    for directory in PROTECTED:
        for path in local_files(root / directory):
            with path.open("rb") as stream:
                result[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def load_v3(root):
    result = {}
    for path in sorted((root / "data/segmentation_v3").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        doc_id = document["document_id"]
        if path.stem != doc_id or document["num_clauses"] != len(document["clauses"]):
            raise ValueError(f"Inconsistent V3 document: {path.name}")
        for clause in document["clauses"]:
            cid = clause["clause_id"]
            if cid in result or clause["document_id"] != doc_id or not cid.startswith(doc_id + "_clause_"):
                raise ValueError(f"Duplicate or inconsistent V3 identity: {cid}")
            result[cid] = {
                "document_id": doc_id, "clause_id": cid, "clause_text": clause["text"],
                "source": path.relative_to(root).as_posix(), "segment_type": clause["segment_type"],
                "start": clause["start"], "end": clause["end"], "is_suspicious": clause["is_suspicious"],
            }
    if not result:
        raise ValueError("No V3 clauses found")
    return result


def snapshot(root):
    reference = load_v3(root)
    inputs = {}
    grouped = defaultdict(list)
    for name in INPUT_NAMES:
        rows, _ = csv_rows(root / "data/human_review" / name)
        inputs[name] = rows
        seen = set()
        for row in rows:
            cid = row["clause_id"]
            if cid in seen or cid not in reference:
                raise ValueError(f"Duplicate or unknown review identity in {name}: {cid}")
            seen.add(cid)
            if any(row[key] != reference[cid][key] for key in ("document_id", "clause_text")):
                raise ValueError(f"Review text/document differs from V3: {name}, {cid}")
            grouped[cid].append(dict(row, review_file=f"data/human_review/{name}"))
    if {row["clause_id"] for row in inputs["review_dataset.csv"]} != set(reference):
        raise ValueError("review_dataset.csv does not cover the V3 source exactly")
    logs, _ = csv_rows(root / "reports/human_review/llm_review_log.csv")
    successes = {}
    conflicting_logs = set()
    for row in logs:
        cid = row["clause_id"]
        if not cid:
            continue
        if cid not in reference or row["document_id"] != reference[cid]["document_id"]:
            raise ValueError(f"Unknown or inconsistent log identity: {cid}")
        if row["status"] == "SUCCESS":
            value = float(row["confidence"])
            if row["suggested_action"] not in ACTIONS or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid logged suggestion: {cid}")
            if cid in successes and any(row[k] != successes[cid][k] for k in ("suggested_action", "confidence")):
                conflicting_logs.add(cid)
            successes[cid] = row
    stats, _ = csv_rows(root / "reports/human_review/llm_review_statistics.csv")
    return reference, inputs, grouped, logs, successes, conflicting_logs, stats


def resolve_clause(source, records, success=None, failed=False, priority=False, conflicting_log=False):
    """Resolve structural evidence conservatively; never assign a clause category."""
    humans = [row for row in records if truth(row.get("reviewed")) and row.get("human_action")]
    human_actions = {row["human_action"] for row in humans}
    action = success["suggested_action"] if success else ""
    evidence = []
    if success:
        evidence = [row for row in records if row["review_file"].endswith(
            ("llm_pseudo_labels.csv", "human_review_required.csv"))
            and row.get("ai_suggested_action") == action
            and row.get("ai_confidence") and float(row["ai_confidence"]) == float(success["confidence"])]
    conflict = (conflicting_log or (bool(success) and not evidence) or len(human_actions) > 1
                or bool(human_actions - set(ACTIONS)) or (bool(success) and bool(human_actions - {action}))
                or any(row.get("human_corrected_text", "").strip() for row in humans))
    if not success and len(human_actions) == 1:
        action = next(iter(human_actions))
    all_actions = human_actions | ({action} if action else set())
    merge = bool(all_actions & {"MERGE_WITH_NEXT", "MERGE_WITH_PREVIOUS"})
    split = "SPLIT" in all_actions
    pending_queue = any(row["review_file"].endswith("human_review_required.csv")
                        and not truth(row.get("reviewed"))
                        and (row.get("review_reason") != "API_FAILURE" or not success) for row in records)
    ai_reason = next((row.get("ai_reason", "") for row in evidence if row.get("ai_reason")), "")
    reasons = []
    if conflict:
        reasons.append("Conflicting/incomplete review evidence or unapplied corrected text; preserve V3 text for adjudication.")
        status, effective_action = "CONFLICT_REVIEW", "REVIEW"
    elif action == "REMOVE_NON_LEGAL":
        status, effective_action = "EXCLUDED_NON_LEGAL", action
        reasons.append("Provisional structural exclusion only; not legal or category ground truth.")
    elif merge:
        status, effective_action = "PENDING_MERGE", action
        reasons.append("Physical merge not applied; no validated review-to-merge mechanism established.")
    elif split:
        status, effective_action = "PENDING_SPLIT", action
        reasons.append("Physical split not applied; human boundary review required.")
    elif action == "REVIEW":
        status, effective_action = "PENDING_REVIEW", action
    elif action == "KEEP":
        status = "PSEUDO_KEEP" if success else "HUMAN_RECORDED_KEEP"
        effective_action = action
    elif failed:
        status, effective_action = "API_FAILURE", ""
        reasons.append("API failure; no successful suggestion exists. No action inferred.")
    else:
        status = "PRIORITY_UNREVIEWED" if priority else "UNREVIEWED"
        effective_action = ""
        reasons.append("No completed structural review; retained provisionally, not assumed KEEP.")
    if ai_reason:
        reasons.append("LLM reason: " + ai_reason)
    queue_reasons = sorted({r.get("review_reason", "") for r in records if r.get("review_reason")})
    reasons.extend(queue_reasons)
    if human_actions:
        reasons.append("Recorded human actions: " + ", ".join(sorted(human_actions)))
    provenance = [{key: row.get(key, "") for key in (
        "human_action", "human_corrected_text", "human_notes", "reviewer", "review_method",
        "review_timestamp", "review_file")} for row in humans]
    return {
        **source, "review_status": status, "review_action": effective_action,
        "confidence": success["confidence"] if success else "",
        "review_reason": " ".join(reasons) or "Human validation of the structural suggestion remains necessary.",
        "ai_suggested_action": success["suggested_action"] if success else "",
        "ai_confidence": success["confidence"] if success else "", "ai_reason": ai_reason,
        "confidence_basis": "LLM_SUGGESTION" if success else "",
        "human_action": "; ".join(sorted(human_actions)),
        "human_review_records": json.dumps(provenance, ensure_ascii=False),
        "review_source": json.dumps(sorted({r["review_file"] for r in records
                                            if r.get("human_action") or r in evidence}), ensure_ascii=False),
        "review_timestamp": success["timestamp"] if success else "",
        "llm_status": "SUCCESS" if success else ("ERROR" if failed else "NOT_REVIEWED"),
        "is_priority": priority, "needs_human_review": status != "EXCLUDED_NON_LEGAL" or pending_queue,
        "unresolved_merge": merge, "unresolved_split": split,
    }


def clean_rows(state):
    reference, inputs, grouped, logs, successes, conflicts, _ = state
    priority = {row["clause_id"] for row in inputs["priority_review.csv"]}
    failures = {row["clause_id"] for row in logs if row["status"] == "ERROR"} - set(successes)
    all_rows = [resolve_clause(source, grouped[cid], successes.get(cid), cid in failures,
                               cid in priority, cid in conflicts) for cid, source in reference.items()]
    retained = [row for row in all_rows if row["review_status"] != "EXCLUDED_NON_LEGAL"]
    removed = [row for row in all_rows if row["review_status"] == "EXCLUDED_NON_LEGAL"]
    return all_rows, retained, removed


def annotation_candidate(record):
    identity = any(record.get(k) for k in ("clause_id", "clause_text", "text", "content"))
    fields = [key for key in record if key.lower() in LABEL_FIELDS and record[key] not in (None, "", [], {})]
    return bool(identity and fields)


def nested_records(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from nested_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_records(child)


def audit_annotations(root):
    """Inspect local data, not examples/test fixtures, for actual label assignments."""
    candidates, unsupported = [], []
    scanned = Counter()
    doc_keys = set()
    expected_counts, compressed_counts = {}, {}
    excluded = {"tests", "data/cleaned"}
    for path in local_files(root):
        rel = path.relative_to(root).as_posix()
        if any(rel == prefix or rel.startswith(prefix + "/") for prefix in excluded):
            continue
        if path.name.startswith("test_") or path.name == "test_save.csv":
            continue
        # Code, configuration, documentation, PDFs, and credentials are not label tables.
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pkl", ".pickle", ".xlsx", ".zip", ".arrow", ".h5"}:
            unsupported.append(rel)
        elif suffix in {".csv", ".json", ".jsonl"}:
            scanned[suffix] += 1
            if suffix == ".csv":
                records, _ = csv_rows(path)
            elif suffix == ".json":
                records = nested_records(json.loads(path.read_text(encoding="utf-8")))
            else:
                records = (record for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
                           for record in nested_records(json.loads(line)))
            count = sum(annotation_candidate(row) for row in records)
            if count:
                candidates.append({"file": rel, "candidate_records": count})
        elif suffix == ".txt":
            scanned[suffix] += 1
            text = path.read_text(encoding="utf-8")
            blocks = re.findall(r"\[INIT_CLAUSE\](.*?)\[END_CLAUSE\]", text, re.DOTALL)
            count = sum(bool(re.search(r"\[INIT_CLASSE\]\s*\S.*?\[END_CLASSE\]", block, re.DOTALL))
                        for block in blocks)
            if count:
                candidates.append({"file": rel, "candidate_records": count})
        elif path.name == "in.tsv.xz" and rel.startswith("external_data/kleister-nda/"):
            with lzma.open(path, "rt", encoding="utf-8") as stream:
                count = 0
                for row in csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE):
                    if len(row) != 6:
                        raise ValueError(f"Unexpected Kleister input schema: {rel}")
                    count += 1
                    doc_keys.update(row[1].split())
            compressed_counts[rel] = count
        elif suffix == ".tsv":
            scanned[suffix] += 1
            text = path.read_text(encoding="utf-8")
            if path.name.startswith("expected") and rel.startswith("external_data/kleister-nda/"):
                lines = [line for line in text.splitlines() if line.strip()]
                expected_counts[rel] = len(lines)
                doc_keys.update(part.split("=", 1)[0] for line in lines for part in line.split() if "=" in part)
            elif path.name != "in-header.tsv":
                unsupported.append(rel)
    return {"scanned_formats": dict(scanned), "candidate_files": candidates, "unsupported_files": unsupported,
            "document_level_keys": sorted(doc_keys), "expected_document_rows": expected_counts,
            "compressed_document_rows": compressed_counts,
            "external_pdf_count": len(list((root / "external_data/kleister-nda/documents").glob("*.pdf")))}


def parse_labels(row):
    """Optional future labels must be explicit assignments, never inferred from text."""
    fields = [key for key in row if key.lower() in LABEL_FIELDS and str(row[key]).strip()]
    if not fields:
        return [], False
    if len(fields) != 1:
        raise ValueError("Ambiguous label columns")
    key = fields[0]
    value = row[key]
    if key.lower() in {"label", "category", "class", "clause_category"} and value in CATEGORIES:
        labels = [value]
    else:
        labels = json.loads(value)
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
        raise ValueError("Labels must be a nonempty JSON array of category names")
    if len(set(labels)) != len(labels) or set(labels) - set(CATEGORIES):
        raise ValueError("Duplicate or unknown categories")
    verified = truth(row.get("label_verified")) and bool(row.get("label_source", "").strip()) and bool(
        row.get("label_reviewer", "").strip())
    return labels, verified


def validate_rows(rows, columns, reference, expected_ids=None):
    checks = []

    def add(check, affected, detail, severity="ERROR"):
        checks.append({"check": check, "status": severity if affected else "PASS", "count": len(affected),
                       "clause_ids": json.dumps(sorted(set(affected)), ensure_ascii=False), "details": detail})

    add("required_columns", sorted(set(REQUIRED) - set(columns)), "Missing required CSV fields")
    ids = [row.get("clause_id", "") for row in rows]
    add("missing_clause_ids", [str(i + 1) for i, cid in enumerate(ids) if not cid.strip()], "CSV record numbers")
    counts = Counter(ids)
    add("duplicate_clause_ids", [cid for cid, count in counts.items() if count > 1], "Repeated identities")
    add("missing_clause_text", [row.get("clause_id", "") for row in rows if not row.get("clause_text")], "Absent or empty text")
    add("empty_clauses", [row.get("clause_id", "") for row in rows if row.get("clause_text") and not row["clause_text"].strip()], "Whitespace-only text")
    texts = Counter(" ".join(row.get("clause_text", "").split()) for row in rows if row.get("clause_text", "").strip())
    add("duplicate_text", [row["clause_id"] for row in rows if texts.get(" ".join(row.get("clause_text", "").split()), 0) > 1],
        "Whitespace-normalized duplicate text: all affected rows; retained for document provenance, not deduplicated.", "WARN")
    add("invalid_review_actions", [row.get("clause_id", "") for row in rows if row.get("review_action", "") not in ("",) + ACTIONS], "Allowed: six structural actions; blank only for unreviewed/API-failure rows")
    add("missing_review_status", [row.get("clause_id", "") for row in rows if not row.get("review_status", "").strip()], "Missing structural review state")
    add("invalid_review_status", [row.get("clause_id", "") for row in rows if row.get("review_status") not in STATUSES], "Unknown structural review state")
    add("action_status_consistency", [row.get("clause_id", "") for row in rows
        if (row.get("review_action") == "REMOVE_NON_LEGAL" or
            (not row.get("review_action") and row.get("review_status") not in {"UNREVIEWED", "PRIORITY_UNREVIEWED", "API_FAILURE"}))],
        "Excluded non-legal actions must not be in clean CSV; blank actions must not imply completion")
    add("inconsistent_document_ids", [row.get("clause_id", "") for row in rows
        if row.get("clause_id") not in reference or row.get("document_id") != reference[row["clause_id"]]["document_id"]], "Identity checked against original V3, not only ID prefix")
    add("source_text_integrity", [row.get("clause_id", "") for row in rows
        if row.get("clause_id") in reference and row.get("clause_text") != reference[row["clause_id"]]["clause_text"]], "No text edits or physical merge/split allowed")
    add("source_provenance", [row.get("clause_id", "") for row in rows
        if row.get("clause_id") in reference and row.get("source") != reference[row["clause_id"]]["source"]], "Source must match the actual V3 file")
    if expected_ids is not None:
        add("cleaning_coverage", sorted(set(ids) ^ set(expected_ids)), "Exact retained-ID reconciliation against source reviews")
    invalid_confidence = []
    for row in rows:
        try:
            value = row.get("confidence", "")
            if value != "" and (not math.isfinite(float(value)) or not 0 <= float(value) <= 1):
                raise ValueError("out of range")
        except (ValueError, TypeError):
            invalid_confidence.append(row.get("clause_id", ""))
    add("confidence_range", invalid_confidence, "Blank when unknown; otherwise finite number in [0, 1]")
    malformed, unverified = [], []
    distribution = Counter()
    labeled_ids, multi_ids = set(), set()
    for row in rows:
        try:
            labels, verified = parse_labels(row)
            if labels and not verified:
                unverified.append(row["clause_id"])
            if labels and verified and row.get("clause_id") in reference:
                if row["clause_id"] not in labeled_ids:
                    distribution.update(labels)
                labeled_ids.add(row["clause_id"])
                if len(labels) > 1:
                    multi_ids.add(row["clause_id"])
        except (ValueError, TypeError, json.JSONDecodeError):
            malformed.append(row.get("clause_id", ""))
    add("multi_label_format", malformed, "If populated: JSON list of distinct exact category names; scalar accepted only in a single-label column")
    add("label_provenance", unverified, "Assignments need label_verified=true, label_source, and label_reviewer", "BLOCKED")
    checks.append({"check": "label_availability", "status": "PASS" if labeled_ids else "BLOCKED", "count": len(labeled_ids),
                   "clause_ids": json.dumps(sorted(labeled_ids)), "details": "Verified clause-level category assignments; structural actions are not labels"})
    checks.append({"check": "multi_label_availability", "status": "PASS" if multi_ids else "BLOCKED", "count": len(multi_ids),
                   "clause_ids": json.dumps(sorted(multi_ids)), "details": "Verified clauses assigned more than one category; parser capability is not annotated data"})
    return checks, {"genuinely_labeled_clauses": len(labeled_ids), "multi_labeled_clauses": len(multi_ids),
                    "available_categories": sorted(distribution), "category_distribution": dict(distribution)}


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows])


def reports(state, all_rows, retained, removed, audit, checks, label_info, protected):
    reference, inputs, _, logs, successes, _, saved_stats = state
    actions = Counter(row["suggested_action"] for row in successes.values())
    confidences = [float(row["confidence"]) for row in successes.values()]
    priority_ids = {row["clause_id"] for row in inputs["priority_review.csv"]}
    failures = [row for row in logs if row["status"] in {"ERROR", "CONNECTIVITY_ERROR"}]
    conflicts = [row for row in all_rows if row["review_status"] == "CONFLICT_REVIEW"]
    old_humans = {row["clause_id"] for rows in inputs.values() for row in rows if truth(row.get("reviewed")) and row.get("human_action")}
    source_status = saved_stats[-1].get("run_status", "UNKNOWN") if saved_stats else "UNKNOWN"
    summary = {
        "total_documents": len({row["document_id"] for row in reference.values()}),
        "total_v3_segments": len(reference), "total_priority_segments": len(priority_ids),
        "successful_llm_reviewed_segments": len(successes),
        "unique_segments_with_llm_or_recorded_human_review": len(set(successes) | old_humans),
        "recorded_human_review_segments": len(old_humans),
        "priority_without_llm_success": len(priority_ids - set(successes)),
        **{action: actions[action] for action in ACTIONS},
        "high_confidence": sum(value >= 0.9 for value in confidences),
        "medium_confidence": sum(0.7 <= value < 0.9 for value in confidences),
        "low_confidence": sum(value < 0.7 for value in confidences), "api_failure_events": len(failures),
        "cleaned_candidate_clauses": len(retained), "removed_segments": len(removed),
        "conflicting_review_cases": len(conflicts),
        "unresolved_merge_cases": sum(row["unresolved_merge"] for row in retained),
        "unresolved_split_cases": sum(row["unresolved_split"] for row in retained),
        "unresolved_retained_cases": sum(row["needs_human_review"] for row in retained),
        "unresolved_priority_cases": sum(row["needs_human_review"] and row["is_priority"] for row in retained),
        "human_review_cases_including_excluded_audits": sum(row["needs_human_review"] for row in all_rows),
        "existing_human_review_queue": len(inputs["human_review_required.csv"]),
        **label_info, "module3a_saved_status": source_status, "module5_training_possible": False,
    }
    distribution = [{"action": action, "count": actions[action],
                     "percentage": round(actions[action] / len(successes) * 100, 2) if successes else 0,
                     "basis": "unique_SUCCESS_clause_ids_in_llm_review_log"} for action in ACTIONS]
    analysis = "# Module 3A Final Snapshot Analysis\n\n"
    analysis += (f"The user has closed Module 3A. Its persisted status is `{source_status}`, not a fully reviewed queue. "
                 "Module 4 uses only saved evidence; no Gemini calls or reruns of Modules 1, 2, or 3A.\n\n")
    analysis += table(["Metric", "Value"], list(summary.items())[:19]) + "\n\n"
    analysis += "## Counting and confidence definitions\n"
    analysis += ("Counts use unique SUCCESS clause IDs across the log, reconciled with both pseudo-label and human-review CSVs. "
                 "Human review records are reported separately and do not turn suggestions into category labels. "
                 "Confidence: high >= 0.90; medium >= 0.70 and < 0.90; low < 0.70. "
                 "API failures count persisted terminal ERROR/CONNECTIVITY_ERROR events, not recovered retry attempts. "
                 "API errors and unreviewed rows are never counted as AI REVIEW.\n\n")
    analysis += "## Input reconciliation\n" + table(["File", "Rows"], [(name, len(rows)) for name, rows in inputs.items()]) + "\n\n"
    analysis += "## Saved statistics (historical run, not recalculated coverage)\n"
    analysis += table(["Field", "Saved value"], list(saved_stats[-1].items()) if saved_stats else []) + "\n\n"
    analysis += "## Conservative resolution of conflicting actions\n"
    analysis += ("Older priority-queue records include reviewer `tester` and reason `test reason`. Their research provenance is not established. "
                 "They are not silently promoted to trusted human annotations or discarded. Any conflicting recorded human/LLM action "
                 "is retained with effective action REVIEW, original actions, and provenance. `test_save.csv` is a test artifact, not an authority. "
                 "This is why the effective removal count can be smaller than the LLM REMOVE_NON_LEGAL count.\n\n")
    analysis += table(["Clause ID", "LLM action", "Recorded human action"],
                      [(r["clause_id"], r["ai_suggested_action"], r["human_action"]) for r in conflicts]) + "\n\n"
    analysis += "## Provisional exclusions (reversible using unchanged V3)\n"
    analysis += table(["Clause ID", "Review action", "Still queued for human audit"],
                      [(r["clause_id"], r["review_action"], r["needs_human_review"]) for r in removed]) + "\n"

    readiness = "# Baseline Dataset Readiness — Module 4\n\n"
    readiness += "## Decision\nModule 5 classifier training cannot begin. No verified clause-level assignments to the paper's categories were found.\n\n"
    readiness += table(["Metric", "Value"], [(key, value) for key, value in summary.items()
                        if key not in ACTIONS and key not in {"category_distribution", "available_categories"}]) + "\n\n"
    readiness += "## Meaning of clean and unresolved\n"
    readiness += ("The cleaned CSV is a **provisional clause candidate dataset**, not a fully human-validated legal-clause corpus. "
                  "Unreviewed V3 segments are retained with blank review_action and UNREVIEWED/PRIORITY_UNREVIEWED status; "
                  "no implicit KEEP decisions are invented. API failures have API_FAILURE status. "
                  "Non-conflicting REMOVE_NON_LEGAL decisions exclude rows only from this derived CSV. "
                  "KEEP is retained; merge, split, REVIEW, and conflicting evidence are retained and flagged. "
                  "Duplicate text is reported, not dropped across documents. All clause text, IDs, and offsets remain those of V3.\n\n"
                  "Unresolved retained cases include unreviewed non-priority segments. Unresolved priority cases are a narrower subset. "
                  "Merge-involved counts include recorded human merge requests in conflicts as well as LLM merge suggestions; "
                  "these overlap conflict counts and must not be summed. Human-review cases include audits on excluded rows. "
                  "No physical merge/split is performed: segmenter heuristics are not a validated mechanism for applying review decisions.\n\n")
    readiness += table(["Retained review status", "Count"], sorted(Counter(r["review_status"] for r in retained).items())) + "\n\n"
    readiness += "## Category and multi-label evidence\n"
    readiness += (f"Available genuine categories: {len(label_info['available_categories'])}. "
                  "The following is the requested taxonomy with **observed annotation counts**, not assigned labels. "
                  "No category column or artificial category assignments were added to the clean dataset.\n\n")
    readiness += table(["Requested category", "Verified clauses"], [(name, label_info["category_distribution"].get(name, 0)) for name in CATEGORIES]) + "\n\n"
    readiness += "Local evidence:\n"
    readiness += ("- V3 JSON and the review CSVs contain structural metadata/actions, not category annotations.\n"
                  "- `data/raw`, `data/splits`, and `training` have no annotated training corpus in this snapshot.\n"
                  "- `tests/test_module1.py` and `README.md` contain synthetic/example class tags; they are not research annotations.\n"
                  "- `external_data/kleister-nda/README.md` describes document-level extraction, not clause-level classification.\n"
                  "- Multiple `party=` values in Kleister-NDA are not multiple clause-category labels.\n"
                  "- Local structured/text data were scanned; test/cache/dependency artifacts and generated cleaned data were excluded. "
                  "Potential label assignments or unsupported data formats block a build pending provenance inspection.\n\n")
    readiness += "```json\n" + json.dumps(audit, indent=2, ensure_ascii=False) + "\n```\n\n"
    readiness += "## Missing data required before Module 5\n"
    readiness += ("1. Genuine, provenance-backed human clause-category annotations: document_id, stable clause_id (or exact source span), "
                  "clause text, and one or more explicitly assigned categories from the 14-category taxonomy. "
                  "Obtain the authors' annotated data separately or have qualified annotators label this corpus; no download is performed here.\n"
                  "2. Explicit multi-label sets for clauses where multiple categories apply, plus annotation guidelines, reviewer identity, "
                  "and adjudication/version records. Single-label examples or parser support do not supply multi-label data.\n"
                  "3. Complete/adjudicate structural review, including unreviewed segments, conflicting test-like records, "
                  "merge/split decisions, and exclusion audits, before treating clause boundaries as validated.\n"
                  "4. Verify adequate genuine examples per target category and create document-disjoint train/validation/test partitions "
                  "after annotations exist. No unsupported minimum sample size is assumed.\n\n")
    readiness += "## Validation and preservation\n"
    readiness += table(["Check", "Status", "Count"], [(c["check"], c["status"], c["count"]) for c in checks]) + "\n\n"
    digest = hashlib.sha256(json.dumps(protected, sort_keys=True).encode()).hexdigest()
    readiness += (f"Protected source snapshot: {len(protected)} files; SHA-256 manifest digest `{digest}`. "
                  "The build verifies identical before/after hashes for original segmentation, human-review outputs, "
                  "Kleister-NDA files, and human-review reports. Validation warnings/blockers are not a training approval.\n\n"
                  "Module 4 ends here. No Gemini requests, downloads, classifier training, or Module 5 execution.\n")
    return summary, distribution, analysis, readiness


def execute(root, build=False):
    before = fingerprint(root)
    state = snapshot(root)
    all_rows, retained, removed = clean_rows(state)
    clean_path = root / "data/cleaned/clean_clause_dataset.csv"
    if build:
        if clean_path.exists():
            existing, _ = csv_rows(clean_path)
            if any(any(key.lower() in LABEL_FIELDS and str(value).strip() for key, value in row.items()) for row in existing):
                raise ValueError("Existing clean dataset contains label fields; refusing to overwrite annotations")
        audit = audit_annotations(root)
        if audit["candidate_files"] or audit["unsupported_files"]:
            raise ValueError("Annotation provenance inspection required: " + json.dumps(audit, ensure_ascii=False))
        rows = retained
        columns = list(all_rows[0])
    else:
        rows, columns = csv_rows(clean_path)
    checks, label_info = validate_rows(rows, columns, state[0], {r["clause_id"] for r in retained})
    if fingerprint(root) != before:
        raise RuntimeError("Protected inputs changed during processing; refusing to publish")
    report_path = root / "reports/clean_dataset_validation.csv"
    validation_text = csv_text(checks, ("check", "status", "count", "clause_ids", "details"))
    if build:
        summary, distribution, analysis, readiness = reports(
            state, all_rows, retained, removed, audit, checks, label_info, before)
        if any(check["status"] == "ERROR" for check in checks):
            raise ValueError("Structural validation failed; clean dataset not published")
        write_text(clean_path, csv_text(retained, columns))
        write_text(root / "reports/module3a_action_distribution.csv", csv_text(distribution, ("action", "count", "percentage", "basis")))
        write_text(root / "reports/module3a_final_analysis.md", analysis)
        write_text(root / "reports/baseline_dataset_readiness.md", readiness)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({"clean_rows": len(rows), **label_info,
                          "validation": [{k: c[k] for k in ("check", "status", "count")} for c in checks]}, indent=2))
    write_text(report_path, validation_text)
    if fingerprint(root) != before:
        raise RuntimeError("Protected source integrity check failed")
    print(f"Protected source integrity verified: {len(before)} files. No API calls. No training.")
    return 1 if any(check["status"] == "ERROR" for check in checks) else 0


def self_test():
    """Synthetic in-memory fixtures; never read or write research data."""
    class Module4Tests(unittest.TestCase):
        def setUp(self):
            self.source = {"document_id": "fixture", "clause_id": "fixture_clause_000", "clause_text": "fixture text",
                           "source": "data/segmentation_v3/fixture.json", "segment_type": "legal_clause"}

        def resolve(self, action=None, humans=(), failed=False):
            success = {"suggested_action": action, "confidence": "0.95", "timestamp": "fixture"} if action else None
            records = [{"ai_suggested_action": action, "ai_confidence": "0.95", "ai_reason": "Fixture only",
                        "review_file": "data/human_review/llm_pseudo_labels.csv"}] if action else []
            records += [{"human_action": value, "reviewed": "True", "review_file": "priority_review.csv"} for value in humans]
            return resolve_clause(self.source, records, success, failed, True)

        def test_all_actions_and_original_text(self):
            expected = {"KEEP": "PSEUDO_KEEP", "REMOVE_NON_LEGAL": "EXCLUDED_NON_LEGAL",
                        "MERGE_WITH_NEXT": "PENDING_MERGE", "MERGE_WITH_PREVIOUS": "PENDING_MERGE",
                        "SPLIT": "PENDING_SPLIT", "REVIEW": "PENDING_REVIEW"}
            for action, status in expected.items():
                with self.subTest(action=action):
                    row = self.resolve(action)
                    self.assertEqual(row["review_status"], status)
                    self.assertEqual(row["clause_text"], self.source["clause_text"])

        def test_conflicting_removal_is_retained(self):
            row = self.resolve("REMOVE_NON_LEGAL", ["KEEP"])
            self.assertEqual(row["review_status"], "CONFLICT_REVIEW")
            self.assertEqual(row["review_action"], "REVIEW")
            self.assertEqual(row["ai_suggested_action"], "REMOVE_NON_LEGAL")

        def test_unreviewed_and_failure_do_not_invent_keep(self):
            for failed in (True, False):
                row = self.resolve(failed=failed)
                self.assertEqual(row["review_action"], "")
                self.assertEqual(row["confidence"], "")

        def test_duplicates_missing_empty_and_identity(self):
            row = self.resolve()
            broken = dict(row, document_id="wrong", clause_text=" ", review_action="BAD", review_status="")
            checks, _ = validate_rows([row, broken], list(row), {row["clause_id"]: self.source})
            results = {check["check"]: check for check in checks}
            for name in ("duplicate_clause_ids", "empty_clauses", "invalid_review_actions", "missing_review_status", "inconsistent_document_ids"):
                self.assertEqual(results[name]["status"], "ERROR")
            checks, _ = validate_rows([dict(row, clause_text="")], list(row), {row["clause_id"]: self.source})
            self.assertEqual(next(c for c in checks if c["check"] == "missing_clause_text")["status"], "ERROR")

        def test_duplicate_text_is_warning_not_dropped(self):
            row = self.resolve()
            other = dict(row, clause_id="fixture_clause_001")
            checks, _ = validate_rows([row, other], list(row), {row["clause_id"]: self.source, other["clause_id"]: other})
            self.assertEqual(next(c for c in checks if c["check"] == "duplicate_text")["status"], "WARN")

        def test_labels_need_format_vocabulary_and_provenance(self):
            self.assertEqual(parse_labels(self.resolve()), ([], False))
            labels = json.dumps(list(CATEGORIES[:2]))
            self.assertFalse(parse_labels({"labels": labels})[1])
            self.assertTrue(parse_labels({"labels": labels, "label_verified": "true", "label_source": "fixture", "label_reviewer": "fixture"})[1])
            for value in ("fake", '["unknown"]', '[1]', '[]', json.dumps([CATEGORIES[0]] * 2)):
                with self.assertRaises((ValueError, TypeError)):
                    parse_labels({"labels": value})

        def test_document_metadata_is_not_a_clause_annotation(self):
            self.assertFalse(annotation_candidate({"filename": "fixture.pdf", "party": "fixture"}))
            self.assertFalse(annotation_candidate({"clause_id": "fixture", "ai_suggested_action": "KEEP"}))
            self.assertTrue(annotation_candidate({"clause_id": "fixture", "labels": [CATEGORIES[0]]}))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Module4Tests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Build the clean candidate CSV and Module 4 reports offline")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic in-memory safety/validation checks only")
    args = parser.parse_args()
    try:
        sys.exit(self_test() if args.self_test else execute(ROOT, args.build))
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f"Module 4 stopped safely: {error}", file=sys.stderr)
        sys.exit(1)
