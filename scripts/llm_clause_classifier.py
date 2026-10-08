"""scripts/llm_clause_classifier.py

LLM-Assisted Multi-Model Ensemble Pseudo-Labeling Pipeline for NDA Clauses.

Generates initial pre-annotations across 14 NDA clause categories using an
ensemble of Gemini models (gemini-3.6-flash, gemini-3.5-flash, gemini-3.1-flash-lite).

All labels are strictly recorded with label_source = 'llm_pseudo_label'.

Status values:
  - HIGH_CONFIDENCE: All successful models agree with high confidence
  - MEDIUM_CONFIDENCE: Moderate agreement among successful models
  - LOW_CONFIDENCE: Low agreement or confidence among successful models
  - DISAGREEMENT: Genuine multi-model disagreement (all models responded successfully)
  - API_ERROR: One or more model API calls failed (not genuine disagreement)
  - INVALID_RESPONSE: Model returned unparseable/invalid response

Features:
  - Robust Rate-Limit (429/503) handling with adaptive backoff
  - Resume support (skips already-processed clause_ids, including API_ERROR rows)
  - Resumable with --retry-errors to reprocess only API_ERROR/INVALID_RESPONSE rows
  - Quota exhaustion safe stop (preserves completed progress)
  - Downstream priority queue, category distribution, & verification sample generation

Usage:
    python scripts/llm_clause_classifier.py --limit 10
    python scripts/llm_clause_classifier.py
    python scripts/llm_clause_classifier.py --retry-errors
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
import pandas as pd
import dotenv

dotenv.load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

# Approved 14 Categories
APPROVED_CATEGORIES = (
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

CATEGORY_MAP = {c.lower(): c for c in APPROVED_CATEGORIES}

CLASSIFICATION_PROMPT_TEMPLATE = """You are an expert legal document analyst.
Your task is to classify the provided NDA (Non-Disclosure Agreement) clause into one or more of the following 14 categories:

1. Party Identification
2. Purpose
3. NDA Type
4. Definition of Confidential Information
5. Confidentiality Obligations
6. Authorized Disclosure
7. Non-Confidential Information
8. Liability for Damages
9. Competition Rights
10. Term and Termination
11. Intellectual Property
12. Employees
13. Governing Law and Jurisdiction
14. Additional Information

CRITICAL RULES:
- You must select ONLY from the 14 categories listed above. Do NOT invent new category names.
- Multiple categories are allowed if the clause covers multiple aspects.
- If no specific category is applicable, return ["Additional Information"].
- Return ONLY valid JSON with no markdown block formatting, no explanations outside JSON.

JSON Structure:
{{
  "clause_id": "{clause_id}",
  "labels": ["Category 1", "Category 2"],
  "confidence": 0.95,
  "reason": "Short 1-sentence justification"
}}

Document ID: {document_id}
Clause ID: {clause_id}
Clause Text:
"{clause_text}"
"""

DEFAULT_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
DEFAULT_TIMEOUT_SEC = 45.0  # Model B (gemini-3.5-flash) needs ~10-15s for classification prompts
MIN_MODELS_FOR_ENSEMBLE = 2  # Minimum successful model responses needed for a valid ensemble

def init_genai_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError:
        raise ImportError("google-genai package is required. Install via pip install google-genai")

def check_available_models(client, candidate_models):
    """Check model availability using the models.list() API (zero quota cost).
    A 429 RESOURCE_EXHAUSTED means the model EXISTS but is rate-limited — that's
    not 'unavailable'. The old approach wasted quota on test requests.
    """
    print("Checking availability of candidate Gemini models via API listing...")
    available = []
    try:
        listed_models = {str(m.name) for m in client.models.list()}
    except Exception as e:
        print(f"  [WARN] Could not list models: {e}")
        print(f"  Assuming all candidate models are available.")
        return list(candidate_models)

    for m in candidate_models:
        # Model names in listing are prefixed with "models/"
        if f"models/{m}" in listed_models or m in listed_models:
            print(f"  [AVAILABLE] Model: {m}")
            available.append(m)
        else:
            print(f"  [NOT FOUND] Model {m} not in API listing")
            time.sleep(1)
    print(f"Total available models for ensemble: {len(available)} / {len(candidate_models)}")
    return available

def parse_llm_json_response(raw_text: str, clause_id: str) -> tuple[dict, str]:
    """Parse model response. Returns (result_dict, status) where status is 'SUCCESS' or 'INVALID_RESPONSE'."""
    clean_text = raw_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    try:
        data = json.loads(clean_text)
    except Exception as e:
        print(f"    [WARN] Failed to parse JSON: {e}. Raw output: {raw_text[:100]}")
        return {
            "clause_id": clause_id,
            "labels": [],
            "confidence": 0.0,
            "reason": f"INVALID_RESPONSE: JSON parsing error - {str(e)[:100]}"
        }, "INVALID_RESPONSE"

    raw_labels = data.get("labels", [])
    if isinstance(raw_labels, str):
        raw_labels = [raw_labels]

    valid_labels = []
    for lbl in raw_labels:
        lbl_str = str(lbl).strip()
        if lbl_str in APPROVED_CATEGORIES:
            valid_labels.append(lbl_str)
        elif lbl_str.lower() in CATEGORY_MAP:
            valid_labels.append(CATEGORY_MAP[lbl_str.lower()])

    if not valid_labels:
        valid_labels = ["Additional Information"]

    conf = data.get("confidence", 0.5)
    try:
        conf = float(conf)
        conf = max(0.0, min(1.0, conf))
    except Exception:
        conf = 0.5

    reason = str(data.get("reason", "No justification provided")).strip()[:200]

    return {
        "clause_id": clause_id,
        "labels": valid_labels,
        "confidence": conf,
        "reason": reason
    }, "SUCCESS"

def query_model_with_retry(client, model_name: str, prompt: str, clause_id: str, max_retries: int = 5, timeout_sec: float = None) -> tuple[dict, str]:
    """
    Returns (result_dict, status_string).
    Status is one of: 'SUCCESS', 'API_ERROR', 'INVALID_RESPONSE'
    On API_ERROR, labels will be empty and confidence 0.0 — never fabricated.
    """
    if timeout_sec is None:
        timeout_sec = DEFAULT_TIMEOUT_SEC

    last_error = "Unknown error"
    for attempt in range(1, max_retries + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(client.models.generate_content, model=model_name, contents=prompt)
                res = future.result(timeout=timeout_sec)

            if res and res.text:
                parsed, parse_status = parse_llm_json_response(res.text, clause_id)
                return parsed, parse_status
            else:
                last_error = "Empty response from model"
                print(f"    [EMPTY] {model_name} returned empty response (Attempt {attempt}/{max_retries}).")
                time.sleep(3)
        except TimeoutError:
            last_error = f"Timeout after {timeout_sec}s"
            print(f"    [TIMEOUT] {model_name} timed out after {timeout_sec}s (Attempt {attempt}/{max_retries}). Retrying...")
            time.sleep(3)
        except Exception as e:
            err_msg = str(e)
            last_error = err_msg[:200]
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg or "503" in err_msg:
                # Check for explicit retry delay in message (e.g. retryDelay: '24s')
                delay_match = re.search(r"retryDelay':\s*'(\d+)s'", err_msg) or re.search(r"retry in (\d+\.?\d*)s", err_msg)
                if delay_match:
                    wait_time = int(float(delay_match.group(1))) + 2
                else:
                    wait_time = min(60, 15 * attempt)
                print(f"    [RATE LIMIT / BUSY 429/503] {model_name} (Attempt {attempt}/{max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    [API ERROR] {model_name}: {e}")
                time.sleep(3)

    print(f"    [FAIL] {model_name} failed after {max_retries} attempts for Clause {clause_id}.")
    return {
        "clause_id": clause_id,
        "labels": [],
        "confidence": 0.0,
        "reason": f"API_ERROR: {last_error}"
    }, "API_ERROR"

def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    union = set1.union(set2)
    if not union:
        return 1.0
    return len(set1.intersection(set2)) / len(union)

def compute_ensemble_metrics(m_preds: list[dict], m_statuses: list[str]):
    """
    Compute ensemble metrics using ONLY successful model predictions.
    Failed models (API_ERROR, INVALID_RESPONSE) are excluded from
    agreement/disagreement computation to prevent misclassification.
    """
    # Separate successful predictions from failed ones
    successful_preds = [p for p, s in zip(m_preds, m_statuses) if s == "SUCCESS"]
    failed_statuses = [s for s in m_statuses if s != "SUCCESS"]
    num_successful = len(successful_preds)
    num_failed = len(failed_statuses)
    total_models = len(m_preds)

    # If no models succeeded, mark as API_ERROR
    if num_successful == 0:
        all_reasons = [p.get("reason", "") for p in m_preds]
        return {
            "final_pseudo_labels": json.dumps([]),
            "average_confidence": 0.0,
            "agreement_score": 0.0,
            "label_disagreement": False,
            "pseudo_label_quality": "API_ERROR",
            "reason": f"All {total_models} models failed: " + " | ".join(all_reasons)[:250]
        }

    # If fewer than MIN_MODELS_FOR_ENSEMBLE succeeded, mark as API_ERROR
    if num_successful < MIN_MODELS_FOR_ENSEMBLE:
        reason_parts = [p.get("reason", "") for p in m_preds]
        return {
            "final_pseudo_labels": json.dumps(successful_preds[0]["labels"] if successful_preds else []),
            "average_confidence": successful_preds[0]["confidence"] if successful_preds else 0.0,
            "agreement_score": 0.0,
            "label_disagreement": False,
            "pseudo_label_quality": "API_ERROR",
            "reason": f"Only {num_successful}/{total_models} models responded (need {MIN_MODELS_FOR_ENSEMBLE}): " + " | ".join(reason_parts)[:200]
        }

    # Compute metrics using ONLY successful predictions
    labels_sets = [set(p["labels"]) for p in successful_preds]
    confidences = [p["confidence"] for p in successful_preds]
    avg_conf = sum(confidences) / len(confidences)

    j_sims = []
    n = len(labels_sets)
    if n > 1:
        for i in range(n):
            for j in range(i + 1, n):
                j_sims.append(jaccard_similarity(labels_sets[i], labels_sets[j]))
        agreement_score = sum(j_sims) / len(j_sims)
    else:
        agreement_score = 1.0

    all_identical = all(ls == labels_sets[0] for ls in labels_sets)
    disagreement = not all_identical

    vote_counts = {}
    for ls in labels_sets:
        for cat in ls:
            vote_counts[cat] = vote_counts.get(cat, 0) + 1

    majority_threshold = max(2, (n // 2) + 1) if n >= 2 else 1
    final_labels = [cat for cat, votes in vote_counts.items() if votes >= majority_threshold]

    if not final_labels:
        best_pred = max(successful_preds, key=lambda x: x["confidence"])
        final_labels = list(best_pred["labels"])

    # Determine quality — ONLY from successful models
    if agreement_score >= 0.8 and avg_conf >= 0.8:
        quality = "HIGH_CONFIDENCE"
    elif agreement_score >= 0.5 and avg_conf >= 0.6:
        quality = "MEDIUM_CONFIDENCE"
    elif agreement_score < 0.5:
        quality = "DISAGREEMENT"  # Genuine disagreement — all responding models actually disagreed
    else:
        quality = "LOW_CONFIDENCE"

    # Note if any models failed (but enough succeeded for ensemble)
    reasons = [p.get("reason", "") for p in m_preds]
    combined_reason = " | ".join([r for r in reasons if r])[:250]
    if num_failed > 0:
        combined_reason = f"[{num_failed}/{total_models} models failed] " + combined_reason
        combined_reason = combined_reason[:250]

    return {
        "final_pseudo_labels": json.dumps(final_labels),
        "average_confidence": round(avg_conf, 4),
        "agreement_score": round(agreement_score, 4),
        "label_disagreement": disagreement,
        "pseudo_label_quality": quality,
        "reason": combined_reason
    }

# clean_fallback_rows has been REMOVED.
# Rows must never be silently deleted from the CSV.
# Instead, API_ERROR rows are preserved and can be retried with --retry-errors.

def get_retryable_ids(output_path: Path) -> set:
    """Return clause_ids that have API_ERROR or INVALID_RESPONSE status and should be retried."""
    retryable = set()
    if not output_path.exists():
        return retryable
    try:
        df = pd.read_csv(output_path)
        if "pseudo_label_quality" in df.columns:
            mask = df["pseudo_label_quality"].isin(["API_ERROR", "INVALID_RESPONSE"])
            # Also catch legacy rows with "API call failed" in the reason field
            if "reason" in df.columns:
                mask = mask | df["reason"].str.contains("API call failed", na=False)
            retryable = set(df.loc[mask, "clause_id"].dropna().astype(str))
    except Exception as e:
        print(f"Could not identify retryable rows: {e}")
    return retryable


def run_pseudolabel_pipeline(limit: int = None, input_path: Path = None, output_path: Path = None, retry_errors: bool = False):
    input_path = input_path or ROOT / "data/annotations/annotation_queue.csv"
    output_path = output_path or ROOT / "data/annotations/llm_pseudo_labels.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Input queue not found at {input_path}")

    client = init_genai_client()
    available_models = check_available_models(client, DEFAULT_MODELS)
    if not available_models:
        raise RuntimeError("No Gemini models available for generation.")
    if len(available_models) < MIN_MODELS_FOR_ENSEMBLE:
        raise RuntimeError(f"Only {len(available_models)} models available, need at least {MIN_MODELS_FOR_ENSEMBLE} for ensemble.")

    df_queue = pd.read_csv(input_path)
    print(f"\nLoaded {len(df_queue)} clauses from {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    retryable_ids = set()

    fieldnames = [
        "clause_id",
        "document_id",
        "clause_text",
        "model_a_labels",
        "model_b_labels",
        "model_c_labels",
        "final_pseudo_labels",
        "model_a_confidence",
        "model_b_confidence",
        "model_c_confidence",
        "average_confidence",
        "agreement_score",
        "pseudo_label_quality",
        "reason",
        "label_source",
    ]

    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            if "clause_id" in existing_df.columns:
                existing_ids = set(existing_df["clause_id"].dropna().astype(str))
                print(f"Found {len(existing_ids)} clause_ids already in {output_path}")
        except Exception as e:
            print(f"Could not read existing output for resume: {e}")

    if retry_errors:
        retryable_ids = get_retryable_ids(output_path)
        print(f"Retry mode: {len(retryable_ids)} clause_ids with API_ERROR/INVALID_RESPONSE will be reprocessed.")
        if not retryable_ids:
            print("No retryable rows found. Nothing to do.")
            return
        # Remove retryable IDs from existing so they get reprocessed
        # We'll write to a temp file and merge at the end
        skip_ids = existing_ids - retryable_ids
    else:
        skip_ids = existing_ids
        print(f"Resuming pipeline: skipping {len(skip_ids)} already-processed clauses.")

    # For retry mode: rebuild CSV without the retryable rows, then append new results
    if retry_errors and retryable_ids and output_path.exists():
        existing_df = pd.read_csv(output_path)
        # Keep only non-retryable rows
        keep_df = existing_df[~existing_df["clause_id"].astype(str).isin(retryable_ids)]
        # Write them to a temp file first, then replace
        temp_path = output_path.with_suffix(".tmp.csv")
        keep_df.to_csv(temp_path, index=False)
        # Replace original with cleaned version
        import shutil
        shutil.move(str(temp_path), str(output_path))
        print(f"Removed {len(existing_df) - len(keep_df)} retryable rows from CSV. Preserved {len(keep_df)} valid rows.")

    file_mode = "a" if output_path.exists() and len(skip_ids) > 0 else "w"
    write_header = not (output_path.exists() and len(skip_ids) > 0)

    out_file = open(output_path, mode=file_mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        out_file.flush()

    processed_count = len(skip_ids)
    newly_processed = 0
    consecutive_quota_failures = 0

    for idx, row in df_queue.iterrows():
        cid = str(row["clause_id"])
        doc_id = str(row["document_id"])
        text = str(row["clause_text"])

        if cid in skip_ids:
            continue

        if limit is not None and newly_processed >= limit:
            print(f"\nReached limit of {limit} newly processed clauses. Stopping for evaluation.")
            break

        print(f"\nProcessing [{processed_count + 1}/{len(df_queue)}] Clause ID: {cid} (Doc: {doc_id})", flush=True)

        prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
            clause_id=cid,
            document_id=doc_id,
            clause_text=text
        )

        model_preds = []
        model_statuses = []

        for i, m_name in enumerate(available_models):
            print(f"  -> Model {chr(65+i)} ({m_name})...", end="", flush=True)
            pred, status = query_model_with_retry(client, m_name, prompt, cid)
            model_preds.append(pred)
            model_statuses.append(status)
            status_icon = "OK" if status == "SUCCESS" else status
            print(f" [{status_icon}] Labels: {pred['labels']} (Conf: {pred['confidence']})", flush=True)
            # Polite pause between requests to prevent hitting RPM limits
            time.sleep(4)

        # Check if all models failed due to daily quota
        if not any(s == "SUCCESS" for s in model_statuses):
            consecutive_quota_failures += 1
            if consecutive_quota_failures >= 3:
                print("\n[CRITICAL API QUOTA REACHED] 3 consecutive clauses failed on all models due to API Quota/Rate limits.")
                print("Safely stopping pipeline to preserve completed results.")
                break
        else:
            consecutive_quota_failures = 0

        # Pad to exactly 3 model slots if fewer models were available
        while len(model_preds) < 3:
            model_preds.append({
                "clause_id": cid,
                "labels": [],
                "confidence": 0.0,
                "reason": "Model slot unavailable"
            })
            model_statuses.append("API_ERROR")

        ensemble_res = compute_ensemble_metrics(model_preds, model_statuses)

        record = {
            "clause_id": cid,
            "document_id": doc_id,
            "clause_text": text,
            "model_a_labels": json.dumps(model_preds[0]["labels"]),
            "model_b_labels": json.dumps(model_preds[1]["labels"]),
            "model_c_labels": json.dumps(model_preds[2]["labels"]),
            "final_pseudo_labels": ensemble_res["final_pseudo_labels"],
            "model_a_confidence": model_preds[0]["confidence"],
            "model_b_confidence": model_preds[1]["confidence"],
            "model_c_confidence": model_preds[2]["confidence"],
            "average_confidence": ensemble_res["average_confidence"],
            "agreement_score": ensemble_res["agreement_score"],
            "pseudo_label_quality": ensemble_res["pseudo_label_quality"],
            "reason": ensemble_res["reason"],
            "label_source": "llm_pseudo_label",
        }

        writer.writerow(record)
        out_file.flush()
        skip_ids.add(cid)
        processed_count += 1
        newly_processed += 1

    out_file.close()
    print(f"\nPipeline finished run. Total processed clauses: {processed_count} / {len(df_queue)}")
    print(f"  Newly processed this run: {newly_processed}")

    if output_path.exists():
        generate_secondary_artifacts(output_path)

def generate_secondary_artifacts(llm_labels_path: Path):
    print("\nGenerating downstream priority queue, distribution reports, and verification sample...")
    df = pd.read_csv(llm_labels_path)
    if len(df) == 0:
        print("Dataset is empty. Skipping downstream artifact generation.")
        return

    quality_weight = df["pseudo_label_quality"].map({
        "API_ERROR": 1.0,
        "INVALID_RESPONSE": 1.0,
        "DISAGREEMENT": 0.9,
        "LOW_CONFIDENCE": 0.7,
        "MEDIUM_CONFIDENCE": 0.4,
        "HIGH_CONFIDENCE": 0.1
    }).fillna(0.5)

    text_len_bonus = (df["clause_text"].str.len() / 1000.0).clip(0, 0.5)
    uncertainty_score = (1.0 - df["agreement_score"]) + 0.5 * (1.0 - df["average_confidence"]) + quality_weight + text_len_bonus

    df_priority = df.copy()
    df_priority["uncertainty_priority_score"] = uncertainty_score.round(4)
    df_priority = df_priority.sort_values(by="uncertainty_priority_score", ascending=False)

    priority_path = ROOT / "data/annotations/human_review_priority.csv"
    priority_path.parent.mkdir(parents=True, exist_ok=True)
    df_priority.to_csv(priority_path, index=False)
    print(f"  -> Saved human review priority queue to: {priority_path}")

    cat_counts = {c: 0 for c in APPROVED_CATEGORIES}
    cat_confs = {c: [] for c in APPROVED_CATEGORIES}
    cat_agreements = {c: [] for c in APPROVED_CATEGORIES}

    for idx, row in df.iterrows():
        try:
            labels = json.loads(row["final_pseudo_labels"])
        except Exception:
            labels = ["Additional Information"]
        conf = float(row["average_confidence"])
        agree = float(row["agreement_score"])
        for c in labels:
            if c in cat_counts:
                cat_counts[c] += 1
                cat_confs[c].append(conf)
                cat_agreements[c].append(agree)

    total_clauses = len(df)
    dist_rows = []
    for c in APPROVED_CATEGORIES:
        cnt = cat_counts[c]
        pct = round((cnt / total_clauses) * 100, 2) if total_clauses > 0 else 0.0
        avg_c = round(sum(cat_confs[c]) / len(cat_confs[c]), 4) if cat_confs[c] else 0.0
        avg_a = round(sum(cat_agreements[c]) / len(cat_agreements[c]), 4) if cat_agreements[c] else 0.0
        dist_rows.append({
            "Category": c,
            "Number of clauses": cnt,
            "Percentage": pct,
            "Average confidence": avg_c,
            "Average agreement score": avg_a
        })

    dist_df = pd.DataFrame(dist_rows)
    dist_csv_path = ROOT / "reports/llm_pseudolabel_category_distribution.csv"
    dist_csv_path.parent.mkdir(parents=True, exist_ok=True)
    dist_df.to_csv(dist_csv_path, index=False)
    print(f"  -> Saved category distribution CSV to: {dist_csv_path}")

    dist_md_path = ROOT / "reports/llm_pseudolabel_category_distribution.md"
    md_content = ["# LLM Pseudo-Label Category Distribution Report\n"]
    md_content.append(f"**Total Clauses Processed**: {total_clauses}\n")
    md_content.append("| Category | Number of Clauses | Percentage (%) | Average Confidence | Average Agreement Score |")
    md_content.append("|---|---|---|---|---|")
    zero_cats = []
    for r in dist_rows:
        md_content.append(f"| {r['Category']} | {r['Number of clauses']} | {r['Percentage']}% | {r['Average confidence']} | {r['Average agreement score']} |")
        if r['Number of clauses'] == 0:
            zero_cats.append(r['Category'])

    if zero_cats:
        md_content.append("\n> [!WARNING]")
        md_content.append(f"> The following categories received 0 pseudo-label predictions: **{', '.join(zero_cats)}**.")
        md_content.append("> Note: Per strict research protocol, no artificial labels have been added.")

    dist_md_path.write_text("\n".join(md_content), encoding="utf-8")
    print(f"  -> Saved category distribution Markdown to: {dist_md_path}")

    sample_size = min(40, len(df))
    sample_dfs = []
    for q_tier, group in df.groupby("pseudo_label_quality"):
        n_sample = max(1, int(len(group) / len(df) * sample_size))
        sample_dfs.append(group.sample(n=min(n_sample, len(group)), random_state=42))

    df_sample = pd.concat(sample_dfs).drop_duplicates(subset=["clause_id"]).head(sample_size)
    sample_path = ROOT / "data/annotations/human_verification_sample.csv"
    df_sample.to_csv(sample_path, index=False)
    print(f"  -> Saved human verification sample ({len(df_sample)} clauses) to: {sample_path}")

    report_path = ROOT / "reports/module5_llm_pseudolabel_report.md"
    q_counts = df["pseudo_label_quality"].value_counts().to_dict()

    rep_content = [
        "# Module 5 — LLM Pseudo-Labeling Pipeline Report\n",
        "## Executive Summary",
        "**RESEARCH DISCLAIMER**:",
        "> \"These are LLM-generated pseudo-labels and are not equivalent to expert human annotations.\"\n",
        f"- **Total Clauses Processed**: {len(df)}",
        f"- **High Confidence Count**: {q_counts.get('HIGH_CONFIDENCE', 0)}",
        f"- **Medium Confidence Count**: {q_counts.get('MEDIUM_CONFIDENCE', 0)}",
        f"- **Low Confidence Count**: {q_counts.get('LOW_CONFIDENCE', 0)}",
        f"- **Disagreement Count**: {q_counts.get('DISAGREEMENT', 0)}",
        f"- **Average Ensemble Confidence**: {df['average_confidence'].mean():.4f}",
        f"- **Average Agreement Score**: {df['agreement_score'].mean():.4f}",
        f"- **Human Verification Sample Size**: {len(df_sample)}",
        "\n## Limitations",
        "1. Pseudo-labels reflect LLM prior biases and may miss domain-specific legal nuances.",
        "2. Highly imbalanced categories may receive zero predictions if clauses are absent in source corpus.",
        "3. Pseudo-labels must only be used in separate experimental training setups (Experiment B & C).",
    ]
    report_path.write_text("\n".join(rep_content), encoding="utf-8")
    print(f"  -> Saved pipeline report to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Pseudo-Labeling Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clauses to process")
    parser.add_argument("--retry-errors", action="store_true", help="Reprocess only API_ERROR/INVALID_RESPONSE rows")
    args = parser.parse_args()

    run_pseudolabel_pipeline(limit=args.limit, retry_errors=args.retry_errors)
