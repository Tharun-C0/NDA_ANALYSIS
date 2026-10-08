"""
scripts/resume_runner.py

Robust recovery and resume executor for Module 5B LLM Pseudo-Labeling Pipeline.
Processes remaining clauses from pseudolabel_resume_queue.csv without overwriting
valid existing predictions in llm_pseudo_labels.csv.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
import pandas as pd
import dotenv

dotenv.load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

# Approved 14 Categories
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
DEFAULT_TIMEOUT_SEC = 60.0
MIN_MODELS_FOR_ENSEMBLE = 2


def init_genai_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    from google import genai
    return genai.Client(api_key=api_key)


def check_available_models_once(client, candidate_models):
    print("Checking availability of Gemini models via API listing...", flush=True)
    try:
        listed_models = {str(m.name) for m in client.models.list()}
        available = []
        for m in candidate_models:
            if f"models/{m}" in listed_models or m in listed_models:
                print(f"  [AVAILABLE] Model: {m}", flush=True)
                available.append(m)
            else:
                print(f"  [WARN] Model {m} not in API listing, but will attempt use.", flush=True)
                available.append(m)
        return available
    except Exception as e:
        print(f"  [WARN] Could not list models ({e}). Proceeding with all candidates: {candidate_models}", flush=True)
        return list(candidate_models)


def parse_llm_json_response(raw_text: str, clause_id: str) -> tuple[dict, str]:
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
        return {
            "clause_id": clause_id,
            "labels": [],
            "confidence": 0.0,
            "reason": f"INVALID_RESPONSE: JSON parse error - {str(e)[:100]}"
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


def query_model_with_retry(client, model_name: str, prompt: str, clause_id: str, max_retries: int = 4, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> tuple[dict, str, float, dict]:
    """
    Returns (result_dict, status_string, total_rate_limit_wait_sec, quota_info_dict).
    """
    last_error = "Unknown error"
    total_rate_wait = 0.0
    quota_info = {"is_exhausted": False, "retry_delay": 0, "model": model_name}

    for attempt in range(1, max_retries + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(client.models.generate_content, model=model_name, contents=prompt)
                res = future.result(timeout=timeout_sec)

            if res and res.text:
                parsed, parse_status = parse_llm_json_response(res.text, clause_id)
                return parsed, parse_status, total_rate_wait, quota_info
            else:
                last_error = "Empty response from model"
                time.sleep(2)
        except TimeoutError:
            last_error = f"Timeout after {timeout_sec}s"
            time.sleep(2)
        except Exception as e:
            err_msg = str(e)
            last_error = err_msg[:250]
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg or "503" in err_msg:
                delay_match = re.search(r"retryDelay':\s*'(\d+)s'", err_msg) or re.search(r"retry in (\d+\.?\d*)s", err_msg)
                server_delay = int(float(delay_match.group(1))) if delay_match else min(45, 10 * attempt)
                
                # If server delay is long (> 120s), do NOT sleep for hours inside the runner
                if server_delay > 120:
                    quota_info["is_exhausted"] = True
                    quota_info["retry_delay"] = server_delay
                    quota_info["error_msg"] = last_error
                    return {
                        "clause_id": clause_id,
                        "labels": [],
                        "confidence": 0.0,
                        "reason": f"API_QUOTA_EXHAUSTED: Model {model_name} quota exceeded. Retry after {server_delay}s."
                    }, "API_QUOTA_EXHAUSTED", total_rate_wait, quota_info

                wait_time = server_delay + 2
                total_rate_wait += wait_time
                time.sleep(wait_time)
            else:
                time.sleep(2)

    return {
        "clause_id": clause_id,
        "labels": [],
        "confidence": 0.0,
        "reason": f"API_ERROR: {last_error}"
    }, "API_ERROR", total_rate_wait, quota_info


def query_openrouter_api(api_key: str, model_name: str, prompt: str, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> tuple[str, int, str]:
    """
    Sends HTTP POST request to OpenRouter chat completions API.
    Returns (response_content, http_status_code, error_message).
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/nda-research",
        "X-Title": "NDA Clause Classifier"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1000
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            resp_body = resp.read().decode("utf-8")
            resp_json = json.loads(resp_body)
            choices = resp_json.get("choices", [])
            if choices and len(choices) > 0:
                content = choices[0].get("message", {}).get("content", "")
                return content, resp.status, ""
            return "", resp.status, "Empty choices in response"
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            pass
        return "", e.code, f"HTTP {e.code}: {err_body[:250]}"
    except urllib.error.URLError as e:
        return "", 0, f"URL Error: {e.reason}"
    except TimeoutError:
        return "", 408, f"Timeout after {timeout_sec}s"
    except Exception as e:
        return "", 0, f"Exception: {str(e)[:250]}"


def query_openrouter_model_with_retry(api_key: str, model_name: str, prompt: str, clause_id: str, max_retries: int = 4, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> tuple[dict, str, float, dict]:
    """
    Query model via OpenRouter's OpenAI-compatible API.
    Returns (result_dict, status_string, total_rate_limit_wait_sec, quota_info_dict).
    """
    last_error = "Unknown error"
    total_rate_wait = 0.0
    quota_info = {"is_exhausted": False, "retry_delay": 0, "model": model_name}

    if not api_key:
        quota_info["is_exhausted"] = True
        quota_info["error_msg"] = "OPENROUTER_API_KEY environment variable is missing"
        return {
            "clause_id": clause_id,
            "labels": [],
            "confidence": 0.0,
            "reason": "API_ERROR: OPENROUTER_API_KEY is missing"
        }, "API_ERROR", total_rate_wait, quota_info

    for attempt in range(1, max_retries + 1):
        content, status_code, err_msg = query_openrouter_api(api_key, model_name, prompt, timeout_sec=timeout_sec)

        if status_code == 200 and content:
            parsed, parse_status = parse_llm_json_response(content, clause_id)
            return parsed, parse_status, total_rate_wait, quota_info

        last_error = err_msg if err_msg else f"HTTP Status {status_code}"

        if status_code == 401:
            quota_info["is_exhausted"] = True
            quota_info["error_msg"] = "Invalid OpenRouter API Key (HTTP 401)"
            return {
                "clause_id": clause_id,
                "labels": [],
                "confidence": 0.0,
                "reason": "API_ERROR: Invalid OpenRouter API Key (HTTP 401)"
            }, "API_ERROR", total_rate_wait, quota_info

        if status_code == 402:
            quota_info["is_exhausted"] = True
            quota_info["error_msg"] = "Insufficient OpenRouter Credits (HTTP 402)"
            return {
                "clause_id": clause_id,
                "labels": [],
                "confidence": 0.0,
                "reason": "API_ERROR: Insufficient OpenRouter Credits (HTTP 402)"
            }, "API_ERROR", total_rate_wait, quota_info

        if status_code == 429:
            server_delay = min(45, 10 * attempt)
            wait_time = server_delay + 2
            total_rate_wait += wait_time
            print(f"    [OPENROUTER RATE LIMIT 429] Waiting {wait_time}s (Attempt {attempt}/{max_retries})...", flush=True)
            time.sleep(wait_time)
        elif status_code >= 500 or status_code in (408, 0) or "Timeout" in err_msg or "URL Error" in err_msg:
            wait_time = 3 * attempt
            print(f"    [OPENROUTER SERVER/NET ERROR {status_code}] {last_error}. Retrying in {wait_time}s...", flush=True)
            time.sleep(wait_time)
        else:
            time.sleep(2)

    return {
        "clause_id": clause_id,
        "labels": [],
        "confidence": 0.0,
        "reason": f"API_ERROR: OpenRouter failed ({last_error})"
    }, "API_ERROR", total_rate_wait, quota_info


def test_openrouter_connectivity(model_name: str = None) -> bool:
    """
    Executes a single minimal test request to OpenRouter to verify connectivity, API key,
    and output schema validity without modifying any datasets.
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not openrouter_key:
        print("[FAIL] OPENROUTER_API_KEY environment variable is missing or empty.", flush=True)
        print("       Please set OPENROUTER_API_KEY=<your_key> in your .env file or environment.", flush=True)
        return False

    target_model = model_name or os.getenv("OPENROUTER_MODEL", "google/gemini-3.6-flash").strip()
    print("=======================================================", flush=True)
    print("      OPENROUTER CONNECTIVITY TEST                    ", flush=True)
    print("=======================================================", flush=True)
    print(f"  Provider    : openrouter", flush=True)
    print(f"  Target Model: {target_model}", flush=True)
    print(f"  Endpoint    : https://openrouter.ai/api/v1/chat/completions", flush=True)
    print("Sending 1 minimal test classification request...", flush=True)

    test_prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
        clause_id="TEST_OPENROUTER_001",
        document_id="TEST_DOC",
        clause_text="This Agreement shall be governed by and construed in accordance with the laws of the State of California."
    )

    start_time = time.time()
    content, status_code, err_msg = query_openrouter_api(openrouter_key, target_model, test_prompt, timeout_sec=30.0)
    elapsed = time.time() - start_time

    if status_code != 200:
        print(f"\n[FAIL] OpenRouter HTTP Status Code: {status_code} ({elapsed:.2f}s)", flush=True)
        print(f"       Error Details: {err_msg}", flush=True)
        if status_code == 401:
            print("       -> HTTP 401 Unauthorized: Invalid API key.", flush=True)
        elif status_code == 402:
            print("       -> HTTP 402 Payment Required: Insufficient credits or payment setup required.", flush=True)
        elif status_code == 429:
            print("       -> HTTP 429 Rate Limit: Quota or rate limit exceeded.", flush=True)
        elif status_code >= 500:
            print("       -> HTTP 5xx Server Error: OpenRouter/Provider internal failure.", flush=True)
        return False

    print(f"\n[SUCCESS] HTTP {status_code} OK ({elapsed:.2f}s)", flush=True)
    print(f"Raw Response      : {content.strip()}", flush=True)

    parsed, parse_status = parse_llm_json_response(content, "TEST_OPENROUTER_001")
    print(f"Parse Status      : {parse_status}", flush=True)
    print(f"Parsed Labels     : {parsed.get('labels')}", flush=True)
    print(f"Parsed Confidence : {parsed.get('confidence')}", flush=True)
    print(f"Parsed Reason     : {parsed.get('reason')}", flush=True)

    if parse_status == "SUCCESS" and len(parsed.get("labels", [])) > 0:
        print("\n=======================================================", flush=True)
        print(" >>> OpenRouter connectivity test PASSED successfully! <<<", flush=True)
        print("=======================================================\n", flush=True)
        return True
    else:
        print("\n[WARN] Response received but JSON parsing failed or labels empty.", flush=True)
        return False


def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    union = set1.union(set2)
    if not union:
        return 1.0
    return len(set1.intersection(set2)) / len(union)


def compute_ensemble_metrics(m_preds: list[dict], m_statuses: list[str], min_models: int = MIN_MODELS_FOR_ENSEMBLE):
    successful_preds = [p for p, s in zip(m_preds, m_statuses) if s == "SUCCESS"]
    failed_statuses = [s for s in m_statuses if s != "SUCCESS"]
    num_successful = len(successful_preds)
    num_failed = len(failed_statuses)
    total_models = len(m_preds)

    if num_successful < min_models:
        reason_parts = [p.get("reason", "") for p in m_preds]
        has_quota_exhausted = any(s == "API_QUOTA_EXHAUSTED" for s in m_statuses)
        quality_tag = "API_QUOTA_EXHAUSTED" if has_quota_exhausted else "API_ERROR"
        return {
            "final_pseudo_labels": json.dumps(successful_preds[0]["labels"] if successful_preds else []),
            "average_confidence": successful_preds[0]["confidence"] if successful_preds else 0.0,
            "agreement_score": 0.0,
            "label_disagreement": False,
            "pseudo_label_quality": quality_tag,
            "reason": f"Only {num_successful}/{total_models} models responded: " + " | ".join(reason_parts)[:200]
        }



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

    if agreement_score >= 0.8 and avg_conf >= 0.8:
        quality = "HIGH_CONFIDENCE"
    elif agreement_score >= 0.5 and avg_conf >= 0.6:
        quality = "MEDIUM_CONFIDENCE"
    elif agreement_score < 0.5:
        quality = "DISAGREEMENT"
    else:
        quality = "LOW_CONFIDENCE"

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


def save_current_llm_labels(output_path: Path, records_dict: dict, fieldnames: list):
    """
    Saves records_dict (clause_id -> dict) to output_path cleanly as CSV.
    """
    temp_path = output_path.with_suffix(".tmp.csv")
    with open(temp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cid in sorted(records_dict.keys()):
            writer.writerow(records_dict[cid])
    import shutil
    shutil.move(str(temp_path), str(output_path))


def generate_final_reports(output_path: Path, total_v3_count: int):
    print("\n========================================================")
    print("      GENERATING FINAL MODULE 5B REPORTS & ARTIFACTS    ")
    print("========================================================")

    df = pd.read_csv(output_path)
    total_csv_rows = len(df)
    unique_clause_count = df["clause_id"].nunique()
    duplicate_count = total_csv_rows - unique_clause_count

    quality_counts = df["pseudo_label_quality"].value_counts().to_dict()
    high_conf = quality_counts.get("HIGH_CONFIDENCE", 0)
    med_conf = quality_counts.get("MEDIUM_CONFIDENCE", 0)
    low_conf = quality_counts.get("LOW_CONFIDENCE", 0)
    agreement_count = high_conf + med_conf + low_conf
    disagreement_count = quality_counts.get("DISAGREEMENT", 0)
    api_error_count = quality_counts.get("API_ERROR", 0)
    invalid_response_count = quality_counts.get("INVALID_RESPONSE", 0)
    usable_pseudo_labeled = agreement_count + disagreement_count

    # Model-wise counts
    model_a_success = sum(df["model_a_confidence"] > 0)
    model_b_success = sum(df["model_b_confidence"] > 0)
    model_c_success = sum(df["model_c_confidence"] > 0)

    # Category distribution
    cat_counts = {cat: 0 for cat in APPROVED_CATEGORIES}
    for labels_str in df["final_pseudo_labels"].dropna():
        try:
            lbls = json.loads(labels_str)
            for l in lbls:
                if l in cat_counts:
                    cat_counts[l] += 1
        except Exception:
            pass

    # Save category distribution CSV
    dist_csv_path = ROOT / "data/annotations/llm_pseudo_label_distribution.csv"
    dist_df = pd.DataFrame([{"category": cat, "count": cnt} for cat, cnt in cat_counts.items()])
    dist_df.to_csv(dist_csv_path, index=False)
    print(f"Saved Category Distribution to: {dist_csv_path}")

    # Save human verification sample CSV (DISAGREEMENT + LOW_CONFIDENCE + API_ERROR)
    verify_df = df[df["pseudo_label_quality"].isin(["DISAGREEMENT", "LOW_CONFIDENCE", "API_ERROR", "INVALID_RESPONSE"])].copy()
    verify_csv_path = ROOT / "data/annotations/human_verification_sample.csv"
    verify_df.to_csv(verify_csv_path, index=False)
    print(f"Saved Human Verification Sample ({len(verify_df)} clauses) to: {verify_csv_path}")

    # Confidence Stats
    avg_conf_series = df["average_confidence"]
    min_conf = round(float(avg_conf_series.min()), 4) if len(avg_conf_series) > 0 else 0.0
    max_conf = round(float(avg_conf_series.max()), 4) if len(avg_conf_series) > 0 else 0.0
    mean_conf = round(float(avg_conf_series.mean()), 4) if len(avg_conf_series) > 0 else 0.0
    std_conf = round(float(avg_conf_series.std()), 4) if len(avg_conf_series) > 1 else 0.0

    # Generate module5b_completion_report.md
    report_path = ROOT / "reports/module5b_completion_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_content = f"""# Module 5B: Multi-Model LLM Pseudo-Labeling Completion Report

> [!IMPORTANT]
> **DISCLAIMER**: The annotations generated in this module are **LLM pseudo-labels** produced by a 3-model ensemble (Gemini 3.6 Flash, Gemini 3.5 Flash, Gemini 3.1 Flash Lite). They represent automated pre-annotations intended to accelerate downstream annotation and model training, **NOT human ground-truth labels**.

---

## 1. Executive Summary

- **Total V3 Clauses in Dataset**: {total_v3_count}
- **Successfully Pseudo-Labeled Clauses**: {usable_pseudo_labeled} / {total_v3_count} ({usable_pseudo_labeled / total_v3_count * 100:.2f}%)
- **Agreement Count**: {agreement_count} ({agreement_count / total_v3_count * 100:.2f}%)
  - **HIGH_CONFIDENCE**: {high_conf}
  - **MEDIUM_CONFIDENCE**: {med_conf}
  - **LOW_CONFIDENCE**: {low_conf}
- **Genuine Disagreement Count**: {disagreement_count} ({disagreement_count / total_v3_count * 100:.2f}%)
- **API_ERROR Clauses**: {api_error_count}
- **INVALID_RESPONSE Clauses**: {invalid_response_count}
- **Duplicate Rows Count**: {duplicate_count}
- **Clauses Requiring Human Verification**: {len(verify_df)}

---

## 2. Multi-Model Ensemble Performance

| Model Name | Model Slot | Successful Responses | Success Rate |
| :--- | :--- | :--- | :--- |
| `gemini-3.6-flash` | Model A | {model_a_success} / {total_csv_rows} | {model_a_success / total_csv_rows * 100:.2f}% |
| `gemini-3.5-flash` | Model B | {model_b_success} / {total_csv_rows} | {model_b_success / total_csv_rows * 100:.2f}% |
| `gemini-3.1-flash-lite` | Model C | {model_c_success} / {total_csv_rows} | {model_c_success / total_csv_rows * 100:.2f}% |

---

## 3. Confidence Statistics

- **Mean Ensemble Confidence**: {mean_conf}
- **Standard Deviation**: {std_conf}
- **Minimum Confidence**: {min_conf}
- **Maximum Confidence**: {max_conf}

---

## 4. Category Distribution Across 14 Approved NDA Categories

| Category Name | Clause Frequency | Percentage |
| :--- | :--- | :--- |
"""
    total_labels_assigned = sum(cat_counts.values())
    for cat, cnt in cat_counts.items():
        pct = (cnt / total_labels_assigned * 100) if total_labels_assigned > 0 else 0.0
        report_content += f"| {cat} | {cnt} | {pct:.2f}% |\n"

    report_content += f"""
---

## 5. Artifacts Produced

1. **Primary Pseudo-Label Dataset**: `data/annotations/llm_pseudo_labels.csv` ({total_csv_rows} rows)
2. **Resume Queue**: `data/annotations/pseudolabel_resume_queue.csv` (0 remaining)
3. **Category Distribution Summary**: `data/annotations/llm_pseudo_label_distribution.csv`
4. **Human Verification Sample**: `data/annotations/human_verification_sample.csv` ({len(verify_df)} rows)
5. **Completion Report**: `reports/module5b_completion_report.md`

---
*Report generated automatically upon Module 5B completion.*
"""
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(report_content)
    print(f"Saved Completion Report to: {report_path}")


# -----------------------------------------------------------------------
# DOCUMENT DIVERSITY PRIORITY TIERS
#   Tier 1 (priority=0): Document has ZERO valid pseudo-labels → unseen
#   Tier 2 (priority=1): Document has very few valid labels (<5) → sparse
#   Tier 3 (priority=2): Document already well-represented (>=5 valid)
# Within each tier the original queue order is preserved.
# -----------------------------------------------------------------------
FEW_VALID_THRESHOLD = 5  # clauses; below this → Tier 2


def _build_diversity_sorted_queue(queue_rows: list[dict], valid_doc_counts: dict[str, int]) -> list[dict]:
    """Re-orders queue_rows by document diversity priority without changing
    the relative order within the same priority tier."""
    def _priority(row):
        cnt = valid_doc_counts.get(str(row["document_id"]), 0)
        if cnt == 0:
            return 0  # Tier 1: completely unseen
        elif cnt < FEW_VALID_THRESHOLD:
            return 1  # Tier 2: sparsely labeled
        else:
            return 2  # Tier 3: already represented
    return sorted(queue_rows, key=_priority)


def _print_diversity_header(queue_rows: list[dict], valid_doc_counts: dict[str, int], max_clauses_shown: int):
    """Print the document diversity header before processing."""
    unseen_docs = {r["document_id"] for r in queue_rows if valid_doc_counts.get(str(r["document_id"]), 0) == 0}
    sparse_docs = {r["document_id"] for r in queue_rows
                   if 0 < valid_doc_counts.get(str(r["document_id"]), 0) < FEW_VALID_THRESHOLD}
    repr_docs   = {r["document_id"] for r in queue_rows
                   if valid_doc_counts.get(str(r["document_id"]), 0) >= FEW_VALID_THRESHOLD}

    print("\n=======================================================")
    print("  DOCUMENT DIVERSITY PRIORITY SELECTION")
    print("=======================================================")
    print(f"  Tier 1 — Unseen documents (0 valid labels)   : {len(unseen_docs)}")
    print(f"  Tier 2 — Sparse documents (<{FEW_VALID_THRESHOLD} valid labels) : {len(sparse_docs)}")
    print(f"  Tier 3 — Represented documents (>={FEW_VALID_THRESHOLD} valid)  : {len(repr_docs)}")
    print()

    # Show what the first max_clauses_shown clauses will look like
    shown = queue_rows[:max_clauses_shown]
    doc_plan: dict[str, int] = {}
    for r in shown:
        doc_plan[str(r["document_id"])[:8]] = doc_plan.get(str(r["document_id"])[:8], 0) + 1
    print(f"  Next {len(shown)} clauses will come from:")
    for doc_prefix, cnt in doc_plan.items():
        tier = "T1" if doc_prefix[:8] in {d[:8] for d in unseen_docs} else (
               "T2" if doc_prefix[:8] in {d[:8] for d in sparse_docs} else "T3")
        print(f"    [{tier}] doc {doc_prefix}: {cnt} clause(s)")
    print("=======================================================")


def run_resume_pipeline(max_clauses: int = None, dry_run: bool = False, provider: str = None):
    queue_path = ROOT / "data/annotations/pseudolabel_resume_queue.csv"
    output_path = ROOT / "data/annotations/llm_pseudo_labels.csv"

    if not queue_path.exists():
        raise FileNotFoundError(f"Resume queue not found at {queue_path}")

    # Total V3 count across JSONs
    v3_files = list((ROOT / "data/segmentation_v3").glob("*.json"))
    total_v3 = 0
    for f in v3_files:
        with open(f, "r", encoding="utf-8") as jf:
            d = json.load(jf)
            c_list = d if isinstance(d, list) else d.get("clauses", [])
            total_v3 += len(c_list)

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

    # Load existing llm_pseudo_labels.csv into a dictionary keyed by clause_id
    existing_records = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_records[r["clause_id"]] = r

    # Determine valid completed IDs and per-document valid counts
    already_valid_ids = set()
    valid_doc_counts: dict[str, int] = {}
    for cid, r in existing_records.items():
        q = r.get("pseudo_label_quality", "").strip()
        if q in ["HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE", "DISAGREEMENT"]:
            already_valid_ids.add(cid)
            doc = r.get("document_id", "")
            valid_doc_counts[doc] = valid_doc_counts.get(doc, 0) + 1

    # Read queue; filter to clauses not yet valid
    df_queue = pd.read_csv(queue_path)
    raw_pending = []
    for _, r in df_queue.iterrows():
        cid = str(r["clause_id"])
        if cid not in already_valid_ids:
            raw_pending.append(r.to_dict())

    # ── DOCUMENT DIVERSITY SORT ────────────────────────────────────────────
    diversity_sorted = _build_diversity_sorted_queue(raw_pending, valid_doc_counts)

    target_n = max_clauses if (max_clauses and max_clauses > 0) else len(diversity_sorted)
    queue_to_process = diversity_sorted[:target_n]

    total_queue_len = len(queue_to_process)
    print(f"Loaded resume queue: {len(df_queue)} total in file, {total_queue_len} targeted for this run.")
    print(f"Existing valid records in CSV: {len(already_valid_ids)}")

    # Print diversity header (always shown, even in dry-run)
    _print_diversity_header(diversity_sorted, valid_doc_counts,
                            max_clauses_shown=total_queue_len)

    if total_queue_len == 0:
        print("Resume queue is empty! Everything is already processed.")
        generate_final_reports(output_path, total_v3)
        return

    # ── DRY RUN: print plan and exit WITHOUT touching LLM APIs ─────────────
    if dry_run:
        print("\n[DRY RUN] Selected clause IDs (no API calls made):")
        print(f"{'#':>4}  {'Doc Prefix':12}  {'Tier':4}  Clause ID")
        print("-" * 80)
        for i, item in enumerate(queue_to_process, 1):
            doc_id = str(item["document_id"])
            cnt = valid_doc_counts.get(doc_id, 0)
            if cnt == 0:
                tier = "T1"
            elif cnt < FEW_VALID_THRESHOLD:
                tier = "T2"
            else:
                tier = "T3"
            print(f"{i:>4}  {doc_id[:12]:12}  {tier:4}  {item['clause_id']}")
        print()
        print(f"[DRY RUN] Total clauses that WOULD be processed: {total_queue_len}")
        print("[DRY RUN] No LLM API calls were made. No files were modified.")
        return

    active_provider = (provider or os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    if active_provider == "openrouter":
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        openrouter_model = os.getenv("OPENROUTER_MODEL", "google/gemini-3.6-flash").strip()
        openrouter_models_env = os.getenv("OPENROUTER_MODELS")
        if openrouter_models_env:
            available_models = [m.strip() for m in openrouter_models_env.split(",") if m.strip()]
        else:
            available_models = [openrouter_model]
        client = None
        min_models_needed = 1 if len(available_models) == 1 else min(2, len(available_models))
        print(f"\n[PROVIDER] LLM_PROVIDER=openrouter (Model(s): {available_models})", flush=True)
    else:
        active_provider = "gemini"
        client = init_genai_client()
        available_models = check_available_models_once(client, DEFAULT_MODELS)
        min_models_needed = MIN_MODELS_FOR_ENSEMBLE
        print("\n[PROVIDER] LLM_PROVIDER=gemini (Direct Google Gemini API)", flush=True)

    # Performance tracking
    total_rate_waits = 0
    total_wait_duration = 0.0
    api_errors_session = 0
    disagreements_session = 0
    processed_this_session = 0
    consecutive_api_errors = 0
    MAX_CONSECUTIVE_API_ERRORS = 5

    for idx, item in enumerate(queue_to_process, start=1):
        cid = str(item["clause_id"])
        doc_id = str(item["document_id"])
        text = str(item["clause_text"])

        prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
            clause_id=cid,
            document_id=doc_id,
            clause_text=text
        )

        model_preds = []
        model_statuses = []

        quota_hit = None
        for i, m_name in enumerate(available_models):
            if active_provider == "openrouter":
                pred, status, wait_sec, q_info = query_openrouter_model_with_retry(
                    openrouter_key, m_name, prompt, cid, timeout_sec=DEFAULT_TIMEOUT_SEC
                )
            else:
                pred, status, wait_sec, q_info = query_model_with_retry(
                    client, m_name, prompt, cid, timeout_sec=DEFAULT_TIMEOUT_SEC
                )

            model_preds.append(pred)
            model_statuses.append(status)
            if wait_sec > 0:
                total_rate_waits += 1
                total_wait_duration += wait_sec
            if q_info.get("is_exhausted"):
                quota_hit = q_info
                break
            time.sleep(2)  # Polite pause

        while len(model_preds) < 3:
            model_preds.append({
                "clause_id": cid,
                "labels": [],
                "confidence": 0.0,
                "reason": f"Model slot unavailable (Quota/Error stopped: {quota_hit['model'] if quota_hit else 'None'})"
            })
            model_statuses.append("API_QUOTA_EXHAUSTED" if (quota_hit and quota_hit.get("is_exhausted")) else "API_ERROR")

        ensemble_res = compute_ensemble_metrics(model_preds, model_statuses, min_models=min_models_needed)

        if ensemble_res["pseudo_label_quality"] in ["API_ERROR", "API_QUOTA_EXHAUSTED"]:
            api_errors_session += 1
            consecutive_api_errors += 1
        else:
            consecutive_api_errors = 0
            if ensemble_res["pseudo_label_quality"] == "DISAGREEMENT":
                disagreements_session += 1

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
            "reason": f"[{active_provider}] {ensemble_res['reason']}"[:250],
            "label_source": "llm_pseudo_label",
        }

        # Update existing records dictionary (unique clause_id key)
        existing_records[cid] = record
        processed_this_session += 1

        status_tag = ensemble_res["pseudo_label_quality"]
        print(f"[{idx:03d}/{total_queue_len:03d}] clause_id {cid}... {status_tag} (Conf: {ensemble_res['average_confidence']}, Agreement: {ensemble_res['agreement_score']})", flush=True)

        # Batch flush to disk after every row for maximum safety
        save_current_llm_labels(output_path, existing_records, fieldnames)

        # Periodically update the resume queue file by removing processed valid clauses
        if ensemble_res["pseudo_label_quality"] in ["HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE", "DISAGREEMENT"]:
            already_valid_ids.add(cid)

        # Update remaining queue on disk
        df_curr_queue = pd.read_csv(queue_path)
        rem_queue = df_curr_queue[~df_curr_queue["clause_id"].isin(already_valid_ids)]
        rem_queue.to_csv(queue_path, index=False)

        if quota_hit:
            print("\n=======================================================", flush=True)
            print("  QUOTA EXHAUSTED", flush=True)
            print(f"  Model: {quota_hit['model']}", flush=True)
            print(f"  Retry after: {quota_hit['retry_delay']} seconds (~{quota_hit['retry_delay']/3600:.2f} hours)", flush=True)
            print(f"  Successfully saved records: {len(already_valid_ids)}", flush=True)
            print(f"  Remaining queue: {len(rem_queue)}", flush=True)
            print("=======================================================\n", flush=True)
            print("Safely stopping current batch. Releasing lock. Progress is fully preserved.\n", flush=True)
            break

        if idx % 10 == 0 or idx == total_queue_len:
            rem = total_queue_len - idx
            print(f"\n--- PROGRESS SUMMARY [{idx}/{total_queue_len}] ---")
            print(f"  Completed in session : {processed_this_session}")
            print(f"  Remaining in target  : {rem}")
            print(f"  API Errors (session) : {api_errors_session}")
            print(f"  Disagreements (sess) : {disagreements_session}")
            print(f"  Rate-Limit Waits     : {total_rate_waits} (Total Wait: {total_wait_duration:.1f}s)")
            print(f"-----------------------------------------\n", flush=True)

        if consecutive_api_errors >= MAX_CONSECUTIVE_API_ERRORS:
            print(f"\n[RATE_LIMIT_STOP] Encountered {consecutive_api_errors} consecutive API_ERROR responses. Stopping safely to avoid hammering LLM API.", flush=True)
            break

    generate_final_reports(output_path, total_v3)



class ProcessLock:
    def __init__(self, lockfile_path: Path):
        self.lockfile_path = lockfile_path
        self.fp = None

    def acquire(self) -> bool:
        try:
            self.fp = open(self.lockfile_path, "a+")
            self.fp.seek(0)
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(self.fp.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fp.seek(0)
            self.fp.truncate()
            self.fp.write(str(os.getpid()))
            self.fp.flush()
            return True
        except (IOError, OSError, ImportError):
            if self.fp:
                try:
                    self.fp.close()
                except Exception:
                    pass
                self.fp = None
            return False

    def release(self):
        if self.fp:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    self.fp.seek(0)
                    msvcrt.locking(self.fp.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
                self.fp.close()
            except Exception:
                pass
            self.fp = None
            try:
                if self.lockfile_path.exists():
                    self.lockfile_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module 5B Resume Runner — document-diversity-aware pseudo-labeling pipeline."
    )
    parser.add_argument(
        "--max-clauses", type=int, default=None,
        help="Maximum number of clauses to process in this run."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the next N clauses that WOULD be processed (document diversity order) without making any LLM API calls."
    )
    parser.add_argument(
        "--provider", type=str, choices=["gemini", "openrouter"], default=None,
        help="Specify LLM provider ('gemini' or 'openrouter'). Overrides LLM_PROVIDER environment variable."
    )
    parser.add_argument(
        "--test-openrouter", action="store_true",
        help="Run a 1-request connectivity test against OpenRouter using model google/gemini-3.6-flash."
    )
    args = parser.parse_args()

    if args.test_openrouter:
        success = test_openrouter_connectivity()
        sys.exit(0 if success else 1)

    if args.dry_run:
        # Dry-run does not need a lock — no files are modified.
        run_resume_pipeline(max_clauses=args.max_clauses, dry_run=True, provider=args.provider)
        sys.exit(0)

    lock_file = ROOT / "data/annotations/.resume_runner.lock"
    lock = ProcessLock(lock_file)
    if not lock.acquire():
        print(f"[ERROR] Another instance of resume_runner.py is already running (Lock: {lock_file}). Exiting cleanly.", flush=True)
        sys.exit(1)
    try:
        run_resume_pipeline(max_clauses=args.max_clauses, dry_run=False, provider=args.provider)
    finally:
        lock.release()



