import os
import json
import time
import random
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import argparse

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
except ImportError:
    genai = None

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "human_review" / "priority_review.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "human_review"
PSEUDO_LABELS_PATH = os.path.join(OUTPUT_DIR, "llm_pseudo_labels.csv")
REVIEW_REQUIRED_PATH = os.path.join(OUTPUT_DIR, "human_review_required.csv")

REPORTS_DIR = PROJECT_ROOT / "reports" / "human_review"
STATS_PATH = os.path.join(REPORTS_DIR, "llm_review_statistics.csv")
LOG_PATH = os.path.join(REPORTS_DIR, "llm_review_log.csv")

random.seed(42)

def init_gemini():
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing in .env")
    if genai is None:
        raise ImportError("google-genai is not installed")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=60000, retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )


def safe_error(error):
    message = f"{type(error).__name__}: {error}"
    key = os.getenv("GEMINI_API_KEY")
    if key:
        message = message.replace(key, "[REDACTED]")
    return re.sub(r"AIza[\w-]+", "[REDACTED]", message)


def daily_quota_exhausted(error):
    message = str(error).lower()
    return any(marker in message for marker in (
        "perday", "per_day", "per day", "daily", "per-day",
        "requests/day", "tokens/day",
    )) and any(marker in message for marker in (
        "429", "resource_exhausted", "quota", "exhausted",
    ))


def call_gemini(client, c_type, length, current_text, prev_text, next_text):
    prompt = f"""You are assisting with preliminary NDA clause segmentation.

Your task is ONLY to determine the structural status of the provided text.

You are NOT performing legal analysis.
You are NOT assigning NDA clause categories.
You are NOT determining whether a contract is legally valid.
You are NOT providing legal advice.

Choose exactly one:
KEEP
MERGE_WITH_NEXT
MERGE_WITH_PREVIOUS
SPLIT
REMOVE_NON_LEGAL
REVIEW

KEEP: Meaningful contractual/legal content.
MERGE_WITH_NEXT: Current segment is clearly incomplete and continues into the next segment.
MERGE_WITH_PREVIOUS: Current segment clearly continues the previous segment.
SPLIT: One segment contains multiple clearly separate clauses.
REMOVE_NON_LEGAL: Obvious metadata, page numbers, SEC exhibit identifiers, isolated document titles, repeated headers/footers, company names appearing alone, or obvious OCR artifacts.
REVIEW: Uncertain or ambiguous.

CURRENT SEGMENT (Type: {c_type}, Length: {length}):
{current_text}

PREVIOUS SEGMENT:
{prev_text}

NEXT SEGMENT:
{next_text}
"""
    model_name = "models/gemini-3.6-flash"
    print(f"Model configured: {model_name}")
    
    attempts_429 = 0
    attempts_503 = 0
    attempts_timeout = 0
    attempts_total = 0
    
    while True:
        attempts_total += 1
        try:
            print("Calling Gemini API...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "suggested_action": {
                                "type": "string",
                                "enum": ["KEEP", "MERGE_WITH_NEXT", "MERGE_WITH_PREVIOUS", "SPLIT", "REMOVE_NON_LEGAL", "REVIEW"]
                            },
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                            "needs_context": {"type": "boolean"}
                        },
                        "required": ["suggested_action", "confidence", "reason", "needs_context"]
                    }
                )
            )
            print("Gemini response received")
            result = json.loads(response.text)
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
            if result.get("suggested_action") not in {
                "KEEP", "MERGE_WITH_NEXT", "MERGE_WITH_PREVIOUS", "SPLIT",
                "REMOVE_NON_LEGAL", "REVIEW",
            }:
                raise ValueError("Invalid suggested_action")
            confidence = result.get("confidence")
            if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                    or not 0 <= confidence <= 1):
                raise ValueError("Confidence must be between 0 and 1")
            if not isinstance(result.get("reason"), str) or not result["reason"].strip():
                raise ValueError("A nonempty reason is required")
            if not isinstance(result.get("needs_context"), bool):
                raise ValueError("needs_context must be boolean")
            return result, None
        except Exception as e:
            error = safe_error(e)
            print(error)
            err_str = error.lower()
            if daily_quota_exhausted(error):
                print("Daily quota exhausted; no retry.")
                return None, error
            if attempts_total >= 3:
                return None, error
            
            if "timeout" in err_str:
                attempts_timeout += 1
                if attempts_timeout > 2:
                    return None, error
                print(f"API TIMEOUT. Waiting 10 seconds (attempt {attempts_timeout}/2)")
                time.sleep(10)
                continue
                
            if "503" in err_str or "model_capacity_exhausted" in err_str:
                attempts_503 += 1
                if attempts_503 > 2:
                    return None, error
                print(f"503 MODEL CAPACITY EXHAUSTED. Waiting 30 seconds (attempt {attempts_503}/2)")
                time.sleep(30)
                continue
                
            if "429" in err_str or "resource_exhausted" in err_str:
                attempts_429 += 1
                temporary = any(marker in err_str for marker in (
                    "perminute", "per_minute", "per minute", "rate limit",
                    "retrydelay", "retry_delay", "retry in",
                ))
                if attempts_429 >= 3 or not temporary:
                    return None, error
                print(f"429 TEMPORARY RATE LIMIT. Waiting 60 seconds (attempt {attempts_429 + 1}/3)")
                time.sleep(60)
                continue
                
            return None, error

def read_records(path):
    if not os.path.exists(path):
        return []
    try:
        return pd.read_csv(path, keep_default_na=False).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def append_records(path, rows):
    if not rows:
        return
    columns = []
    if os.path.exists(path):
        try:
            columns = list(pd.read_csv(path, nrows=0).columns)
        except pd.errors.EmptyDataError:
            pass
    frame = pd.DataFrame(rows)
    if not columns and path == REVIEW_REQUIRED_PATH:
        columns = list(dict.fromkeys(list(frame.columns) + ["ai_confidence_level", "needs_context"]))
        frame = frame.reindex(columns=columns, fill_value="")
        columns = []
    if columns:
        frame = frame.reindex(columns=columns, fill_value="")
    with open(path, "a", encoding="utf-8", newline="") as output:
        frame.to_csv(output, index=False, header=not columns)
        output.flush()
        os.fsync(output.fileno())


def successful_records(logs, pseudo_labels, review_required):
    records = {}
    for row in pseudo_labels + review_required:
        if (row.get("clause_id") and row.get("ai_suggested_action")
                and row.get("ai_confidence") not in (None, "")
                and row.get("review_method") == "llm_auto_pseudolabel"):
            records[row["clause_id"]] = {
                "suggested_action": row["ai_suggested_action"],
                "confidence": row["ai_confidence"],
            }
    for row in logs:
        if row.get("clause_id") and row.get("status") == "SUCCESS":
            records[row["clause_id"]] = row
    return records


def check_connectivity(client):
    print("One connectivity check: models/gemini-3.6-flash; timeout=60000 ms; no retries")
    started = time.monotonic()
    try:
        client.models.generate_content(
            model="models/gemini-3.6-flash",
            contents="Reply OK.",
            config=types.GenerateContentConfig(max_output_tokens=32),
        )
        print(f"Connectivity check succeeded in {time.monotonic() - started:.2f} seconds")
        return None
    except Exception as error:
        message = safe_error(error)
        print(message)
        return message


def process_segments(test_mode=False, dry_run=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH, keep_default_na=False)
    if df["clause_id"].eq("").any() or df["clause_id"].duplicated().any():
        raise ValueError("Priority clause IDs must be nonempty and unique")
    
    start_time_total = time.time()
    total_segments = len(df)
    
    pseudo_labels = read_records(PSEUDO_LABELS_PATH)
    review_required = read_records(REVIEW_REQUIRED_PATH)
    logs = read_records(LOG_PATH)
    successful_clauses = set(successful_records(logs, pseudo_labels, review_required))
    priority_ids = set(df["clause_id"])
    already_completed = len(successful_clauses & priority_ids)
    pending = df.loc[~df["clause_id"].isin(successful_clauses)]
    print(f"Total priority segments: {total_segments}")
    print(f"Already completed: {already_completed}; remaining: {len(pending)}")
    print("Existing SUCCESS IDs:", sorted({r["clause_id"] for r in logs if r.get("status") == "SUCCESS"}))
    if not pending.empty:
        print("First unprocessed:", pending.iloc[0]["clause_id"])
    if dry_run:
        return {"total_segments": total_segments, "already_completed": already_completed,
                "remaining": len(pending)}
        
    stats = {
        "total_segments": total_segments,
        "high_confidence": 0,
        "medium_confidence": 0,
        "low_confidence": 0,
        "ai_review": 0,
        "merge_cases": 0,
        "split_cases": 0,
        "random_audit_cases": 0,
        "human_review_required": 0,
        "api_failures": 0
    }
    
    processed_in_this_run = 0
    initial_log_count = len(logs)
    saved_counts = [len(pseudo_labels), len(review_required), len(logs)]
    run_status = "RUNNING"

    def checkpoint():
        for number, (path, records) in enumerate((
            (PSEUDO_LABELS_PATH, pseudo_labels),
            (REVIEW_REQUIRED_PATH, review_required),
            (LOG_PATH, logs),
        )):
            append_records(path, records[saved_counts[number]:])
            saved_counts[number] = len(records)
        successes = successful_records(logs, pseudo_labels, review_required)
        successes = {key: value for key, value in successes.items() if key in priority_ids}
        confidences = [float(row["confidence"]) for row in successes.values()]
        actions = [row["suggested_action"] for row in successes.values()]
        stats.update({
            "already_completed_before_run": already_completed,
            "newly_processed": processed_in_this_run,
            "successful": len(successes),
            "successful_this_run": len(successes) - already_completed,
            "remaining": total_segments - len(successes),
            "high_confidence": sum(value >= 0.90 for value in confidences),
            "medium_confidence": sum(0.70 <= value < 0.90 for value in confidences),
            "low_confidence": sum(value < 0.70 for value in confidences),
            "ai_review": actions.count("REVIEW"),
            "keep": actions.count("KEEP"),
            "remove_non_legal": actions.count("REMOVE_NON_LEGAL"),
            "merge_with_next": actions.count("MERGE_WITH_NEXT"),
            "merge_with_previous": actions.count("MERGE_WITH_PREVIOUS"),
            "split_cases": actions.count("SPLIT"),
            "human_review_required": len({row["clause_id"] for row in review_required
                                          if row.get("clause_id") in priority_ids}),
            "random_audit_cases": len({row["clause_id"] for row in review_required
                                       if "RANDOM_AUDIT" in row.get("review_reason", "")}),
            "api_failures": sum(row.get("status") in {"ERROR", "CONNECTIVITY_ERROR"} for row in logs),
            "api_failures_this_run": sum(row.get("status") in {"ERROR", "CONNECTIVITY_ERROR"}
                                         for row in logs[initial_log_count:]),
            "connectivity_failures_this_run": sum(row.get("status") == "CONNECTIVITY_ERROR"
                                                  for row in logs[initial_log_count:]),
            "total_processing_time_seconds": round(time.time() - start_time_total, 2),
            "run_status": run_status,
            "timestamp": datetime.now().isoformat(),
        })
        stats["merge_cases"] = stats["merge_with_next"] + stats["merge_with_previous"]
        stats["percentage_requiring_human_review"] = round(
            stats["human_review_required"] / total_segments * 100, 2
        ) if total_segments else 0.0
        temporary = str(STATS_PATH) + ".tmp"
        pd.DataFrame([stats]).to_csv(temporary, index=False)
        os.replace(temporary, STATS_PATH)

    if pending.empty:
        run_status = "COMPLETE"
        checkpoint()
        print(json.dumps(stats, indent=2))
        return stats

    context_df = pd.read_csv(OUTPUT_DIR / "review_dataset.csv", keep_default_na=False)
    context_rows = {(row["document_id"], row["clause_id"]): row["clause_text"]
                    for row in context_df.to_dict("records")}
    for row in pending.to_dict("records"):
        if context_rows.get((row["document_id"], row["clause_id"])) != row["clause_text"]:
            raise ValueError(f"Missing or inconsistent source context for {row['clause_id']}")

    client = None
    try:
        client = init_gemini()
        connectivity_error = check_connectivity(client)
    except Exception as error:
        connectivity_error = safe_error(error)
    if connectivity_error:
        run_status = "PAUSED_DAILY_QUOTA" if daily_quota_exhausted(connectivity_error) else "PAUSED_CONNECTIVITY"
        logs.append({"timestamp": datetime.now().isoformat(), "document_id": "", "clause_id": "",
                     "status": "CONNECTIVITY_ERROR", "suggested_action": "", "confidence": "",
                     "error": connectivity_error})
        checkpoint()
        if client is not None:
            client.close()
        print("Connectivity check failed. No segments submitted; no further API calls.")
        print(json.dumps(stats, indent=2))
        return stats
    
    for idx, row in df.iterrows():
        doc_id = row["document_id"]
        c_id = row["clause_id"]
        c_text = row["clause_text"]
        
        # Get prev/next
        prev_text = "N/A"
        next_text = "N/A"
        
        # Find prev in original df
        prefix, number = c_id.rsplit("_", 1)
        previous_id = f"{prefix}_{int(number) - 1:03d}"
        prev_text = context_rows.get((doc_id, previous_id), "N/A")
                
        # Find next in original df
        next_id = f"{prefix}_{int(number) + 1:03d}"
        next_text = context_rows.get((doc_id, next_id), "N/A")
                
        if c_id in successful_clauses:
            print(f"{c_id} -> SKIPPED (already successful)")
            continue
            
        print(f"Processing {idx+1}/{total_segments}\nSegment ID: {c_id}")
        
        start_t = time.time()
        res, err = call_gemini(client, row["segment_type"], row["length"], c_text, prev_text, next_text)
        end_t = time.time()
        processed_in_this_run += 1
        
        timestamp = datetime.now().isoformat()
        
        if err:
            logs.append({
                "timestamp": timestamp,
                "document_id": doc_id,
                "clause_id": c_id,
                "status": "ERROR",
                "suggested_action": "",
                "confidence": "",
                "error": err
            })
            
            row_dict = row.to_dict()
            row_dict["previous_text"] = prev_text
            row_dict["next_text"] = next_text
            row_dict["ai_suggested_action"] = ""
            row_dict["ai_confidence"] = ""
            row_dict["ai_reason"] = ""
            row_dict["review_reason"] = "API_FAILURE"
            row_dict["review_timestamp"] = timestamp
            review_required.append(row_dict)
            stats["api_failures"] += 1
            stats["human_review_required"] += 1
            if daily_quota_exhausted(err):
                run_status = "PAUSED_DAILY_QUOTA"
                checkpoint()
                break
            checkpoint()
            if test_mode and processed_in_this_run >= 1:
                break
            continue
            
        action = res.get("suggested_action")
        conf = res.get("confidence", 0)
        reason = res.get("reason", "")
        
        print(f"Segment ID: {c_id}")
        print(f"AI action: {action}")
        print(f"Confidence: {conf}")
        print(f"Reason: {reason}")
        print(f"API response time: {end_t - start_t:.2f} seconds")
        
        # Confidence level
        if conf >= 0.90:
            conf_level = "HIGH_CONFIDENCE"
            stats["high_confidence"] += 1
        elif conf >= 0.70:
            conf_level = "MEDIUM_CONFIDENCE"
            stats["medium_confidence"] += 1
        else:
            conf_level = "LOW_CONFIDENCE"
            stats["low_confidence"] += 1
            
        logs.append({
            "timestamp": timestamp,
            "document_id": doc_id,
            "clause_id": c_id,
            "status": "SUCCESS",
            "suggested_action": action,
            "confidence": conf,
            "error": ""
        })
        
        # Build base dictionary for both outputs
        base_row = row.to_dict()
        base_row["previous_text"] = prev_text
        base_row["next_text"] = next_text
        base_row["ai_suggested_action"] = action
        base_row["ai_confidence"] = conf
        base_row["ai_confidence_level"] = conf_level
        base_row["ai_reason"] = reason
        base_row["needs_context"] = res.get("needs_context", False)
        base_row["review_method"] = "llm_auto_pseudolabel"
        base_row["review_timestamp"] = timestamp
        base_row["reviewed"] = False
        base_row["human_action"] = ""
        base_row["human_is_valid_clause"] = ""
        base_row["human_corrected_text"] = ""
        base_row["human_notes"] = ""
        base_row["reviewer"] = ""
        
        needs_human = False
        review_reasons = []
        if base_row["needs_context"]:
            needs_human = True
            review_reasons.append("NEEDS_CONTEXT")
        
        if conf_level == "LOW_CONFIDENCE":
            needs_human = True
            review_reasons.append("LOW_CONFIDENCE")
            
        if action == "REVIEW":
            needs_human = True
            review_reasons.append("AI_REVIEW")
            stats["ai_review"] += 1
            
        if action in ["MERGE_WITH_NEXT", "MERGE_WITH_PREVIOUS"]:
            needs_human = True
            review_reasons.append("MERGE_DECISION")
            stats["merge_cases"] += 1
            
        if action == "SPLIT":
            needs_human = True
            review_reasons.append("SPLIT_DECISION")
            stats["split_cases"] += 1
            
        if not needs_human and conf_level == "HIGH_CONFIDENCE":
            if random.random() < 0.10: # 10% audit
                needs_human = True
                review_reasons.append("RANDOM_AUDIT")
                stats["random_audit_cases"] += 1
                
        if needs_human:
            stats["human_review_required"] += 1
            rev_row = base_row.copy()
            rev_row["review_reason"] = "; ".join(review_reasons)
            review_required.append(rev_row)
        else:
            pseudo_labels.append(base_row)
            
        successful_clauses.add(c_id)
        checkpoint()
        time.sleep(0.5) # rate limit
        
        if test_mode and processed_in_this_run >= 1:
            break

    if run_status == "RUNNING":
        run_status = "COMPLETE" if priority_ids <= successful_clauses else "PAUSED_WITH_PENDING"
    checkpoint()
    client.close()
    print(json.dumps(stats, indent=2))
    
    print("="*50)
    print("AUTOMATED LLM-ASSISTED REVIEW RESULTS")
    print("="*50)
    print(f"Total segments: {total_segments}")
    print(f"High-confidence: {stats['high_confidence']}")
    print(f"Medium-confidence: {stats['medium_confidence']}")
    print(f"Low-confidence: {stats['low_confidence']}")
    print(f"AI REVIEW: {stats['ai_review']}")
    print(f"Merge cases: {stats['merge_cases']}")
    print(f"Split cases: {stats['split_cases']}")
    print(f"Random audit: {stats['random_audit_cases']}")
    print(f"Human review required: {stats['human_review_required']}")
    
    end_time_total = time.time()
    print(f"Processing time: {end_time_total - start_time_total:.2f} seconds")
    
    print()
    print("Output locations:")
    print(f" - {PSEUDO_LABELS_PATH}")
    print(f" - {REVIEW_REQUIRED_PATH}")
    print(f" - {STATS_PATH}")
    print(f" - {LOG_PATH}")
    print("="*50)
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Check connectivity, then attempt one pending segment")
    parser.add_argument("--dry-run", action="store_true", help="Verify saved progress without API calls or output writes")
    args = parser.parse_args()
    process_segments(test_mode=args.test, dry_run=args.dry_run)
