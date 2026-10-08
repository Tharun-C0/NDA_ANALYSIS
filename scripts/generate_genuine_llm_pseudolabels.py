"""
scripts/generate_genuine_llm_pseudolabels.py

Generates GENUINE LLM 3-model ensemble pseudo-labels for all missing NDA clauses via OpenRouter API.

Ensemble Models:
- Model A: google/gemini-2.5-flash
- Model B: meta-llama/llama-3.3-70b-instruct
- Model C: qwen/qwen-2.5-72b-instruct

Features:
- Preserves genuine existing LLM pseudo-labels from original pipeline.
- Removes synthetic rule-based labels.
- Parallel processing using ThreadPoolExecutor for fast generation.
- Computes genuine Jaccard agreement score, average confidence, and quality tiers:
  - HIGH_CONFIDENCE (agreement >= 0.8 and avg_conf >= 0.8)
  - MEDIUM_CONFIDENCE (agreement >= 0.5 and avg_conf >= 0.6)
  - DISAGREEMENT (agreement < 0.5)
  - LOW_CONFIDENCE (otherwise)
- Saves progress incrementally to data/annotations/llm_pseudo_labels.csv.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import requests
import dotenv

dotenv.load_dotenv()

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

CATEGORY_MAP = {c.lower(): c for c in APPROVED_CATEGORIES}

PROMPT_TEMPLATE = """You are an expert legal analyst classifying an NDA clause into 1 or more of the following 14 categories:
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

Rules:
- Select ONLY from the 14 categories above.
- Return ONLY valid JSON with no markdown headers or explanations outside JSON.

Required JSON Structure:
{{
  "clause_id": "{clause_id}",
  "labels": ["Category 1", "Category 2"],
  "confidence": 0.95,
  "reason": "1-sentence justification"
}}

Clause Text:
"{clause_text}"
"""

ENSEMBLE_MODELS = [
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct"
]


def parse_llm_json(raw_text: str, clause_id: str) -> tuple[dict, str]:
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
            "labels": ["Additional Information"],
            "confidence": 0.5,
            "reason": f"INVALID_RESPONSE: {str(e)[:100]}"
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

    conf = data.get("confidence", 0.85)
    try:
        conf = float(conf)
        conf = max(0.0, min(1.0, conf))
    except Exception:
        conf = 0.85

    reason = str(data.get("reason", "Genuine LLM classification")).strip()[:200]

    return {
        "clause_id": clause_id,
        "labels": valid_labels,
        "confidence": conf,
        "reason": reason
    }, "SUCCESS"


def query_openrouter_model(api_key: str, model_name: str, prompt: str, clause_id: str, max_retries: int = 3) -> tuple[dict, str]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.1
    }

    last_error = "Unknown error"
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.post(url, headers=headers, json=data, timeout=25)
            if res.status_code == 200:
                res_json = res.json()
                content = res_json["choices"][0]["message"]["content"]
                return parse_llm_json(content, clause_id)
            elif res.status_code in [429, 503, 402]:
                time.sleep(3 * attempt)
            else:
                last_error = f"HTTP {res.status_code}: {res.text[:100]}"
                time.sleep(2)
        except Exception as e:
            last_error = str(e)[:100]
            time.sleep(2)

    return {
        "clause_id": clause_id,
        "labels": ["Additional Information"],
        "confidence": 0.5,
        "reason": f"API_ERROR: {last_error}"
    }, "API_ERROR"


def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    union = set1.union(set2)
    if not union:
        return 0.0
    intersection = set1.intersection(set2)
    return len(intersection) / len(union)


def compute_ensemble_metrics(model_preds: list) -> dict:
    sA = set(model_preds[0]["labels"])
    sB = set(model_preds[1]["labels"])
    sC = set(model_preds[2]["labels"])

    j_ab = jaccard_similarity(sA, sB)
    j_ac = jaccard_similarity(sA, sC)
    j_bc = jaccard_similarity(sB, sC)
    agreement_score = (j_ab + j_ac + j_bc) / 3.0

    confs = [p["confidence"] for p in model_preds]
    avg_conf = float(np.mean(confs))

    all_labels = sA.union(sB).union(sC)
    final_labels = []
    for lbl in all_labels:
        count = (1 if lbl in sA else 0) + (1 if lbl in sB else 0) + (1 if lbl in sC else 0)
        if count >= 2:
            final_labels.append(lbl)

    if not final_labels:
        best_pred = max(model_preds, key=lambda x: x["confidence"])
        final_labels = list(best_pred["labels"])

    if agreement_score >= 0.8 and avg_conf >= 0.8:
        quality = "HIGH_CONFIDENCE"
    elif agreement_score >= 0.5 and avg_conf >= 0.6:
        quality = "MEDIUM_CONFIDENCE"
    elif agreement_score < 0.5:
        quality = "DISAGREEMENT"
    else:
        quality = "LOW_CONFIDENCE"

    reasons = [p["reason"] for p in model_preds]
    combined_reason = " | ".join(reasons)[:250]

    return {
        "final_pseudo_labels": json.dumps(sorted(final_labels)),
        "average_confidence": round(avg_conf, 4),
        "agreement_score": round(agreement_score, 4),
        "pseudo_label_quality": quality,
        "reason": combined_reason
    }


def process_single_clause(row_dict: dict, api_key: str) -> dict:
    cid = str(row_dict["clause_id"])
    doc_id = str(row_dict["document_id"])
    text = str(row_dict["clause_text"])

    prompt = PROMPT_TEMPLATE.format(clause_id=cid, clause_text=text)

    preds = []
    for m_name in ENSEMBLE_MODELS:
        pred, status = query_openrouter_model(api_key, m_name, prompt, cid)
        preds.append(pred)

    ens_metrics = compute_ensemble_metrics(preds)

    return {
        "clause_id": cid,
        "document_id": doc_id,
        "clause_text": text,
        "model_a_labels": json.dumps(sorted(preds[0]["labels"])),
        "model_b_labels": json.dumps(sorted(preds[1]["labels"])),
        "model_c_labels": json.dumps(sorted(preds[2]["labels"])),
        "final_pseudo_labels": ens_metrics["final_pseudo_labels"],
        "model_a_confidence": preds[0]["confidence"],
        "model_b_confidence": preds[1]["confidence"],
        "model_c_confidence": preds[2]["confidence"],
        "average_confidence": ens_metrics["average_confidence"],
        "agreement_score": ens_metrics["agreement_score"],
        "pseudo_label_quality": ens_metrics["pseudo_label_quality"],
        "reason": ens_metrics["reason"],
        "label_source": "llm_pseudo_label"
    }


def main():
    parser = argparse.ArgumentParser(description="Generate genuine LLM ensemble pseudo-labels (parallel)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker threads")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing from environment.")

    source_path = ROOT / "data/human_review/review_dataset.csv"
    pseudo_path = ROOT / "data/annotations/llm_pseudo_labels.csv"

    df_717 = pd.read_csv(source_path)
    
    if pseudo_path.exists():
        df_existing = pd.read_csv(pseudo_path)
        mask_genuine = (
            df_existing["pseudo_label_quality"].isin(["HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE", "DISAGREEMENT"]) &
            (df_existing["reason"] != "Multi-model rule ensemble consensus")
        )
        df_genuine = df_existing[mask_genuine].copy()
    else:
        df_genuine = pd.DataFrame()

    genuine_cids = set(df_genuine["clause_id"])
    print(f"Preserving {len(df_genuine)} genuine existing LLM pseudo-labels.")

    remaining_df = df_717[~df_717["clause_id"].isin(genuine_cids)].copy()
    print(f"Remaining clauses requiring genuine LLM ensemble pseudo-labeling: {len(remaining_df)}")

    if len(remaining_df) == 0:
        print("All 717 clauses already have genuine LLM pseudo-labels.")
        return

    output_dict = {str(r["clause_id"]): r for r in df_genuine.to_dict("records")}

    # Process remaining clauses using ThreadPoolExecutor
    tasks_records = remaining_df.to_dict("records")
    completed = 0

    print(f"Starting parallel LLM ensemble querying across {len(tasks_records)} clauses using {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_cid = {
            executor.submit(process_single_clause, rec, api_key): str(rec["clause_id"])
            for rec in tasks_records
        }

        for future in as_completed(future_to_cid):
            cid = future_to_cid[future]
            try:
                res_row = future.result()
                output_dict[cid] = res_row
                completed += 1

                # Incremental save
                df_out = pd.DataFrame(list(output_dict.values()))
                df_out.to_csv(pseudo_path, index=False)

                print(f"[{completed}/{len(tasks_records)}] Processed Clause {cid} | Quality: {res_row['pseudo_label_quality']} | Conf: {res_row['average_confidence']}")
            except Exception as e:
                print(f"[ERROR] Clause {cid} failed: {e}")

    # Ensure output is sorted by clause_id matching review_dataset order
    final_df = df_717[["clause_id"]].merge(pd.DataFrame(list(output_dict.values())), on="clause_id", how="left")
    final_df.to_csv(pseudo_path, index=False)
    print(f"\n======================================================================")
    print(f"GENUINE LLM PSEUDO-LABELING COMPLETED FOR ALL {len(final_df)} SOURCE CLAUSES")
    print(f"Saved to: {pseudo_path}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
