"""
scripts/_generate_module13_queues.py

Generates Task 4 document priority CSV and Task 5 targeted clause queue CSV for Module 13.
"""

import re
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SIGNALS = {
    "Liability for Damages": [r"\bdamage", r"\bliabil", r"\bindemn", r"\bloss", r"\bbreach", r"\bpenalty", r"\bremedy", r"\bremedies"],
    "Competition Rights": [r"\bcompet", r"\bsolicit", r"\brival", r"\bexclusive", r"\bentic"],
    "Intellectual Property": [r"\bpatent", r"\bcopyright", r"\btrademark", r"\btrade secret", r"\binvention", r"\bintellectual property", r"\bproprietary", r"\bownership", r"\blicense"],
    "Governing Law and Jurisdiction": [r"\bgovern", r"\bjurisdiction", r"\bvenue", r"\barbitrat", r"\bcourt", r"\blaws of", r"\bstate of"]
}

def generate_queues():
    df_labels = pd.read_csv(ROOT / "data/annotations/llm_pseudo_labels.csv")
    df_valid = df_labels[df_labels["pseudo_label_quality"].isin(["HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE", "DISAGREEMENT"])].copy()
    valid_doc_ids = set(df_valid["document_id"].unique())

    df_queue = pd.read_csv(ROOT / "data/annotations/pseudolabel_resume_queue.csv")
    pending_queue = df_queue[~df_queue["clause_id"].isin(df_labels["clause_id"])].copy()
    unseen_doc_ids = sorted(list(set(pending_queue["document_id"].unique()) - valid_doc_ids))

    # 1. Generate Document Priority Ranking
    unseen_scores = []
    for doc_id in unseen_doc_ids:
        doc_clauses = pending_queue[pending_queue["document_id"] == doc_id]
        cnts = {cat: 0 for cat in SIGNALS}

        for _, r in doc_clauses.iterrows():
            txt = str(r["clause_text"]).lower()
            for cat, patterns in SIGNALS.items():
                hit = False
                for pat in patterns:
                    if re.search(pat, txt):
                        hit = True
                        break
                if hit:
                    cnts[cat] += 1

        distinct_target_cats = sum(1 for cat, count in cnts.items() if count > 0)
        total_signals = sum(cnts.values())
        score = (10 * distinct_target_cats) + (2 * total_signals) + (0.1 * len(doc_clauses))

        unseen_scores.append({
            "document_id": doc_id[:8],
            "full_document_id": doc_id,
            "pending_clause_count": len(doc_clauses),
            "liability_signal_count": cnts["Liability for Damages"],
            "competition_signal_count": cnts["Competition Rights"],
            "ip_signal_count": cnts["Intellectual Property"],
            "governing_law_signal_count": cnts["Governing Law and Jurisdiction"],
            "distinct_target_categories": distinct_target_cats,
            "priority_score": round(score, 2)
        })

    df_doc_rank = pd.DataFrame(unseen_scores).sort_values(by="priority_score", ascending=False)
    df_doc_rank["priority_rank"] = range(1, len(df_doc_rank) + 1)

    doc_rank_csv = ROOT / "reports/module13_targeted_document_priority.csv"
    doc_cols = ["priority_rank", "document_id", "full_document_id", "pending_clause_count",
                "liability_signal_count", "competition_signal_count", "ip_signal_count",
                "governing_law_signal_count", "distinct_target_categories", "priority_score"]
    df_doc_rank[doc_cols].to_csv(doc_rank_csv, index=False)
    print(f"Saved document priority ranking to: {doc_rank_csv}")

    # Map document score & rank to clauses
    doc_score_map = df_doc_rank.set_index("full_document_id")["priority_score"].to_dict()
    doc_rank_map = df_doc_rank.set_index("full_document_id")["priority_rank"].to_dict()

    # 2. Generate Targeted Clause Queue
    clause_rows = []
    for _, r in pending_queue.iterrows():
        doc_id = r["document_id"]
        if doc_id not in unseen_doc_ids:
            continue

        txt = str(r["clause_text"])
        txt_lower = txt.lower()

        suspected_cats = []
        all_matched_terms = []

        for cat, patterns in SIGNALS.items():
            cat_matched_terms = []
            for pat in patterns:
                matches = re.findall(pat, txt_lower)
                if matches:
                    cat_matched_terms.extend(matches)
            if cat_matched_terms:
                suspected_cats.append(cat)
                all_matched_terms.extend(sorted(list(set(cat_matched_terms))))

        if suspected_cats:
            clause_rows.append({
                "document_id": doc_id[:8],
                "clause_id": r["clause_id"],
                "clause_text": txt,
                "suspected_category_signal": "; ".join(suspected_cats),
                "signal_terms": "; ".join(sorted(list(set(all_matched_terms)))),
                "priority_score": doc_score_map.get(doc_id, 0.0),
                "priority_rank": doc_rank_map.get(doc_id, 99),
                "label_source": "heuristic_candidate_only"
            })

    df_clause_queue = pd.DataFrame(clause_rows).sort_values(by=["priority_rank", "priority_score"], ascending=[True, False])
    clause_queue_csv = ROOT / "data/annotations/module13_targeted_clause_queue.csv"
    df_clause_queue.to_csv(clause_queue_csv, index=False)
    print(f"Saved targeted clause queue ({len(df_clause_queue)} clauses) to: {clause_queue_csv}")

if __name__ == "__main__":
    generate_queues()
