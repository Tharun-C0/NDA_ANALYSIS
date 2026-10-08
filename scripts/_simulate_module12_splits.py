"""
scripts/_simulate_module12_splits.py

Module 12 — Task 3 Benchmark Simulation Script.
Evaluates Strategies A, B, C, D on current 237 valid pseudo-labeled clauses across 11 documents.
"""

import json
from pathlib import Path
import pandas as pd

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

def simulate_splits():
    df = pd.read_csv(ROOT / "data/annotations/llm_pseudo_labels.csv")
    dfv = df[df["pseudo_label_quality"].isin(["HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE", "DISAGREEMENT"])].copy()

    # Pre-parse labels for each clause
    clause_cats = {}
    for _, r in dfv.iterrows():
        try:
            clause_cats[r["clause_id"]] = set(json.loads(r["final_pseudo_labels"]))
        except Exception:
            clause_cats[r["clause_id"]] = set()

    # Document-level categories and clause counts
    doc_info = {}
    for doc_id, group in dfv.groupby("document_id"):
        doc_cats = set()
        for cid in group["clause_id"]:
            doc_cats.update(clause_cats[cid])
        doc_info[doc_id] = {
            "prefix": doc_id[:8],
            "full_id": doc_id,
            "clauses": len(group),
            "categories": doc_cats,
            "num_cats": len(doc_cats)
        }

    all_docs = list(doc_info.keys())
    print(f"Loaded {len(all_docs)} unique documents ({len(dfv)} valid clauses).")

    # Strategy A: Current Split (based on prepare_classification_dataset.py)
    # Train: ['0a42e159...', '0859334b...', '266929af...']
    # Val: ['0b59dfc4...', '293f5937...']
    # Test: ['2268c5d1...', '247166e0...', '3504e06a...', '4fd432d8...', '5180f107...', '53c8f90c...']
    strat_a_train = [d for d in all_docs if d[:8] in ["0a42e159", "0859334b", "266929af"]]
    strat_a_val = [d for d in all_docs if d[:8] in ["0b59dfc4", "293f5937"]]
    strat_a_test = [d for d in all_docs if d not in strat_a_train and d not in strat_a_val]

    # Strategy B: Maximize Test Coverage (Doc 0859334b + 0b59dfc4 + 5180f107 in Test)
    strat_b_test = [d for d in all_docs if d[:8] in ["0859334b", "0b59dfc4", "5180f107"]]
    strat_b_val = [d for d in all_docs if d[:8] in ["293f5937", "53c8f90c"]]
    strat_b_train = [d for d in all_docs if d not in strat_b_test and d not in strat_b_val]

    # Strategy C: Maximize Balanced Coverage Across Train/Val/Test
    # Train: 0859334b (13 cats), 0a42e159, 2268c5d1, 247166e0, 266929af (5 docs)
    # Val: 0b59dfc4 (10 cats), 293f5937, 3504e06a (3 docs)
    # Test: 5180f107 (13 cats), 53c8f90c (10 cats), 4fd432d8 (8 cats) (3 docs)
    strat_c_train = [d for d in all_docs if d[:8] in ["0859334b", "0a42e159", "2268c5d1", "247166e0", "266929af"]]
    strat_c_val = [d for d in all_docs if d[:8] in ["0b59dfc4", "293f5937", "3504e06a"]]
    strat_c_test = [d for d in all_docs if d[:8] in ["5180f107", "53c8f90c", "4fd432d8"]]

    # Strategy D: Equalized Document Holdout (Train 5 docs, Val 3 docs, Test 3 docs)
    # Train: 0859334b, 0a42e159, 2268c5d1, 247166e0, 266929af
    # Val: 0b59dfc4, 293f5937, 53c8f90c
    # Test: 5180f107, 3504e06a, 4fd432d8
    strat_d_train = [d for d in all_docs if d[:8] in ["0859334b", "0a42e159", "2268c5d1", "247166e0", "266929af"]]
    strat_d_val = [d for d in all_docs if d[:8] in ["0b59dfc4", "293f5937", "53c8f90c"]]
    strat_d_test = [d for d in all_docs if d[:8] in ["5180f107", "3504e06a", "4fd432d8"]]

    strategies = [
        ("Strategy A (Current Split)", strat_a_train, strat_a_val, strat_a_test),
        ("Strategy B (Maximize Test)", strat_b_train, strat_b_val, strat_b_test),
        ("Strategy C (Balanced Multi-Split)", strat_c_train, strat_c_val, strat_c_test),
        ("Strategy D (Holdout Equalization)", strat_d_train, strat_d_val, strat_d_test)
    ]

    for name, tr_docs, val_docs, te_docs in strategies:
        tr_clauses = sum(doc_info[d]["clauses"] for d in tr_docs)
        val_clauses = sum(doc_info[d]["clauses"] for d in val_docs)
        te_clauses = sum(doc_info[d]["clauses"] for d in te_docs)

        tr_cats = set().union(*(doc_info[d]["categories"] for d in tr_docs)) if tr_docs else set()
        val_cats = set().union(*(doc_info[d]["categories"] for d in val_docs)) if val_docs else set()
        te_cats = set().union(*(doc_info[d]["categories"] for d in te_docs)) if te_docs else set()

        missing_tr = sorted(list(set(APPROVED_CATEGORIES) - tr_cats))
        missing_val = sorted(list(set(APPROVED_CATEGORIES) - val_cats))
        missing_te = sorted(list(set(APPROVED_CATEGORIES) - te_cats))

        # Check minority categories (Liability for Damages, Competition Rights, IP, Governing Law)
        minority_cats = ["Liability for Damages", "Competition Rights", "Intellectual Property", "Governing Law and Jurisdiction"]
        min_tr = [c for c in minority_cats if c in tr_cats]
        min_val = [c for c in minority_cats if c in val_cats]
        min_te = [c for c in minority_cats if c in te_cats]

        tv_leak = len(set(tr_docs).intersection(set(val_docs)))
        tt_leak = len(set(tr_docs).intersection(set(te_docs)))
        vt_leak = len(set(val_docs).intersection(set(te_docs)))
        leakage_status = "ZERO (PASS)" if (tv_leak + tt_leak + vt_leak) == 0 else f"LEAKAGE ({tv_leak}/{tt_leak}/{vt_leak})"

        print(f"\n=== {name} ===")
        print(f"  Train: {len(tr_docs)} docs, {tr_clauses} clauses | Present: {len(tr_cats)}/14 | Missing: {missing_tr}")
        print(f"  Val:   {len(val_docs)} docs, {val_clauses} clauses | Present: {len(val_cats)}/14 | Missing: {missing_val}")
        print(f"  Test:  {len(te_docs)} docs, {te_clauses} clauses | Present: {len(te_cats)}/14 | Missing: {missing_te}")
        print(f"  Minority Cats Present -> Train: {len(min_tr)}/4, Val: {len(min_val)}/4, Test: {len(min_te)}/4")
        print(f"  Leakage Status: {leakage_status}")

if __name__ == "__main__":
    simulate_splits()
