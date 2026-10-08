import json
import random
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

APPROVED_CATEGORIES = [
    'Party Identification', 'Purpose', 'NDA Type', 'Definition of Confidential Information',
    'Confidentiality Obligations', 'Authorized Disclosure', 'Non-Confidential Information',
    'Liability for Damages', 'Competition Rights', 'Term and Termination',
    'Intellectual Property', 'Employees', 'Governing Law and Jurisdiction', 'Additional Information'
]

VALID_QUALITY_BUCKETS = {'HIGH_CONFIDENCE', 'MEDIUM_CONFIDENCE', 'LOW_CONFIDENCE', 'DISAGREEMENT'}


def load_valid_data(input_csv: Path):
    df = pd.read_csv(input_csv)
    df_valid = df[df['pseudo_label_quality'].isin(VALID_QUALITY_BUCKETS)].copy()
    df_valid = df_valid.drop_duplicates(subset=['clause_id'], keep='first').copy()

    def parse_labels(raw):
        try:
            val = json.loads(raw) if isinstance(raw, str) else []
            return [c for c in val if c in APPROVED_CATEGORIES]
        except Exception:
            return []

    df_valid['parsed_labels'] = df_valid['final_pseudo_labels'].apply(parse_labels)
    df_valid['doc_prefix'] = df_valid['document_id'].astype(str)
    return df_valid


def get_doc_profiles(df_valid):
    doc_groups = df_valid.groupby('doc_prefix')
    doc_info = []
    for doc_id, group in doc_groups:
        cat_set = set()
        for cats in group['parsed_labels']:
            cat_set.update(cats)
        doc_info.append({
            'doc_id': doc_id,
            'num_clauses': len(group),
            'categories': cat_set,
            'num_categories': len(cat_set)
        })
    return doc_info


def evaluate_partition(df_valid, train_docs, val_docs, test_docs, strategy_name):
    overlap_tv = train_docs.intersection(val_docs)
    overlap_tt = train_docs.intersection(test_docs)
    overlap_vt = val_docs.intersection(test_docs)
    leakage = len(overlap_tv) + len(overlap_tt) + len(overlap_vt)

    train_df = df_valid[df_valid['doc_prefix'].isin(train_docs)]
    val_df = df_valid[df_valid['doc_prefix'].isin(val_docs)]
    test_df = df_valid[df_valid['doc_prefix'].isin(test_docs)]

    def get_cat_sup(df_sub):
        sup = {c: 0 for c in APPROVED_CATEGORIES}
        for cats in df_sub['parsed_labels']:
            for c in cats:
                sup[c] += 1
        return sup

    train_sup = get_cat_sup(train_df)
    val_sup = get_cat_sup(val_df)
    test_sup = get_cat_sup(test_df)

    train_cats_present = sum(1 for c, cnt in train_sup.items() if cnt > 0)
    val_cats_present = sum(1 for c, cnt in val_sup.items() if cnt > 0)
    test_cats_present = sum(1 for c, cnt in test_sup.items() if cnt > 0)

    missing_train = [c for c, cnt in train_sup.items() if cnt == 0]
    missing_val = [c for c, cnt in val_sup.items() if cnt == 0]
    missing_test = [c for c, cnt in test_sup.items() if cnt == 0]

    return {
        'strategy': strategy_name,
        'train_docs': len(train_docs),
        'val_docs': len(val_docs),
        'test_docs': len(test_docs),
        'train_clauses': len(train_df),
        'val_clauses': len(val_df),
        'test_clauses': len(test_df),
        'train_cats': train_cats_present,
        'val_cats': val_cats_present,
        'test_cats': test_cats_present,
        'missing_train': missing_train,
        'missing_val': missing_val,
        'missing_test': missing_test,
        'leakage': leakage,
        'train_sup': train_sup,
        'val_sup': val_sup,
        'test_sup': test_sup
    }


def run_all_strategies(input_csv: Path):
    print('=======================================================')
    print('  MODULE 9 - DOCUMENT-DISJOINT SPLIT EVALUATION')
    print('=======================================================')
    df_valid = load_valid_data(input_csv)
    doc_info = get_doc_profiles(df_valid)
    print('Total Valid Clauses: %d across %d documents\n' % (len(df_valid), len(doc_info)))

    results = []

    # Strategy A: Ratio-Based Split (70% Train, 15% Val, 15% Test by docs/clauses)
    docs_by_size = sorted(doc_info, key=lambda x: x['num_clauses'], reverse=True)
    doc_ids_by_size = [d['doc_id'] for d in docs_by_size]
    strA_train = set(doc_ids_by_size[:5])
    strA_val = set([doc_ids_by_size[5]])
    strA_test = set([doc_ids_by_size[6]])
    results.append(evaluate_partition(df_valid, strA_train, strA_val, strA_test, 'Strategy A (70/15/15 Ratio)'))

    # Strategy B: Ratio-Based Split (60% Train, 20% Val, 20% Test)
    strB_train = set(doc_ids_by_size[:4])
    strB_val = set(doc_ids_by_size[4:6])
    strB_test = set([doc_ids_by_size[6]])
    results.append(evaluate_partition(df_valid, strB_train, strB_val, strB_test, 'Strategy B (60/20/20 Ratio)'))

    # Strategy C: Greedy Category Maximization
    docs_by_cat = sorted(doc_info, key=lambda x: (x['num_categories'], x['num_clauses']), reverse=True)
    doc_ids_by_cat = [d['doc_id'] for d in docs_by_cat]
    strC_test = set([doc_ids_by_cat[0]]) # 0859334b (13 cats)
    strC_val = set([doc_ids_by_cat[1], doc_ids_by_cat[2]]) # 0b59dfc4 (10 cats), 293f5937 (6 cats)
    strC_train = set(doc_ids_by_cat[3:])
    results.append(evaluate_partition(df_valid, strC_train, strC_val, strC_test, 'Strategy C (Category Maximization)'))

    # Strategy D: Train Category Focus
    strD_train = set([doc_ids_by_cat[0]]) # Give 13-cat doc to Train
    strD_val = set([doc_ids_by_cat[1], doc_ids_by_cat[3]])
    strD_test = set(doc_ids_by_cat[2:]).difference(strD_val)
    results.append(evaluate_partition(df_valid, strD_train, strD_val, strD_test, 'Strategy D (Train Category Focus)'))

    print('%-35s | %-12s | %-12s | %-12s | %-15s | Leakage' % ('Strategy Name', 'Train (D/C/Cat)', 'Val (D/C/Cat)', 'Test (D/C/Cat)', 'Missing in Test'))
    print('-' * 110)
    for r in results:
        tr_str = '%d/%d/%d' % (r['train_docs'], r['train_clauses'], r['train_cats'])
        va_str = '%d/%d/%d' % (r['val_docs'], r['val_clauses'], r['val_cats'])
        te_str = '%d/%d/%d' % (r['test_docs'], r['test_clauses'], r['test_cats'])
        miss_cnt = len(r['missing_test'])
        print('%-35s | %-12s | %-12s | %-12s | %d categories     | %d' % (r['strategy'], tr_str, va_str, te_str, miss_cnt, r['leakage']))

    print('\nDetailed Category Support Breakdown Per Strategy:\n')
    for r in results:
        print('=== %s ===' % r['strategy'])
        print('  Train missing categories (%d): %s' % (len(r['missing_train']), r['missing_train']))
        print('  Val missing categories   (%d): %s' % (len(r['missing_val']), r['missing_val']))
        print('  Test missing categories  (%d): %s' % (len(r['missing_test']), r['missing_test']))
        print()

    return results


def main():
    input_csv = ROOT / 'data' / 'annotations' / 'llm_pseudo_labels.csv'
    run_all_strategies(input_csv)


if __name__ == '__main__':
    main()
