import argparse
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
EXCLUDED_QUALITY_BUCKETS = {'API_ERROR', 'API_QUOTA_EXHAUSTED', 'INVALID_RESPONSE'}


def rebuild_benchmark(input_csv: Path, output_dir: Path, seed: int = 42, val_ratio: float = 0.2, test_ratio: float = 0.2, write: bool = False):
    print('=======================================================')
    print('  MODULE 8 - BENCHMARK RECONSTRUCTION ENGINE')
    print('=======================================================')
    print('Input file:', input_csv)
    print('Output directory:', output_dir)
    print('Mode:', '[WRITE TO DISK]' if write else '[DRY-RUN / REPORT ONLY]')
    print('Seed:', seed)

    if not input_csv.exists():
        raise FileNotFoundError(f'Input file not found: {input_csv}')

    df = pd.read_csv(input_csv)
    initial_count = len(df)
    print(f'Loaded {initial_count} total records from pseudo-label CSV.')

    df_valid = df[df['pseudo_label_quality'].isin(VALID_QUALITY_BUCKETS)].copy()
    excluded_count = initial_count - len(df_valid)
    print(f'Filtered out {excluded_count} invalid/error records.')

    df_valid = df_valid.drop_duplicates(subset=['clause_id'], keep='first').copy()
    print(f'Valid unique clauses remaining: {len(df_valid)}')

    def parse_labels(raw):
        try:
            val = json.loads(raw) if isinstance(raw, str) else []
            return [c for c in val if c in APPROVED_CATEGORIES]
        except Exception:
            return []

    df_valid['parsed_labels'] = df_valid['final_pseudo_labels'].apply(parse_labels)
    df_valid['doc_prefix'] = df_valid['document_id'].astype(str)
    doc_groups = df_valid.groupby('doc_prefix')

    doc_info = []
    for doc_id, group in doc_groups:
        cat_set = set()
        for cats in group['parsed_labels']:
            cat_set.update(cats)
        doc_info.append({'doc_id': doc_id, 'num_clauses': len(group), 'categories': cat_set, 'num_categories': len(cat_set)})

    print(f'Total distinct documents represented: {len(doc_info)}')
    for d in sorted(doc_info, key=lambda x: x['num_clauses'], reverse=True):
        print('  Document %s: %d clauses, %d categories' % (d['doc_id'], d['num_clauses'], d['num_categories']))

    random.seed(seed)
    sorted_docs = sorted(doc_info, key=lambda x: (x['num_categories'], x['num_clauses'] if random.random() > 0.5 else 0), reverse=True)

    train_docs, val_docs, test_docs = set(), set(), set()
    train_cats, val_cats, test_cats = set(), set(), set()
    train_clauses, val_clauses, test_clauses = 0, 0, 0
    total_clauses = len(df_valid)
    target_test = int(total_clauses * test_ratio)
    target_val = int(total_clauses * val_ratio)

    for d in sorted_docs:
        doc_id = d['doc_id']
        d_cats = d['categories']
        d_clauses = d['num_clauses']
        new_test_cats = d_cats - test_cats
        new_val_cats = d_cats - val_cats
        if len(new_test_cats) > 0 and (test_clauses + d_clauses <= target_test * 1.5 or len(test_docs) == 0):
            test_docs.add(doc_id)
            test_cats.update(d_cats)
            test_clauses += d_clauses
        elif len(new_val_cats) > 0 and (val_clauses + d_clauses <= target_val * 1.5 or len(val_docs) == 0):
            val_docs.add(doc_id)
            val_cats.update(d_cats)
            val_clauses += d_clauses
        else:
            train_docs.add(doc_id)
            train_cats.update(d_cats)
            train_clauses += d_clauses

    all_doc_ids = [d['doc_id'] for d in doc_info]
    if len(all_doc_ids) >= 3:
        if len(test_docs) == 0:
            d_move = list(train_docs)[-1]
            train_docs.remove(d_move)
            test_docs.add(d_move)
        if len(val_docs) == 0:
            d_move = list(train_docs)[-1]
            train_docs.remove(d_move)
            val_docs.add(d_move)

    overlap_tv = train_docs.intersection(val_docs)
    overlap_tt = train_docs.intersection(test_docs)
    overlap_vt = val_docs.intersection(test_docs)
    leakage = len(overlap_tv) + len(overlap_tt) + len(overlap_vt)

    print('=======================================================')
    print('  BENCHMARK SPLIT ANALYSIS')
    print('=======================================================')
    print('Document Leakage Count:', leakage, '[PASS: ZERO LEAKAGE]' if leakage == 0 else '[FAIL: LEAKAGE DETECTED]')

    train_df = df_valid[df_valid['doc_prefix'].isin(train_docs)].copy()
    val_df = df_valid[df_valid['doc_prefix'].isin(val_docs)].copy()
    test_df = df_valid[df_valid['doc_prefix'].isin(test_docs)].copy()

    print('Train Split     : %d documents, %d clauses, %d/14 categories' % (len(train_docs), len(train_df), len(train_cats)))
    print('Validation Split: %d documents, %d clauses, %d/14 categories' % (len(val_docs), len(val_df), len(val_cats)))
    print('Test Split      : %d documents, %d clauses, %d/14 categories' % (len(test_docs), len(test_df), len(test_cats)))

    def get_cat_support(df_sub):
        sup = {c: 0 for c in APPROVED_CATEGORIES}
        for cats in df_sub['parsed_labels']:
            for c in cats:
                sup[c] += 1
        return sup

    train_sup = get_cat_support(train_df)
    val_sup = get_cat_support(val_df)
    test_sup = get_cat_support(test_df)

    print('Category Support Per Split:')
    print('%-42s  %-7s  %-7s  %-7s  %s' % ('Category Name', 'Train', 'Val', 'Test', 'Status'))
    print('-' * 80)
    missing_test_cats = []
    for c in APPROVED_CATEGORIES:
        tr_c = train_sup[c]
        va_c = val_sup[c]
        te_c = test_sup[c]
        st = 'OK' if te_c > 0 else '[WARNING: ABSENT FROM TEST]'
        if te_c == 0:
            missing_test_cats.append(c)
        print('%-42s  %-7d  %-7d  %-7d  %s' % (c, tr_c, va_c, te_c, st))

    if missing_test_cats:
        print(f'[WARNING]: {len(missing_test_cats)} categories are ABSENT from Test split:')
        for mc in missing_test_cats:
            print('  -', mc)

    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        drop_cols = ['parsed_labels', 'doc_prefix']
        df_valid_clean = df_valid.drop(columns=drop_cols)
        train_clean = train_df.drop(columns=drop_cols)
        val_clean = val_df.drop(columns=drop_cols)
        test_clean = test_df.drop(columns=drop_cols)
        df_valid_clean.to_csv(output_dir / 'classifier_dataset_all_valid.csv', index=False)
        train_clean.to_csv(output_dir / 'classifier_dataset_train.csv', index=False)
        val_clean.to_csv(output_dir / 'classifier_dataset_val.csv', index=False)
        test_clean.to_csv(output_dir / 'classifier_dataset_test.csv', index=False)
        print(f'[SUCCESS]: Classification benchmark datasets written to {output_dir}')
    else:
        print('[DRY-RUN]: Files were NOT modified. Pass --write to export split files.')

    return {
        'train_docs': len(train_docs),
        'val_docs': len(val_docs),
        'test_docs': len(test_docs),
        'train_clauses': len(train_df),
        'val_clauses': len(val_df),
        'test_clauses': len(test_df),
        'test_categories': len(test_cats),
        'missing_test_categories': missing_test_cats,
        'leakage': leakage
    }


def main():
    parser = argparse.ArgumentParser(description='Rebuild document-disjoint benchmark dataset.')
    parser.add_argument('--input-csv', type=Path, default=ROOT / 'data' / 'annotations' / 'llm_pseudo_labels.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data' / 'classification')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--write', action='store_true', help='Overwrites classification CSVs in data/classification/')
    args = parser.parse_args()
    rebuild_benchmark(args.input_csv, args.output_dir, args.seed, write=args.write)


if __name__ == '__main__':
    main()
