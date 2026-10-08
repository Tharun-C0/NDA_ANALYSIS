import json
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


def check_readiness(input_csv: Path, cls_dir: Path):
    print('=======================================================')
    print('  MODULE 8 - FINAL BENCHMARK READINESS AUDIT')
    print('=======================================================')
    print('Input file:', input_csv)
    print('Classification Dir:', cls_dir)
    print()

    results = {}

    if not input_csv.exists():
        print('CRITICAL ERROR: Input pseudo-label CSV not found.')
        return {'OVERALL': 'FAIL'}

    df = pd.read_csv(input_csv)
    valid_df = df[df['pseudo_label_quality'].isin(VALID_QUALITY_BUCKETS)].copy()
    valid_df['doc_prefix'] = valid_df['document_id'].astype(str)

    # 1. Document Diversity
    num_docs = valid_df['doc_prefix'].nunique()
    if num_docs >= 15:
        results['1. Document Diversity'] = ('PASS', f'{num_docs} documents represented (Target >= 15)')
    elif num_docs >= 10:
        results['1. Document Diversity'] = ('WARN', f'{num_docs} documents represented (Sub-optimal, Target >= 15)')
    else:
        results['1. Document Diversity'] = ('FAIL', f'{num_docs} documents represented (INSUFFICIENT, Target >= 15)')

    # 2. Document Leakage
    train_csv = cls_dir / 'classifier_dataset_train.csv'
    val_csv = cls_dir / 'classifier_dataset_val.csv'
    test_csv = cls_dir / 'classifier_dataset_test.csv'

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        tr_df = pd.read_csv(train_csv)
        va_df = pd.read_csv(val_csv)
        te_df = pd.read_csv(test_csv)
        tr_docs = set(tr_df['document_id'].astype(str))
        va_docs = set(va_df['document_id'].astype(str))
        te_docs = set(te_df['document_id'].astype(str))
        ov = len(tr_docs.intersection(va_docs)) + len(tr_docs.intersection(te_docs)) + len(va_docs.intersection(te_docs))
        if ov == 0:
            results['2. Document Leakage'] = ('PASS', 'Zero document overlap across train/val/test splits')
        else:
            results['2. Document Leakage'] = ('FAIL', f'{ov} document overlaps detected across splits!')
    else:
        results['2. Document Leakage'] = ('WARN', 'Classification split CSVs not exported yet.')

    # 3. Category Coverage Dataset-Wide
    all_cats = set()
    for raw in valid_df['final_pseudo_labels']:
        try:
            val = json.loads(raw) if isinstance(raw, str) else []
            all_cats.update([c for c in val if c in APPROVED_CATEGORIES])
        except Exception:
            pass
    if len(all_cats) == 14:
        results['3. Category Coverage'] = ('PASS', 'All 14 categories represented dataset-wide')
    else:
        results['3. Category Coverage'] = ('FAIL', f'Only {len(all_cats)}/14 categories represented dataset-wide')

    # 4. Duplicate Records
    dup_count = valid_df.duplicated(subset=['clause_id']).sum()
    if dup_count == 0:
        results['4. Duplicate Records'] = ('PASS', 'Zero duplicate clause IDs in valid dataset')
    else:
        results['4. Duplicate Records'] = ('FAIL', f'{dup_count} duplicate clause IDs found!')

    # 5. Invalid Labels
    invalid_count = 0
    for raw in valid_df['final_pseudo_labels']:
        try:
            val = json.loads(raw)
            if not isinstance(val, list):
                invalid_count += 1
        except Exception:
            invalid_count += 1
    if invalid_count == 0:
        results['5. Invalid Labels'] = ('PASS', 'Zero unparseable or invalid labels')
    else:
        results['5. Invalid Labels'] = ('FAIL', f'{invalid_count} invalid label records found')

    # 6. API Error Contamination
    err_in_valid = valid_df['pseudo_label_quality'].isin(['API_ERROR', 'API_QUOTA_EXHAUSTED']).sum()
    if err_in_valid == 0:
        results['6. API Error Contamination'] = ('PASS', 'Zero API_ERROR/QUOTA records in valid evaluation set')
    else:
        results['6. API Error Contamination'] = ('FAIL', f'{err_in_valid} API error records in valid set!')

    # 7. Missing Text
    missing_text = valid_df['clause_text'].isna().sum()
    if missing_text == 0:
        results['7. Missing Text'] = ('PASS', 'Zero null or empty clause texts')
    else:
        results['7. Missing Text'] = ('FAIL', f'{missing_text} clauses missing text content!')

    # 8. Missing Document IDs
    missing_doc = valid_df['document_id'].isna().sum()
    if missing_doc == 0:
        results['8. Missing Document IDs'] = ('PASS', 'Zero null document IDs')
    else:
        results['8. Missing Document IDs'] = ('FAIL', f'{missing_doc} clauses missing document ID!')

    # 9. Metadata Preservation
    meta_cols = ['label_source', 'average_confidence', 'agreement_score', 'pseudo_label_quality']
    has_meta = all(c in valid_df.columns for c in meta_cols)
    if has_meta:
        results['9. Metadata Preservation'] = ('PASS', 'All metadata columns present (label_source, confidence, agreement)')
    else:
        results['9. Metadata Preservation'] = ('FAIL', 'Missing metadata columns in dataset')

    # 10. Test-Set Category Coverage
    if test_csv.exists():
        te_df = pd.read_csv(test_csv)
        te_cats = set()
        for raw in te_df['final_pseudo_labels']:
            try:
                val = json.loads(raw) if isinstance(raw, str) else []
                te_cats.update([c for c in val if c in APPROVED_CATEGORIES])
            except Exception:
                pass
        if len(te_cats) == 14:
            results['10. Test-Set Category Coverage'] = ('PASS', 'All 14 categories present in Test split')
        else:
            results['10. Test-Set Category Coverage'] = ('WARN', f'{len(te_cats)}/14 categories present in Test split (Missing {14 - len(te_cats)})')
    else:
        results['10. Test-Set Category Coverage'] = ('WARN', 'Test CSV not generated yet')

    # 11. Human Verification Availability
    hv_csv = ROOT / 'data' / 'annotations' / 'human_verification_sample.csv'
    if hv_csv.exists() and hv_csv.stat().st_size > 1000:
        results['11. Human Verification Availability'] = ('PASS', f'Human verification sample available ({hv_csv.name})')
    else:
        results['11. Human Verification Availability'] = ('WARN', 'Human verification sample file missing or unpopulated')

    # 12. Pseudo-Label Provenance
    prov_count = (valid_df['label_source'] == 'llm_pseudo_label').sum()
    if prov_count == len(valid_df):
        results['12. Pseudo-Label Provenance'] = ('PASS', '100% of records have explicit llm_pseudo_label provenance')
    else:
        results['12. Pseudo-Label Provenance'] = ('WARN', f'Provenance marked on {prov_count}/{len(valid_df)} records')

    # Overall summary
    print('%-38s | %-6s | %s' % ('Readiness Metric', 'Status', 'Details'))
    print('-' * 80)
    has_fail = False
    has_warn = False
    for metric, (status, detail) in results.items():
        if status == 'FAIL':
            has_fail = True
        if status == 'WARN':
            has_warn = True
        print('%-38s | %-6s | %s' % (metric, status, detail))

    print('=' * 80)
    if has_fail:
        overall = 'BLOCKED'
    elif has_warn:
        overall = 'WARN'
    else:
        overall = 'PASS'
    print('OVERALL BENCHMARK READINESS STATUS:', overall)
    return results, overall


def main():
    input_csv = ROOT / 'data' / 'annotations' / 'llm_pseudo_labels.csv'
    cls_dir = ROOT / 'data' / 'classification'
    check_readiness(input_csv, cls_dir)


if __name__ == '__main__':
    main()
