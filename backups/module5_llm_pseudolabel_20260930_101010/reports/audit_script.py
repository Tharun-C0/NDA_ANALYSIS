import os
import json
import glob
import re
import pandas as pd
import numpy as np

def run_audit():
    seg_dir = r'f:\NDA\data\segmentation'
    json_files = sorted(glob.glob(os.path.join(seg_dir, '*.json')))
    json_files = [f for f in json_files if not f.endswith('sample_manifest.json')]

    audit_rows = []
    suspicious_rows = []

    punct_set = set(".,;:!?-–—\"'()[]{}/*\\&$#@ ")

    for fpath in json_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        doc_id = data['document_id']
        clauses = data.get('clauses', [])
        total_clauses = len(clauses)
        lengths = [len(c['text']) for c in clauses]
        
        susp_count_doc = 0
        
        # Exact boundary count in raw text using regex matching section/clause markers & headers
        full_text = '\n\n'.join([c['text'] for c in clauses])
        boundaries = len(re.findall(r'(?m)^(?:(?:ARTICLE|SECTION)\s+[IVXLCDM0-9]+|\d+\.|\([a-z0-9]+\)|[A-Z\s]{4,}:|EX-\d+|EXHIBIT|WHEREAS|NOW\s+THEREFORE|IN\s+WITNESS\s+WHEREOF)', full_text))
        
        for idx, c in enumerate(clauses):
            c_id = c.get('clause_id', f"{doc_id}_clause_{idx:03d}")
            ctext = c['text']
            t = ctext.strip()
            reasons = []
            
            if len(t) == 0:
                reasons.append("empty")
            elif len(ctext) <= 20:
                reasons.append("short_<=20_chars")
                
            if t and all(ch in punct_set for ch in t):
                reasons.append("only_punctuation")
                
            if t and t.isdigit():
                reasons.append("only_numbers")
                
            if t and re.match(r'^(?:Section|Article|Clause|\d+[\.\d]*)\s*\d*$', t, re.IGNORECASE):
                reasons.append("only_section_numbers")
                
            if t and re.match(r'^(?:Page\s*\d+|\d+\s*of\s*\d+|-\s*\d+\s*-|\d+)$', t, re.IGNORECASE):
                reasons.append("page_numbers")
                
            if t and (t.startswith("EX-") or "exhibit" in t.lower() or (t.isupper() and len(t) < 40 and not t.endswith('.'))):
                reasons.append("header_or_title")
                
            if reasons:
                susp_count_doc += 1
                suspicious_rows.append({
                    'document_id': doc_id,
                    'clause_id': c_id,
                    'clause_text': ctext.replace('\n', ' '),
                    'length': len(ctext),
                    'reason': '; '.join(reasons)
                })
                
        min_len = int(min(lengths)) if lengths else 0
        max_len = int(max(lengths)) if lengths else 0
        med_len = float(np.median(lengths)) if lengths else 0.0
        
        notes = []
        if total_clauses <= 6:
            notes.append("very_few_clauses")
        elif total_clauses >= 40:
            notes.append("many_clauses")
        if susp_count_doc > 0:
            notes.append(f"{susp_count_doc}_suspicious_clauses")
        if abs(total_clauses - boundaries) > 10:
            notes.append(f"boundary_mismatch(clauses={total_clauses},boundaries={boundaries})")
            
        audit_rows.append({
            'document_id': doc_id,
            'total_clauses': total_clauses,
            'suspicious_clauses': susp_count_doc,
            'min_length': min_len,
            'median_length': med_len,
            'max_length': max_len,
            'boundary_count': boundaries,
            'quality_notes': '; '.join(notes) if notes else 'normal'
        })

    audit_df = pd.DataFrame(audit_rows)
    susp_df = pd.DataFrame(suspicious_rows)

    os.makedirs(r'f:\NDA\reports', exist_ok=True)
    audit_df.to_csv(r'f:\NDA\reports\segmentation_quality_audit.csv', index=False)
    susp_df.to_csv(r'f:\NDA\reports\suspicious_clauses.csv', index=False)

    print(f"Generated segmentation_quality_audit.csv ({len(audit_df)} docs) and suspicious_clauses.csv ({len(susp_df)} suspicious clauses).")

if __name__ == '__main__':
    run_audit()
