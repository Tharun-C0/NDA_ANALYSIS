import os
import json

sample_doc_ids = [
    'b82a10c42fc284dba9870ac7c75cd386', # 1. Very few clauses (5 clauses)
    '9a5cb31024ad0a7a4916e4f122ebea4a', # 2. Many clauses (72 clauses)
    '5180f107324abcab9d9ff81aee2db8d3', # 3. Suspicious short clauses (isolated single digits "1", "2", "3")
    '0859334b76224ff82c1312ae7b2b5da1', # 4. Normal document 1 (15 clauses)
    '247166e0245431dcf97ee884f1f07e35'  # 5. Normal document 2 (10 clauses, 0 suspicious)
]

output_dir = r'f:\NDA\reports\manual_review_samples'
os.makedirs(output_dir, exist_ok=True)

seg_dir = r'f:\NDA\data\segmentation'

for doc_id in sample_doc_ids:
    json_path = os.path.join(seg_dir, f"{doc_id}.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Write readable JSON
    out_json = os.path.join(output_dir, f"{doc_id}_sample.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        
    # Write readable TXT
    out_txt = os.path.join(output_dir, f"{doc_id}_sample.txt")
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(f"DOCUMENT ID: {data['document_id']}\n")
        f.write(f"SOURCE FILE: {data['source_file']}\n")
        f.write(f"TOTAL CLAUSES: {data['num_clauses']}\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, clause in enumerate(data['clauses']):
            f.write(f"--- CLAUSE {idx+1}/{data['num_clauses']} ({clause['clause_id']}) [Length: {len(clause['text'])}] ---\n")
            f.write(clause['text'] + "\n\n")

print(f"Exported 5 sample documents to {output_dir}")
