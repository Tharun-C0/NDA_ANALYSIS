import os
import json
import glob
import pandas as pd

def create_human_review_files():
    seg_dir = r"f:\NDA\data\segmentation_v3"
    review_data_dir = r"f:\NDA\data\human_review"
    reports_review_dir = r"f:\NDA\reports\human_review"

    os.makedirs(review_data_dir, exist_ok=True)
    os.makedirs(reports_review_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(seg_dir, "*.json")))
    
    dataset_rows = []
    priority_rows = []
    
    total_segments = 0
    priority_segments = 0

    normal_clauses_pool = []

    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        doc_id = data["document_id"]
        clauses = data.get("clauses", [])
        
        # Determine if doc has unusual clause count
        unusual_count = len(clauses) < 5 or len(clauses) > 60

        # Create human readable TXT
        txt_path = os.path.join(reports_review_dir, f"{doc_id}_review.txt")
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write("========================================\n")
            tf.write(f"DOCUMENT: {doc_id}\n")
            tf.write("========================================\n\n")

            for c in clauses:
                c_id = c["clause_id"]
                c_text = c["text"]
                c_type = c["segment_type"]
                c_len = len(c_text)
                is_suspicious = c.get("is_suspicious", False)

                total_segments += 1
                
                row = {
                    "document_id": doc_id,
                    "clause_id": c_id,
                    "clause_text": c_text,
                    "segment_type": c_type,
                    "length": c_len,
                    "human_is_valid_clause": "",
                    "human_corrected_text": "",
                    "human_action": "",
                    "human_notes": "",
                    "reviewer": "",
                    "reviewed": False
                }
                dataset_rows.append(row)

                # Priority logic
                is_priority = False
                if is_suspicious:
                    is_priority = True
                elif c_len < 20 or c_len > 3000:
                    is_priority = True
                elif c_type in ["header", "footer", "section_heading", "fragment"]:
                    is_priority = True
                elif unusual_count:
                    is_priority = True
                
                if is_priority:
                    priority_rows.append(row)
                    priority_segments += 1
                else:
                    normal_clauses_pool.append(row)

                # Write TXT format
                tf.write(f"CLAUSE: {c_id}\n")
                tf.write(f"TYPE: {c_type}\n")
                tf.write(f"LENGTH: {c_len}\n\n")
                tf.write("TEXT:\n")
                tf.write("-" * 40 + "\n")
                tf.write(f"{c_text}\n")
                tf.write("-" * 40 + "\n\n")
                tf.write("HUMAN DECISION:\n")
                tf.write("[ ] KEEP\n")
                tf.write("[ ] MERGE_WITH_NEXT\n")
                tf.write("[ ] MERGE_WITH_PREVIOUS\n")
                tf.write("[ ] SPLIT\n")
                tf.write("[ ] REMOVE_NON_LEGAL\n")
                tf.write("[ ] REVIEW\n\n")
                tf.write("CORRECTED TEXT:\n")
                tf.write("_" * 40 + "\n\n")
                tf.write("NOTES:\n")
                tf.write("_" * 40 + "\n\n\n")

    # Add a sample of normal legal clauses to priority (e.g., up to 20 or 10%)
    import random
    random.seed(42)
    sample_size = min(len(normal_clauses_pool), 20)
    normal_sample = random.sample(normal_clauses_pool, sample_size)
    priority_rows.extend(normal_sample)
    priority_segments += sample_size

    # Save CSVs
    dataset_csv_path = os.path.join(review_data_dir, "review_dataset.csv")
    priority_csv_path = os.path.join(review_data_dir, "priority_review.csv")
    
    pd.DataFrame(dataset_rows).to_csv(dataset_csv_path, index=False)
    pd.DataFrame(priority_rows).to_csv(priority_csv_path, index=False)

    # Review Statistics
    stats_csv_path = os.path.join(reports_review_dir, "review_statistics.csv")
    stats_data = [{
        "total_segments": total_segments,
        "reviewed_segments": 0,
        "unreviewed_segments": total_segments,
        "keep_count": 0,
        "merge_count": 0,
        "split_count": 0,
        "remove_count": 0,
        "review_count": 0
    }]
    pd.DataFrame(stats_data).to_csv(stats_csv_path, index=False)

    print("==================================================")
    print("MODULE 3A - HUMAN VERIFICATION PREPARATION")
    print("==================================================")
    print(f"Number of documents: {len(json_files)}")
    print(f"Number of V3 segments: {total_segments}")
    print(f"Number of priority segments: {priority_segments}")
    print()
    print(f"Location of review_dataset.csv: {dataset_csv_path}")
    print(f"Location of priority_review.csv: {priority_csv_path}")
    print(f"Location of review instructions: {os.path.join(reports_review_dir, 'instructions.md')}")
    print(f"Location of manual review files: {reports_review_dir}")
    print(f"Location of review statistics: {stats_csv_path}")
    print("==================================================")

if __name__ == "__main__":
    create_human_review_files()
