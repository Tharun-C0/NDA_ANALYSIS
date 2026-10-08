import os
import json
import glob
import pandas as pd
import numpy as np
from clause_segmenter_v3 import ImprovedSegmenterV3, SegmentedDocumentV3

def run_v3_pipeline():
    seg_dir_v1 = r"f:\NDA\data\segmentation"
    seg_dir_v2 = r"f:\NDA\data\segmentation_v2"
    seg_dir_v3 = r"f:\NDA\data\segmentation_v3"
    text_dir = r"f:\NDA\data\processed\text"
    reports_dir = r"f:\NDA\reports"
    samples_dir = os.path.join(reports_dir, "manual_review_samples_v3")

    os.makedirs(seg_dir_v3, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    segmenter = ImprovedSegmenterV3(min_clause_length=20)

    # 1. Get same 20 documents from v2
    v2_json_files = sorted(glob.glob(os.path.join(seg_dir_v2, "*.json")))

    v1_docs = {}
    v2_docs = {}
    v3_docs = {}

    v2_total_clauses = 0
    v3_total_clauses = 0
    v2_susp_clauses = 0
    v3_susp_clauses = 0
    v2_short_clauses = 0
    v3_short_clauses = 0
    
    large_header_segments = 0
    largest_legal_clause_length = 0
    largest_header_length = 0

    comparison_rows = []
    audit_v3_rows = []
    susp_v3_rows = []

    for v2_path in v2_json_files:
        with open(v2_path, "r", encoding="utf-8") as f:
            v2_data = json.load(f)
        doc_id = v2_data["document_id"]
        v2_docs[doc_id] = v2_data

        # Load V1
        v1_path = os.path.join(seg_dir_v1, f"{doc_id}.json")
        if os.path.exists(v1_path):
            with open(v1_path, "r", encoding="utf-8") as f:
                v1_docs[doc_id] = json.load(f)
        else:
            v1_docs[doc_id] = {"clauses": []}

        # V2 Stats
        v2_c = v2_data.get("clauses", [])
        v2_total_clauses += len(v2_c)
        v2_sc = sum(1 for c in v2_c if c.get("is_suspicious", False))
        v2_short = sum(1 for c in v2_c if len(c.get("text", "").strip()) < 20)
        v2_susp_clauses += v2_sc
        v2_short_clauses += v2_short
        
        # Read full text
        text_path = os.path.join(text_dir, f"{doc_id}.txt")
        if not os.path.exists(text_path):
            continue
        with open(text_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        # V2 Boundary approximation for metrics (could use len(v2_c) since each is a boundary)
        v2_boundary = len(v2_c)

        # 3. Segment with V3
        v3_doc = segmenter.segment(full_text, doc_id)
        v3_docs[doc_id] = v3_doc

        # Save V3 JSON
        v3_json_path = os.path.join(seg_dir_v3, f"{doc_id}.json")
        with open(v3_json_path, "w", encoding="utf-8") as f:
            json.dump(v3_doc.to_dict(), f, indent=2, ensure_ascii=False)

        # V3 Stats
        v3_c = v3_doc.clauses
        v3_total_clauses += len(v3_c)
        v3_sc = sum(1 for c in v3_c if c.is_suspicious)
        v3_short = sum(1 for c in v3_c if len(c.text.strip()) < 20)
        v3_susp_clauses += v3_sc
        v3_short_clauses += v3_short
        v3_boundary = len(v3_c)

        v3_lens = [len(c.text.strip()) for c in v3_c]
        v3_min_len = min(v3_lens) if v3_lens else 0

        # Quality metrics tracking
        for c in v3_c:
            l = len(c.text)
            if c.segment_type == "header":
                if l > 500:
                    large_header_segments += 1
                if l > largest_header_length:
                    largest_header_length = l
            if c.segment_type == "legal_clause":
                if l > largest_legal_clause_length:
                    largest_legal_clause_length = l

            if c.is_suspicious:
                susp_v3_rows.append({
                    "document_id": doc_id,
                    "clause_id": c.clause_id,
                    "clause_text": c.text.replace("\n", " "),
                    "length": l,
                    "segment_type": c.segment_type
                })

        comparison_rows.append({
            "document_id": doc_id,
            "v2_clause_count": len(v2_c),
            "v3_clause_count": len(v3_c),
            "v2_suspicious_count": v2_sc,
            "v3_suspicious_count": v3_sc,
            "v2_short_clause_count": v2_short,
            "v3_short_clause_count": v3_short,
            "v2_boundary_count": v2_boundary,
            "v3_boundary_count": v3_boundary,
            "status": v3_doc.quality_status
        })

        audit_v3_rows.append({
            "document_id": doc_id,
            "number_of_clauses": len(v3_c),
            "total_text_length": len(full_text),
            "average_clause_length": np.mean(v3_lens) if v3_lens else 0,
            "median_clause_length": np.median(v3_lens) if v3_lens else 0,
            "minimum_clause_length": v3_min_len,
            "maximum_clause_length": max(v3_lens) if v3_lens else 0,
            "number_of_short_clauses": v3_short,
            "number_of_suspicious_clauses": v3_sc,
            "number_of_detected_boundaries": v3_boundary,
            "quality_warning": v3_doc.quality_status
        })

    # Save reports
    pd.DataFrame(comparison_rows).to_csv(os.path.join(reports_dir, "segmentation_v2_vs_v3.csv"), index=False)
    pd.DataFrame(audit_v3_rows).to_csv(os.path.join(reports_dir, "segmentation_v3_quality_audit.csv"), index=False)
    pd.DataFrame(susp_v3_rows).to_csv(os.path.join(reports_dir, "segmentation_v3_suspicious_clauses.csv"), index=False)

    # 11. Manual Review Samples
    specific_docs = [
        "5180f107324abcab9d9ff81aee2db8d3",
        "f28c4f3d35a152dd415f9b255122cb38",
        "586c367e2c45ebd8b7ba96fcb6006bf6",
        "9a5cb31024ad0a7a4916e4f122ebea4a",
        "d714d261edc4d361e7d2ebabccaada50",
        "0859334b76224ff82c1312ae7b2b5da1" # a normal doc
    ]
    
    for doc_id in specific_docs:
        if doc_id not in v3_docs:
            continue
            
        v1_data = v1_docs[doc_id]
        v2_data = v2_docs[doc_id]
        v3_data = v3_docs[doc_id].to_dict()

        with open(os.path.join(text_dir, f"{doc_id}.txt"), "r", encoding="utf-8") as f:
            raw_text = f.read()

        out_data = {
            "document_id": doc_id,
            "raw_text": raw_text,
            "v1_segmentation": v1_data,
            "v2_segmentation": v2_data,
            "v3_segmentation": v3_data
        }

        with open(os.path.join(samples_dir, f"{doc_id}_comparison.json"), "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)

        with open(os.path.join(samples_dir, f"{doc_id}_comparison.txt"), "w", encoding="utf-8") as f:
            f.write(f"DOCUMENT ID: {doc_id}\n")
            f.write("=" * 80 + "\n\n")
            f.write("--- V1 SEGMENTATION ---\n")
            for c in v1_data.get("clauses", []):
                f.write(f"[{c.get('clause_id')}] ({len(c.get('text',''))} chars)\n{c.get('text','')}\n\n")
            f.write("=" * 80 + "\n\n")
            f.write("--- V2 SEGMENTATION ---\n")
            for c in v2_data.get("clauses", []):
                susp_flag = "[SUSPICIOUS]" if c.get('is_suspicious') else ""
                stype = c.get('segment_type', 'unknown')
                f.write(f"[{c.get('clause_id')}] (Type: {stype}) ({len(c.get('text',''))} chars) {susp_flag}\n{c.get('text','')}\n\n")
            f.write("=" * 80 + "\n\n")
            f.write("--- V3 SEGMENTATION ---\n")
            for c in v3_data.get("clauses", []):
                susp_flag = "[SUSPICIOUS]" if c.get('is_suspicious') else ""
                stype = c.get('segment_type', 'unknown')
                f.write(f"[{c.get('clause_id')}] (Type: {stype}) ({len(c.get('text',''))} chars) {susp_flag}\n{c.get('text','')}\n\n")

    # Print Final Report
    print("="*60)
    print("MODULE 2.2 - V3 PIPELINE REPORT")
    print("="*60)
    print(f"Documents processed: {len(v2_json_files)}")
    print(f"V2 total clauses: {v2_total_clauses}")
    print(f"V3 total clauses: {v3_total_clauses}")
    print(f"V2 suspicious clauses: {v2_susp_clauses}")
    print(f"V3 suspicious clauses: {v3_susp_clauses}")
    print(f"V2 short clauses: {v2_short_clauses}")
    print(f"V3 short clauses: {v3_short_clauses}")
    print()
    print("Quality Checks:")
    print(f"  large_header_segments (>500 chars): {large_header_segments}")
    print(f"  largest_legal_clause_length:        {largest_legal_clause_length}")
    print(f"  largest_header_length:              {largest_header_length}")
    print("="*60)

if __name__ == "__main__":
    run_v3_pipeline()
