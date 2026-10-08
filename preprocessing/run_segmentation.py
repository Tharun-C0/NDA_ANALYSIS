# =============================================================================
# NDA Research Project - Module 2 Main Pipeline
# =============================================================================
# Executes the complete NDA text extraction and clause segmentation pipeline:
#
#   1. Sample NDA documents from external_data/kleister-nda/documents/
#   2. Extract text from PDFs
#   3. Check text quality
#   4. Segment text into candidate clauses
#   5. Save segmented clauses as structured JSON
#   6. Generate human review queue CSV
#   7. Compute and save segmentation statistics
#   8. Generate clause count visualization
#
# Usage:
#   python preprocessing/run_segmentation.py
#   python preprocessing/run_segmentation.py --config path/to/config.yaml
#
# IMPORTANT:
#   The automatically segmented clauses are preliminary outputs and are
#   NOT considered ground-truth annotations.
# =============================================================================

import sys
import csv
import json
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from configs.config_loader import load_config
from preprocessing.document_sampler import sample_documents, save_sample_manifest
from preprocessing.pdf_extractor import extract_batch, ExtractionResult
from preprocessing.clause_segmenter import (
    create_segmenter,
    save_segmented_document,
    SegmentedDocument,
)


def setup_logging(config: dict) -> None:
    """Configure logging from config."""
    log_level = config.get("logging", {}).get("level", "INFO")
    log_to_file = config.get("logging", {}).get("log_to_file", False)
    log_dir = config.get("logging", {}).get("log_dir", "reports/logs")

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(
            log_path / "run_segmentation.log", mode="w", encoding="utf-8"
        ))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def save_extraction_report(
    results: list,
    output_path: str,
) -> None:
    """Save text extraction quality report as CSV (Step 3)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "document_id", "source_pdf", "character_count", "word_count",
            "line_count", "page_count", "pages_with_text", "status",
            "error_message",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "document_id": r.document_id,
                "source_pdf": r.source_file,
                "character_count": r.character_count,
                "word_count": r.word_count,
                "line_count": r.line_count,
                "page_count": r.page_count,
                "pages_with_text": r.pages_with_text,
                "status": r.status,
                "error_message": r.error_message,
            })

    logging.getLogger(__name__).info(f"Extraction report saved to {output_path}")


def save_review_queue(
    segmented_docs: list,
    output_path: str,
) -> None:
    """Create human-review-friendly CSV (Step 6)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "document_id", "clause_id", "clause_text",
            "start_position", "end_position",
            "human_verified", "notes",
        ])
        for doc in segmented_docs:
            for clause in doc.clauses:
                writer.writerow([
                    doc.document_id,
                    clause.clause_id,
                    clause.text,
                    clause.start_position,
                    clause.end_position,
                    "false",
                    "",
                ])

    logging.getLogger(__name__).info(f"Review queue saved to {output_path}")


def compute_segmentation_statistics(
    segmented_docs: list,
    extraction_results: list,
) -> dict:
    """Compute segmentation statistics (Step 7)."""
    total_docs = len(extraction_results)
    success_docs = sum(
        1 for r in extraction_results if r.status in ("success", "partial")
    )
    failed_docs = sum(
        1 for r in extraction_results if r.status in ("failed", "empty")
    )

    all_clause_counts = [len(d.clauses) for d in segmented_docs]
    all_clause_lengths = [
        len(c.text) for d in segmented_docs for c in d.clauses
    ]

    total_clauses = sum(all_clause_counts)

    stats = {
        "documents_processed": total_docs,
        "successful_documents": success_docs,
        "failed_documents": failed_docs,
        "documents_with_clauses": len(segmented_docs),
        "total_extracted_clauses": total_clauses,
    }

    if all_clause_counts:
        stats["avg_clauses_per_document"] = round(
            total_clauses / len(all_clause_counts), 2
        )
        stats["min_clauses_per_document"] = min(all_clause_counts)
        stats["max_clauses_per_document"] = max(all_clause_counts)
    else:
        stats["avg_clauses_per_document"] = 0
        stats["min_clauses_per_document"] = 0
        stats["max_clauses_per_document"] = 0

    if all_clause_lengths:
        stats["avg_clause_length"] = round(
            sum(all_clause_lengths) / len(all_clause_lengths), 2
        )
        stats["min_clause_length"] = min(all_clause_lengths)
        stats["max_clause_length"] = max(all_clause_lengths)
    else:
        stats["avg_clause_length"] = 0
        stats["min_clause_length"] = 0
        stats["max_clause_length"] = 0

    return stats


def save_segmentation_statistics(stats: dict, output_path: str) -> None:
    """Save segmentation statistics as CSV and JSON (Step 7)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # CSV
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, value])

    # Also save as JSON for programmatic access
    json_path = out.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logging.getLogger(__name__).info(
        f"Segmentation statistics saved to {output_path}"
    )


def generate_clause_count_chart(
    segmented_docs: list,
    output_dir: str,
) -> None:
    """Generate a bar chart showing clauses per document (Step 8)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logging.getLogger(__name__).warning(
            "matplotlib not installed — skipping visualization."
        )
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    doc_ids = [d.document_id[:12] + "..." for d in segmented_docs]
    clause_counts = [len(d.clauses) for d in segmented_docs]

    fig, ax = plt.subplots(figsize=(max(8, len(doc_ids) * 0.5), 6))
    bars = ax.bar(range(len(doc_ids)), clause_counts, color="#4A90D9", edgecolor="#2C5F8A")

    ax.set_xlabel("Document", fontsize=11)
    ax.set_ylabel("Number of Clauses", fontsize=11)
    ax.set_title("Clause Count per NDA Document (Baseline Segmentation)", fontsize=13)
    ax.set_xticks(range(len(doc_ids)))
    ax.set_xticklabels(doc_ids, rotation=45, ha="right", fontsize=8)

    # Add value labels
    for bar_item, count in zip(bars, clause_counts):
        ax.text(
            bar_item.get_x() + bar_item.get_width() / 2,
            bar_item.get_height() + 0.3,
            str(count), ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylim(0, max(clause_counts) * 1.15 if clause_counts else 10)
    plt.tight_layout()
    plt.savefig(out_path / "clauses_per_document.png", dpi=150)
    plt.close()

    logging.getLogger(__name__).info(
        f"Clause count chart saved to {out_path / 'clauses_per_document.png'}"
    )


def print_summary(
    stats: dict,
    seg_dir: str,
    review_path: str,
    stats_path: str,
) -> None:
    """Print final Module 2 summary."""
    print("\n" + "=" * 60)
    print("  MODULE 2 COMPLETE")
    print("=" * 60)
    print(f"\n  Documents processed:  {stats['documents_processed']}")
    print(f"  Clauses extracted:    {stats['total_extracted_clauses']}")
    print(f"  Failed documents:     {stats['failed_documents']}")
    print(f"  Output directory:     {seg_dir}")
    print(f"  Review queue:         {review_path}")
    print(f"  Statistics report:    {stats_path}")
    print()
    print("  NOTE: The automatically segmented clauses are preliminary")
    print("  outputs and are NOT considered ground-truth annotations.")
    print("=" * 60)
    print()


# =============================================================================
# Main
# =============================================================================

def main(config_path: str = None) -> None:
    """Run the complete Module 2 pipeline."""
    # ----- Load config -----
    config = load_config(config_path)
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Module 2: NDA Text Extraction & Clause Segmentation")
    logger.info("=" * 60)

    # Resolve paths
    root = Path(config["_project_root"])
    documents_dir = config["dataset"]["documents_dir"]
    text_output_dir = config["extraction"]["output_dir"]
    seg_output_dir = config["segmentation"]["output_dir"]
    reports_dir = config["output"]["reports_dir"]
    figures_dir = config["output"]["figures_dir"]

    sample_size = config["sampling"]["sample_size"]
    seed = config["sampling"]["random_seed"]
    min_chars = config["extraction"]["min_characters"]
    output_encoding = config["extraction"]["output_encoding"]

    seg_strategy = config["segmentation"]["strategy"]
    min_clause_len = config["segmentation"]["min_clause_length"]
    max_clause_len = config["segmentation"]["max_clause_length"]

    # ----- Step 1: Sample documents -----
    logger.info("Step 1: Sampling NDA documents...")
    sampled = sample_documents(documents_dir, sample_size, seed)

    manifest_path = str(Path(seg_output_dir) / "sample_manifest.json")
    save_sample_manifest(sampled, manifest_path, sample_size, seed)
    logger.info(f"  Selected {len(sampled)} documents.")

    # ----- Step 2: Extract text from PDFs -----
    logger.info("Step 2: Extracting text from PDFs...")
    extraction_results = extract_batch(
        sampled, text_output_dir, min_chars, output_encoding
    )

    # ----- Step 3: Text quality report -----
    logger.info("Step 3: Saving text extraction report...")
    report_path = str(Path(reports_dir) / "text_extraction_report.csv")
    save_extraction_report(extraction_results, report_path)

    # Print extraction summary
    success = sum(1 for r in extraction_results if r.status == "success")
    partial = sum(1 for r in extraction_results if r.status == "partial")
    failed = sum(1 for r in extraction_results if r.status in ("failed", "empty"))
    logger.info(
        f"  Extraction: {success} success, {partial} partial, {failed} failed"
    )

    # ----- Step 4: Clause segmentation -----
    logger.info(f"Step 4: Segmenting clauses (strategy: {seg_strategy})...")
    segmenter = create_segmenter(
        strategy=seg_strategy,
        min_clause_length=min_clause_len,
        max_clause_length=max_clause_len,
    )

    segmented_docs = []
    for result in extraction_results:
        if result.status in ("failed", "empty"):
            logger.warning(
                f"  Skipping {result.document_id}: {result.status}"
            )
            continue

        seg_doc = segmenter.segment(result.text, result.document_id)
        segmented_docs.append(seg_doc)

    # ----- Step 5: Save segmented documents -----
    logger.info("Step 5: Saving segmented documents...")
    for doc in segmented_docs:
        save_segmented_document(doc, seg_output_dir)

    # ----- Step 6: Human review queue -----
    logger.info("Step 6: Creating human review queue...")
    review_path = str(Path(seg_output_dir) / "review_queue.csv")
    save_review_queue(segmented_docs, review_path)

    # ----- Step 7: Segmentation statistics -----
    logger.info("Step 7: Computing segmentation statistics...")
    stats = compute_segmentation_statistics(segmented_docs, extraction_results)
    stats_path = str(Path(reports_dir) / "segmentation_statistics.csv")
    save_segmentation_statistics(stats, stats_path)

    # Print statistics
    print("\n" + "=" * 60)
    print("  SEGMENTATION STATISTICS")
    print("=" * 60)
    for key, value in stats.items():
        label = key.replace("_", " ").title()
        print(f"  {label + ':':<35} {value}")
    print("=" * 60)

    # ----- Step 8: Visualization -----
    logger.info("Step 8: Generating visualization...")
    if segmented_docs:
        generate_clause_count_chart(segmented_docs, figures_dir)
    else:
        logger.warning("  No segmented documents — skipping visualization.")

    # ----- Final summary -----
    print_summary(stats, seg_output_dir, review_path, stats_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module 2: NDA Text Extraction & Clause Segmentation"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: configs/config.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
