# =============================================================================
# NDA Research Project - Dataset Statistics
# =============================================================================
# Module 1: Statistical Analysis of the Parsed Dataset
#
# Generates comprehensive statistics about the dataset including:
# - Document-level counts
# - Clause-level counts and length distributions
# - Label frequency analysis
# - Single-label vs. multi-label distribution
# - Class imbalance metrics
#
# All statistics are printed to console and optionally saved to files
# for inclusion in research papers.
# =============================================================================

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

from preprocessing.dataset_loader import ClauseRecord

logger = logging.getLogger(__name__)


def compute_statistics(clauses: List[ClauseRecord]) -> Dict[str, Any]:
    """
    Compute comprehensive dataset statistics.
    
    Args:
        clauses: List of ClauseRecord objects (preprocessed).
    
    Returns:
        Dictionary containing all computed statistics.
    """
    if not clauses:
        return {"error": "No clauses to analyze."}
    
    # --- Document-level statistics ---
    doc_ids = set(c.doc_id for c in clauses)
    clauses_per_doc = Counter(c.doc_id for c in clauses)
    
    # --- Clause-level statistics ---
    clause_lengths = [len(c.text) for c in clauses]
    clause_word_counts = [len(c.text.split()) for c in clauses]
    
    # --- Label statistics ---
    all_labels = []
    label_counter = Counter()
    num_labels_per_clause = []
    
    for c in clauses:
        all_labels.extend(c.labels)
        label_counter.update(c.labels)
        num_labels_per_clause.append(len(c.labels))
    
    unique_labels = sorted(label_counter.keys())
    single_label_count = sum(1 for n in num_labels_per_clause if n == 1)
    multi_label_count = sum(1 for n in num_labels_per_clause if n > 1)
    
    # --- Class imbalance analysis ---
    total_label_assignments = sum(label_counter.values())
    label_frequencies = {
        label: {
            "count": count,
            "percentage": round(100.0 * count / total_label_assignments, 2)
            if total_label_assignments > 0 else 0.0,
        }
        for label, count in label_counter.most_common()
    }
    
    # Identify majority and minority classes
    # Convention: minority class has < mean frequency
    if label_counter:
        mean_count = total_label_assignments / len(label_counter)
        majority_classes = [l for l, c in label_counter.items() if c >= mean_count]
        minority_classes = [l for l, c in label_counter.items() if c < mean_count]
    else:
        majority_classes = []
        minority_classes = []
    
    stats = {
        "documents": {
            "total_documents": len(doc_ids),
            "clauses_per_doc_min": min(clauses_per_doc.values()),
            "clauses_per_doc_max": max(clauses_per_doc.values()),
            "clauses_per_doc_avg": round(
                sum(clauses_per_doc.values()) / len(clauses_per_doc), 2
            ),
        },
        "clauses": {
            "total_clauses": len(clauses),
            "single_label_clauses": single_label_count,
            "multi_label_clauses": multi_label_count,
            "char_length_min": min(clause_lengths),
            "char_length_max": max(clause_lengths),
            "char_length_avg": round(sum(clause_lengths) / len(clause_lengths), 2),
            "word_count_min": min(clause_word_counts),
            "word_count_max": max(clause_word_counts),
            "word_count_avg": round(
                sum(clause_word_counts) / len(clause_word_counts), 2
            ),
        },
        "labels": {
            "num_unique_labels": len(unique_labels),
            "unique_labels": unique_labels,
            "total_label_assignments": total_label_assignments,
            "label_frequencies": label_frequencies,
            "majority_classes": sorted(majority_classes),
            "minority_classes": sorted(minority_classes),
        },
    }
    
    return stats


def print_statistics(stats: Dict[str, Any]) -> None:
    """
    Print dataset statistics in a clear, formatted manner.
    """
    if "error" in stats:
        print(f"\n  ERROR: {stats['error']}")
        return
    
    print("\n" + "=" * 70)
    print("  NDA DATASET STATISTICS")
    print("=" * 70)
    
    # Document stats
    doc = stats["documents"]
    print(f"\n  📄 DOCUMENTS")
    print(f"  {'Total NDA documents:':<40} {doc['total_documents']}")
    print(f"  {'Clauses per document (min):':<40} {doc['clauses_per_doc_min']}")
    print(f"  {'Clauses per document (max):':<40} {doc['clauses_per_doc_max']}")
    print(f"  {'Clauses per document (avg):':<40} {doc['clauses_per_doc_avg']}")
    
    # Clause stats
    cl = stats["clauses"]
    print(f"\n  📝 CLAUSES")
    print(f"  {'Total clauses:':<40} {cl['total_clauses']}")
    print(f"  {'Single-label clauses:':<40} {cl['single_label_clauses']}")
    print(f"  {'Multi-label clauses:':<40} {cl['multi_label_clauses']}")
    print(f"  {'Clause length in chars (min):':<40} {cl['char_length_min']}")
    print(f"  {'Clause length in chars (max):':<40} {cl['char_length_max']}")
    print(f"  {'Clause length in chars (avg):':<40} {cl['char_length_avg']}")
    print(f"  {'Clause word count (min):':<40} {cl['word_count_min']}")
    print(f"  {'Clause word count (max):':<40} {cl['word_count_max']}")
    print(f"  {'Clause word count (avg):':<40} {cl['word_count_avg']}")
    
    # Label stats
    lb = stats["labels"]
    print(f"\n  🏷️  LABELS")
    print(f"  {'Number of unique labels:':<40} {lb['num_unique_labels']}")
    print(f"  {'Total label assignments:':<40} {lb['total_label_assignments']}")
    
    print(f"\n  📊 CLASS DISTRIBUTION")
    print(f"  {'Label':<35} {'Count':>8} {'Percentage':>12}")
    print(f"  {'-' * 55}")
    for label, info in lb["label_frequencies"].items():
        pct = f"{info['percentage']:.2f}%"
        print(f"  {label:<35} {info['count']:>8} {pct:>12}")
    
    print(f"\n  ⚖️  CLASS IMBALANCE")
    print(f"  {'Majority classes:':<40} {', '.join(lb['majority_classes']) or 'N/A'}")
    print(f"  {'Minority classes:':<40} {', '.join(lb['minority_classes']) or 'N/A'}")
    
    print("\n" + "=" * 70)


def save_statistics_json(stats: Dict[str, Any], output_path: str) -> None:
    """Save statistics to a JSON file for programmatic access."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Statistics saved to {output_path}")


def save_class_frequencies_csv(stats: Dict[str, Any], output_path: str) -> None:
    """
    Save class frequency analysis to a CSV file.
    
    The CSV contains columns: label, count, percentage, category
    where category is 'majority' or 'minority'.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lb = stats["labels"]
    majority_set = set(lb["majority_classes"])
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "count", "percentage", "category"])
        
        for label, info in lb["label_frequencies"].items():
            category = "majority" if label in majority_set else "minority"
            writer.writerow([
                label,
                info["count"],
                info["percentage"],
                category,
            ])
    
    logger.info(f"Class frequencies saved to {output_path}")
