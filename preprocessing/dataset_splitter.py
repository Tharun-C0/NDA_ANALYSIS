# =============================================================================
# NDA Research Project - Dataset Splitter
# =============================================================================
# Module 1: Train / Validation / Test Splitting
#
# CRITICAL RESEARCH DECISION:
# Splitting is performed at the DOCUMENT level, NOT the clause level.
# This prevents data leakage: clauses from the same NDA document share
# context, style, and terminology. If clauses from one document appeared
# in both train and test sets, the model could exploit document-specific
# features rather than learning generalizable clause patterns.
#
# This is consistent with standard practice in legal NLP research.
# =============================================================================

import json
import random
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

from preprocessing.dataset_loader import ClauseRecord

logger = logging.getLogger(__name__)


def split_by_documents(
    clauses: List[ClauseRecord],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[ClauseRecord], List[ClauseRecord], List[ClauseRecord]]:
    """
    Split clauses into train/val/test sets at the document level.
    
    All clauses from a single NDA document are assigned to the same split.
    This prevents data leakage between splits.
    
    Args:
        clauses: List of all ClauseRecord objects.
        train_ratio: Proportion of documents for training.
        val_ratio: Proportion of documents for validation.
        test_ratio: Proportion of documents for testing.
        random_seed: Fixed seed for reproducibility.
    
    Returns:
        Tuple of (train_clauses, val_clauses, test_clauses).
    
    Raises:
        ValueError: If ratios don't sum to ~1.0 or if there are too few documents.
    """
    # Validate ratios
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {ratio_sum:.4f} "
            f"({train_ratio} + {val_ratio} + {test_ratio})"
        )
    
    # Group clauses by document
    doc_to_clauses: Dict[str, List[ClauseRecord]] = defaultdict(list)
    for clause in clauses:
        doc_to_clauses[clause.doc_id].append(clause)
    
    doc_ids = sorted(doc_to_clauses.keys())  # Sort for determinism before shuffle
    num_docs = len(doc_ids)
    
    if num_docs < 3:
        raise ValueError(
            f"Need at least 3 documents for a 3-way split, got {num_docs}."
        )
    
    # Deterministic shuffle
    rng = random.Random(random_seed)
    rng.shuffle(doc_ids)
    
    # Calculate split boundaries
    train_end = int(round(num_docs * train_ratio))
    val_end = train_end + int(round(num_docs * val_ratio))
    
    # Ensure each split has at least one document
    if train_end == 0:
        train_end = 1
    if val_end <= train_end:
        val_end = train_end + 1
    if val_end >= num_docs:
        val_end = num_docs - 1
    
    train_docs = doc_ids[:train_end]
    val_docs = doc_ids[train_end:val_end]
    test_docs = doc_ids[val_end:]
    
    # Collect clauses for each split
    train_clauses = [c for doc_id in train_docs for c in doc_to_clauses[doc_id]]
    val_clauses = [c for doc_id in val_docs for c in doc_to_clauses[doc_id]]
    test_clauses = [c for doc_id in test_docs for c in doc_to_clauses[doc_id]]
    
    logger.info(
        f"Dataset split (document-level, seed={random_seed}):\n"
        f"  Train: {len(train_docs)} docs, {len(train_clauses)} clauses\n"
        f"  Val:   {len(val_docs)} docs, {len(val_clauses)} clauses\n"
        f"  Test:  {len(test_docs)} docs, {len(test_clauses)} clauses"
    )
    
    return train_clauses, val_clauses, test_clauses


def save_split(
    clauses: List[ClauseRecord],
    output_path: str,
    save_format: str = "json",
) -> None:
    """
    Save a list of ClauseRecord objects to a file.
    
    Args:
        clauses: List of ClauseRecord objects to save.
        output_path: Path to the output file.
        save_format: "json" or "csv".
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    records = [c.to_dict() for c in clauses]
    
    if save_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    elif save_format == "csv":
        import csv
        if not records:
            # Write empty file with headers
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "global_id", "doc_id", "clause_id", "text", "labels", "num_labels"
                ])
        else:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                for record in records:
                    # Convert labels list to comma-separated string for CSV
                    record_copy = record.copy()
                    record_copy["labels"] = ", ".join(record_copy["labels"])
                    writer.writerow(record_copy)
    else:
        raise ValueError(f"Unsupported save format: {save_format}. Use 'json' or 'csv'.")
    
    logger.info(f"Saved {len(records)} records to {output_path}")


def save_splits(
    train: List[ClauseRecord],
    val: List[ClauseRecord],
    test: List[ClauseRecord],
    splits_dir: str,
    save_format: str = "json",
) -> None:
    """
    Save train/val/test splits to the splits directory.
    
    Also saves a split_info.json metadata file recording the split
    configuration for reproducibility documentation.
    """
    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)
    
    ext = "json" if save_format == "json" else "csv"
    
    save_split(train, splits_path / f"train.{ext}", save_format)
    save_split(val, splits_path / f"val.{ext}", save_format)
    save_split(test, splits_path / f"test.{ext}", save_format)
    
    # Save split metadata
    split_info = {
        "train_documents": sorted(set(c.doc_id for c in train)),
        "val_documents": sorted(set(c.doc_id for c in val)),
        "test_documents": sorted(set(c.doc_id for c in test)),
        "train_clauses": len(train),
        "val_clauses": len(val),
        "test_clauses": len(test),
        "total_clauses": len(train) + len(val) + len(test),
    }
    
    with open(splits_path / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    
    logger.info(f"All splits saved to {splits_dir}")
