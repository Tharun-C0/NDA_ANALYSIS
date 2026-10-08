# =============================================================================
# NDA Research Project - Dataset Loader
# =============================================================================
# Module 1: Dataset Loading and Clause Parsing
#
# Reads annotated NDA text files from data/raw/ and parses them into
# structured clause records. Handles the annotation format:
#
#   [INIT_CLAUSE]
#   clause text content
#   [INIT_CLASSE]label1, label2[END_CLASSE]
#   [END_CLAUSE]
#
# Research Design Decisions:
# - Each file is treated as one NDA document; the filename serves as doc_id.
# - Clause IDs are assigned sequentially within each document.
# - Multi-label clauses are supported (comma-separated labels).
# - Malformed clauses are logged and skipped to avoid silent data corruption.
# - All parsing is deterministic and reproducible.
# =============================================================================

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ClauseRecord:
    """
    Represents a single parsed clause from an NDA document.
    
    Attributes:
        doc_id (str): Identifier of the source NDA document (filename stem).
        clause_id (int): Sequential clause index within the document (0-based).
        text (str): Raw clause text content.
        labels (List[str]): List of class labels assigned to this clause.
        global_id (str): Unique identifier combining doc_id and clause_id.
    """
    
    def __init__(self, doc_id: str, clause_id: int, text: str, labels: List[str]):
        self.doc_id = doc_id
        self.clause_id = clause_id
        self.text = text
        self.labels = labels
        self.global_id = f"{doc_id}_clause_{clause_id}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "global_id": self.global_id,
            "doc_id": self.doc_id,
            "clause_id": self.clause_id,
            "text": self.text,
            "labels": self.labels,
            "num_labels": len(self.labels),
        }
    
    def __repr__(self):
        text_preview = self.text[:80] + "..." if len(self.text) > 80 else self.text
        return (
            f"ClauseRecord(doc_id='{self.doc_id}', clause_id={self.clause_id}, "
            f"labels={self.labels}, text='{text_preview}')"
        )


def parse_clauses_from_text(
    text: str,
    doc_id: str,
    clause_start: str = "[INIT_CLAUSE]",
    clause_end: str = "[END_CLAUSE]",
    class_start: str = "[INIT_CLASSE]",
    class_end: str = "[END_CLASSE]",
) -> Tuple[List[ClauseRecord], List[str]]:
    """
    Parse annotated clause text into structured ClauseRecord objects.
    
    This function extracts all clause blocks from the raw text, separating
    the clause content from the class labels. It handles multi-label
    annotations (comma-separated) and reports any parsing errors.
    
    Args:
        text: Full text content of the NDA file.
        doc_id: Document identifier (typically the filename stem).
        clause_start: Opening delimiter for a clause block.
        clause_end: Closing delimiter for a clause block.
        class_start: Opening delimiter for the class label section.
        class_end: Closing delimiter for the class label section.
    
    Returns:
        Tuple of:
            - List of successfully parsed ClauseRecord objects.
            - List of warning/error messages for malformed sections.
    
    Research Note:
        The parser is designed to be robust against minor formatting issues
        (extra whitespace, blank lines) while strictly requiring the delimiter
        structure. This prevents silent data loss in annotation files.
    """
    clauses = []
    warnings = []
    
    # Build regex pattern to match clause blocks
    # Using re.DOTALL so '.' matches newlines within clause text
    # The pattern captures everything between INIT_CLAUSE and END_CLAUSE
    pattern = re.compile(
        re.escape(clause_start) + r"(.*?)" + re.escape(clause_end),
        re.DOTALL
    )
    
    # Pattern to extract class labels from within a clause block
    class_pattern = re.compile(
        re.escape(class_start) + r"(.*?)" + re.escape(class_end),
        re.DOTALL
    )
    
    matches = pattern.findall(text)
    
    if not matches:
        warnings.append(f"[{doc_id}] No clause blocks found in document.")
        return clauses, warnings
    
    for idx, block in enumerate(matches):
        # Extract class labels
        class_match = class_pattern.search(block)
        
        if class_match is None:
            warnings.append(
                f"[{doc_id}] Clause {idx}: No class labels found. Skipping."
            )
            continue
        
        # Extract the label string and parse individual labels
        label_str = class_match.group(1).strip()
        if not label_str:
            warnings.append(
                f"[{doc_id}] Clause {idx}: Empty class label section. Skipping."
            )
            continue
        
        # Labels may be comma-separated for multi-label clauses
        labels = [label.strip() for label in label_str.split(",") if label.strip()]
        
        if not labels:
            warnings.append(
                f"[{doc_id}] Clause {idx}: No valid labels after parsing. Skipping."
            )
            continue
        
        # Extract clause text (everything before the class label section)
        # Remove the class label block from the clause content
        clause_text = block[:class_match.start()].strip()
        
        if not clause_text:
            warnings.append(
                f"[{doc_id}] Clause {idx}: Empty clause text. Skipping."
            )
            continue
        
        clause = ClauseRecord(
            doc_id=doc_id,
            clause_id=idx,
            text=clause_text,
            labels=labels,
        )
        clauses.append(clause)
    
    return clauses, warnings


def load_single_document(
    filepath: Path,
    encoding: str = "utf-8",
    **parse_kwargs,
) -> Tuple[List[ClauseRecord], List[str]]:
    """
    Load and parse a single NDA document file.
    
    Args:
        filepath: Path to the NDA text file.
        encoding: File encoding (default: utf-8).
        **parse_kwargs: Additional keyword arguments passed to parse_clauses_from_text.
    
    Returns:
        Tuple of parsed clauses and warning messages.
    """
    filepath = Path(filepath)
    doc_id = filepath.stem  # Use filename without extension as document ID
    
    warnings = []
    
    try:
        with open(filepath, "r", encoding=encoding) as f:
            text = f.read()
    except UnicodeDecodeError as e:
        msg = f"[{doc_id}] Encoding error ({encoding}): {e}. Trying latin-1 fallback."
        warnings.append(msg)
        logger.warning(msg)
        try:
            with open(filepath, "r", encoding="latin-1") as f:
                text = f.read()
        except Exception as e2:
            msg = f"[{doc_id}] Failed to read file with fallback encoding: {e2}"
            warnings.append(msg)
            logger.error(msg)
            return [], warnings
    except Exception as e:
        msg = f"[{doc_id}] Failed to read file: {e}"
        warnings.append(msg)
        logger.error(msg)
        return [], warnings
    
    if not text.strip():
        msg = f"[{doc_id}] File is empty."
        warnings.append(msg)
        logger.warning(msg)
        return [], warnings
    
    clauses, parse_warnings = parse_clauses_from_text(text, doc_id, **parse_kwargs)
    warnings.extend(parse_warnings)
    
    return clauses, warnings


def load_dataset(
    raw_dir: str,
    encoding: str = "utf-8",
    file_extensions: tuple = (".txt",),
    **parse_kwargs,
) -> Tuple[List[ClauseRecord], Dict[str, List[str]]]:
    """
    Load all NDA documents from a directory.
    
    Scans the raw_dir for text files matching the specified extensions,
    parses each one, and aggregates all clauses into a single list.
    
    Args:
        raw_dir: Path to the directory containing raw NDA text files.
        encoding: File encoding for reading documents.
        file_extensions: Tuple of valid file extensions to consider.
        **parse_kwargs: Additional arguments for the clause parser.
    
    Returns:
        Tuple of:
            - List of all ClauseRecord objects across all documents.
            - Dictionary mapping doc_id to its list of warning messages.
    
    Raises:
        FileNotFoundError: If raw_dir does not exist.
        ValueError: If no matching files are found in raw_dir.
    """
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            f"Please create it and place your NDA .txt files inside."
        )
    
    # Collect all matching files
    files = sorted([
        f for f in raw_path.iterdir()
        if f.is_file() and f.suffix.lower() in file_extensions
    ])
    
    if not files:
        raise ValueError(
            f"No files with extensions {file_extensions} found in {raw_dir}.\n"
            f"Please place your annotated NDA .txt files in this directory."
        )
    
    logger.info(f"Found {len(files)} NDA document(s) in {raw_dir}")
    
    all_clauses = []
    all_warnings = {}
    
    for filepath in files:
        clauses, warnings = load_single_document(
            filepath, encoding=encoding, **parse_kwargs
        )
        all_clauses.extend(clauses)
        
        if warnings:
            all_warnings[filepath.stem] = warnings
            for w in warnings:
                logger.warning(w)
        
        logger.debug(
            f"  Loaded {filepath.name}: {len(clauses)} clauses"
        )
    
    logger.info(
        f"Total: {len(all_clauses)} clauses from {len(files)} documents. "
        f"{len(all_warnings)} documents had warnings."
    )
    
    return all_clauses, all_warnings
