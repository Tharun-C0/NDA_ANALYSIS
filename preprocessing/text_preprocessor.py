# =============================================================================
# NDA Research Project - Text Preprocessor
# =============================================================================
# Module 1: Clause Text Preprocessing
#
# Cleans clause text while preserving legal meaning. This module is
# intentionally conservative — it removes only formatting artifacts and
# normalizes whitespace. It does NOT:
#   - Paraphrase or rephrase text
#   - Remove legal terminology or boilerplate
#   - Lowercase text (case may carry legal meaning)
#   - Perform stemming or lemmatization
#
# Research Design Decisions:
# - Preprocessing is deterministic: same input always produces same output.
# - No information-destroying transformations at this stage.
# - Tokenization and truncation are deferred to the model-specific pipeline.
# =============================================================================

import re
import unicodedata
import logging
from typing import List

from preprocessing.dataset_loader import ClauseRecord

logger = logging.getLogger(__name__)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in clause text.
    
    - Collapses multiple spaces/tabs into a single space.
    - Preserves paragraph breaks (double newlines) as single newlines.
    - Strips leading/trailing whitespace.
    
    This is safe for legal text because it only affects formatting,
    not the content itself.
    """
    # Replace tabs with spaces
    text = text.replace("\t", " ")
    # Collapse runs of spaces (not newlines) into single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ newlines into 2 (preserve paragraph structure)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Strip overall leading/trailing whitespace
    text = text.strip()
    return text


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters to their canonical form (NFC).
    
    This ensures that characters with multiple unicode representations
    (e.g., accented characters) are stored consistently. Important for
    reproducible tokenization downstream.
    """
    return unicodedata.normalize("NFC", text)


def clean_encoding_artifacts(text: str) -> str:
    """
    Remove common encoding artifacts that appear in PDF-extracted text.
    
    Targets:
    - Null bytes
    - Control characters (except newline, tab, carriage return)
    - Common PDF extraction artifacts (e.g., \x0c form feed)
    
    Does NOT remove any printable characters to preserve legal content.
    """
    # Remove null bytes
    text = text.replace("\x00", "")
    # Remove form feed characters (common in PDF extraction)
    text = text.replace("\x0c", "")
    # Remove other control characters except \n, \r, \t
    text = re.sub(r"[\x01-\x08\x0b\x0e-\x1f\x7f]", "", text)
    return text


def preprocess_clause_text(
    text: str,
    strip_ws: bool = True,
    do_normalize_unicode: bool = True,
) -> str:
    """
    Apply the full preprocessing pipeline to a single clause text.
    
    The pipeline is applied in a fixed order to ensure determinism:
    1. Clean encoding artifacts
    2. Normalize unicode (optional)
    3. Normalize whitespace (optional)
    
    Args:
        text: Raw clause text.
        strip_ws: Whether to normalize whitespace.
        do_normalize_unicode: Whether to normalize unicode.
    
    Returns:
        Cleaned clause text.
    """
    text = clean_encoding_artifacts(text)
    
    if do_normalize_unicode:
        text = normalize_unicode(text)
    
    if strip_ws:
        text = normalize_whitespace(text)
    
    return text


def preprocess_labels(labels: List[str]) -> List[str]:
    """
    Normalize label strings.
    
    - Strips whitespace.
    - Converts to consistent casing (lowercase).
    - Removes empty labels.
    
    Research Note:
        Lowercasing labels ensures that "Confidentiality" and "confidentiality"
        are treated as the same class. This is a standard practice in text
        classification research.
    """
    cleaned = []
    for label in labels:
        label = label.strip().lower()
        if label:
            cleaned.append(label)
    return sorted(set(cleaned))  # Remove duplicates, sort for consistency


def preprocess_clauses(
    clauses: List[ClauseRecord],
    strip_ws: bool = True,
    do_normalize_unicode: bool = True,
) -> List[ClauseRecord]:
    """
    Apply preprocessing to a list of ClauseRecord objects.
    
    Creates new ClauseRecord objects with cleaned text and normalized labels.
    Original records are not modified (immutability for reproducibility).
    
    Args:
        clauses: List of raw ClauseRecord objects from the loader.
        strip_ws: Whether to normalize whitespace in clause text.
        do_normalize_unicode: Whether to normalize unicode characters.
    
    Returns:
        List of preprocessed ClauseRecord objects.
    """
    processed = []
    skipped = 0
    
    for clause in clauses:
        # Preprocess text
        clean_text = preprocess_clause_text(
            clause.text,
            strip_ws=strip_ws,
            do_normalize_unicode=do_normalize_unicode,
        )
        
        # Preprocess labels
        clean_labels = preprocess_labels(clause.labels)
        
        # Skip clauses that become empty after preprocessing
        if not clean_text or not clean_labels:
            logger.warning(
                f"Clause {clause.global_id} became empty after preprocessing. "
                f"Skipping."
            )
            skipped += 1
            continue
        
        processed_clause = ClauseRecord(
            doc_id=clause.doc_id,
            clause_id=clause.clause_id,
            text=clean_text,
            labels=clean_labels,
        )
        processed.append(processed_clause)
    
    if skipped > 0:
        logger.info(f"Skipped {skipped} clauses that became empty after preprocessing.")
    
    logger.info(f"Preprocessed {len(processed)} clauses successfully.")
    
    return processed
