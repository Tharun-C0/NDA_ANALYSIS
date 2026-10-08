# =============================================================================
# NDA Research Project - Clause Segmenter
# =============================================================================
# Module 2: Splits extracted NDA text into candidate clauses.
#
# This module implements a MODULAR segmentation architecture:
#
#   BaseSegmenter (abstract)
#       ├── BaselineSegmenter  — rule-based, uses legal formatting patterns
#       └── (future) LLMSegmenter — pluggable LLM-based segmentation
#
# The baseline segmenter detects clause boundaries using:
# - Numbered section patterns (1., 1.1, (a), (i), Article I, Section 1, etc.)
# - Legal section headings (CONFIDENTIALITY, TERM, etc.)
# - Paragraph boundaries with structural signals
#
# IMPORTANT:
# - The output clauses are PRELIMINARY and are NOT ground-truth annotations.
# - No class labels are assigned. No risk scores are computed.
# - The segmenter is intentionally conservative — it prefers over-segmenting
#   (which can be merged during human review) over under-segmenting
#   (which would lose clause boundaries).
#
# Research Design Decisions:
# - The abstract base class allows drop-in replacement with LLM-based
#   segmenters (Llama, Qwen, Mistral, Gemma) without changing downstream code.
# - Clause IDs encode document provenance for traceability.
# - Start/end character positions are preserved to enable alignment with
#   the original text and future annotation correction.
# =============================================================================

import re
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Clause:
    """
    A single segmented clause from an NDA document.

    NOTE: This is an automatically detected clause boundary, NOT a
    human-verified or ground-truth annotation. No class labels are assigned.
    """
    clause_id: str
    text: str
    start_position: int
    end_position: int

    def to_dict(self) -> dict:
        return {
            "clause_id": self.clause_id,
            "text": self.text,
            "start_position": self.start_position,
            "end_position": self.end_position,
        }


@dataclass
class SegmentedDocument:
    """Collection of clauses extracted from a single NDA document."""
    document_id: str
    source_file: str
    total_characters: int
    clauses: List[Clause]

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "source_file": self.source_file,
            "total_characters": self.total_characters,
            "num_clauses": len(self.clauses),
            "clauses": [c.to_dict() for c in self.clauses],
        }


# =============================================================================
# Abstract Base
# =============================================================================

class BaseSegmenter(ABC):
    """
    Abstract base class for clause segmenters.

    All segmenter implementations must inherit from this class and
    implement the `segment` method. This ensures that baseline and
    LLM-based segmenters are interchangeable in the pipeline.

    To add a new segmenter (e.g., LLM-based):
        class LLMSegmenter(BaseSegmenter):
            def __init__(self, model_name, ...):
                super().__init__()
                self.model = load_model(model_name)

            def segment(self, text, document_id):
                # Use LLM to detect clause boundaries
                ...
    """

    @abstractmethod
    def segment(self, text: str, document_id: str) -> SegmentedDocument:
        """
        Segment document text into clauses.

        Args:
            text: Full extracted text of the NDA document.
            document_id: Unique identifier for the document.

        Returns:
            SegmentedDocument containing detected clauses.
        """
        pass


# =============================================================================
# Baseline Rule-Based Segmenter
# =============================================================================

# Compiled regex patterns for legal section/clause detection.
# These are common patterns found in NDA and legal contract formatting.

# Matches: "1.", "2.", "10.", "1.1", "1.1.1", etc. at line start
_NUMBERED_SECTION = re.compile(
    r"^\s*(\d{1,3}(?:\.\d{1,3})*)\s*[\.\)]\s+",
    re.MULTILINE,
)

# Matches: "(a)", "(b)", "(i)", "(ii)", "(1)", "(2)", etc.
_PAREN_LETTER = re.compile(
    r"^\s*\([a-z]{1,4}\)\s+",
    re.MULTILINE,
)
_PAREN_ROMAN = re.compile(
    r"^\s*\([ivxlcdm]+\)\s+",
    re.MULTILINE | re.IGNORECASE,
)
_PAREN_NUMBER = re.compile(
    r"^\s*\(\d{1,3}\)\s+",
    re.MULTILINE,
)

# Matches: "Article I", "Article 1", "ARTICLE I", "Section 1", "SECTION 1"
_ARTICLE_SECTION = re.compile(
    r"^\s*(?:ARTICLE|Article|SECTION|Section)\s+[IVXLCDM\d]+[\.\:\s]",
    re.MULTILINE,
)

# Matches all-caps headings that are common legal section titles
# (at least 3 uppercase words on a line by themselves)
_LEGAL_HEADING = re.compile(
    r"^[A-Z][A-Z\s,\-/&]{5,}$",
    re.MULTILINE,
)


class BaselineSegmenter(BaseSegmenter):
    """
    Rule-based clause segmenter using legal document formatting patterns.

    This is the BASELINE approach. It detects clause boundaries using:
    1. Numbered sections (1., 2., 1.1, etc.)
    2. Article/Section headings
    3. Legal headings (ALL CAPS lines)
    4. Parenthesized markers ((a), (i), (1))
    5. Paragraph boundaries with structural signals

    The segmenter is conservative: it prefers to produce more smaller
    segments (which can be merged during human review) rather than fewer
    large segments that might span multiple clauses.

    Args:
        min_clause_length: Minimum characters for a clause. Shorter
            segments are merged with the previous clause.
        max_clause_length: Maximum characters. Longer segments are
            split at paragraph boundaries.
    """

    def __init__(
        self,
        min_clause_length: int = 30,
        max_clause_length: int = 5000,
    ):
        self.min_clause_length = min_clause_length
        self.max_clause_length = max_clause_length

    def _find_boundary_positions(self, text: str) -> List[int]:
        """
        Find all candidate clause boundary positions in the text.

        Returns a sorted, deduplicated list of character positions where
        a new clause is likely to begin.
        """
        boundaries = set()

        # Priority 1: Article/Section headings
        for m in _ARTICLE_SECTION.finditer(text):
            boundaries.add(m.start())

        # Priority 2: Numbered sections
        for m in _NUMBERED_SECTION.finditer(text):
            boundaries.add(m.start())

        # Priority 3: Legal headings (ALL CAPS)
        for m in _LEGAL_HEADING.finditer(text):
            line = m.group().strip()
            # Filter out very short caps strings that might just be
            # abbreviations (e.g., "LLC", "INC")
            if len(line) >= 8:
                boundaries.add(m.start())

        # Priority 4: Parenthesized markers
        for pattern in [_PAREN_LETTER, _PAREN_ROMAN, _PAREN_NUMBER]:
            for m in pattern.finditer(text):
                boundaries.add(m.start())

        return sorted(boundaries)

    def _split_at_boundaries(
        self, text: str, boundaries: List[int]
    ) -> List[tuple]:
        """
        Split text at detected boundary positions.

        Returns list of (start, end) tuples representing clause spans.
        """
        if not boundaries:
            # No boundaries found — treat entire text as one clause
            return [(0, len(text))]

        spans = []

        # If first boundary is not at position 0, include preamble
        if boundaries[0] > 0:
            preamble = text[:boundaries[0]].strip()
            if preamble:
                spans.append((0, boundaries[0]))

        # Create spans between consecutive boundaries
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            spans.append((start, end))

        return spans

    def _merge_short_segments(
        self, text: str, spans: List[tuple]
    ) -> List[tuple]:
        """
        Merge segments shorter than min_clause_length with their
        predecessor. This prevents tiny fragments from becoming
        standalone clauses.
        """
        if not spans:
            return spans

        merged = [spans[0]]
        for start, end in spans[1:]:
            segment_text = text[start:end].strip()
            if len(segment_text) < self.min_clause_length and merged:
                # Extend previous span to include this one
                prev_start, _ = merged[-1]
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        return merged

    def _split_long_segments(
        self, text: str, spans: List[tuple]
    ) -> List[tuple]:
        """
        Split segments longer than max_clause_length at paragraph
        boundaries (double newlines).
        """
        result = []
        for start, end in spans:
            segment = text[start:end]
            if len(segment.strip()) <= self.max_clause_length:
                result.append((start, end))
                continue

            # Split at paragraph boundaries (double newline)
            parts = re.split(r"\n\n+", segment)
            pos = start
            for part in parts:
                part_start = text.find(part, pos)
                if part_start == -1:
                    part_start = pos
                part_end = part_start + len(part)
                if part.strip():
                    result.append((part_start, part_end))
                pos = part_end

        return result

    def segment(self, text: str, document_id: str) -> SegmentedDocument:
        """
        Segment NDA text into candidate clauses.

        Args:
            text: Full extracted text of the NDA.
            document_id: Document identifier.

        Returns:
            SegmentedDocument with detected clause boundaries.
        """
        if not text or not text.strip():
            logger.warning(f"[{document_id}] Empty text — no clauses.")
            return SegmentedDocument(
                document_id=document_id,
                source_file=f"{document_id}.txt",
                total_characters=0,
                clauses=[],
            )

        # Step 1: Find candidate boundary positions
        boundaries = self._find_boundary_positions(text)

        # Step 2: Split text at boundaries
        spans = self._split_at_boundaries(text, boundaries)

        # Step 3: Merge very short segments
        spans = self._merge_short_segments(text, spans)

        # Step 4: Split very long segments
        spans = self._split_long_segments(text, spans)

        # Step 5: Create Clause objects
        clauses = []
        clause_idx = 0
        for start, end in spans:
            clause_text = text[start:end].strip()
            if not clause_text:
                continue

            clause = Clause(
                clause_id=f"{document_id}_clause_{clause_idx:03d}",
                text=clause_text,
                start_position=start,
                end_position=end,
            )
            clauses.append(clause)
            clause_idx += 1

        logger.info(
            f"[{document_id}] Segmented into {len(clauses)} clauses "
            f"({len(text)} chars, {len(boundaries)} boundaries detected)"
        )

        return SegmentedDocument(
            document_id=document_id,
            source_file=f"{document_id}.txt",
            total_characters=len(text),
            clauses=clauses,
        )


# =============================================================================
# Segmenter Factory
# =============================================================================

def create_segmenter(
    strategy: str = "baseline",
    min_clause_length: int = 30,
    max_clause_length: int = 5000,
    **kwargs,
) -> BaseSegmenter:
    """
    Factory function to create a segmenter instance.

    This is the single entry point for obtaining a segmenter. When
    LLM-based segmenters are added, they will be registered here.

    Args:
        strategy: "baseline" or (future) "llm".
        min_clause_length: Minimum clause length in characters.
        max_clause_length: Maximum clause length in characters.

    Returns:
        An instance of a BaseSegmenter subclass.

    Raises:
        ValueError: If strategy is not recognized.
    """
    if strategy == "baseline":
        return BaselineSegmenter(
            min_clause_length=min_clause_length,
            max_clause_length=max_clause_length,
        )
    # Future: elif strategy == "llm": return LLMSegmenter(**kwargs)
    else:
        raise ValueError(
            f"Unknown segmentation strategy: '{strategy}'. "
            f"Available: 'baseline'. Future: 'llm'."
        )


# =============================================================================
# I/O Functions
# =============================================================================

def save_segmented_document(
    doc: SegmentedDocument,
    output_dir: str,
) -> Path:
    """Save a segmented document to JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filepath = out_path / f"{doc.document_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)

    return filepath


def load_segmented_document(filepath: str) -> SegmentedDocument:
    """Load a segmented document from JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    clauses = [
        Clause(
            clause_id=c["clause_id"],
            text=c["text"],
            start_position=c["start_position"],
            end_position=c["end_position"],
        )
        for c in data["clauses"]
    ]

    return SegmentedDocument(
        document_id=data["document_id"],
        source_file=data["source_file"],
        total_characters=data["total_characters"],
        clauses=clauses,
    )
