# =============================================================================
# NDA Research Project - Improved Clause Segmenter (v2)
# =============================================================================
# Module 2.1: Advanced Rule-Based Segmentation & Segment Classification
#
# Improvements over Module 2:
# 1. Cleaner preprocessing & boundary filtering (prevents single digits, page numbers, SEC headers as clause boundaries)
# 2. Context-aware legal section boundary detection (handles 1., 1.1, ARTICLE I, SECTION 1, etc.)
# 3. Structural segment classification (legal_clause, section_heading, header, footer, page_number, fragment, unknown)
# 4. Merging broken fragments into adjacent legal clauses when appropriate
# 5. Pipeline-quality warning calculation (OK, REVIEW, HIGH_RISK)
#
# IMPORTANT:
# - Outputs to data/segmentation_v2/
# - Does NOT overwrite original data/segmentation/ outputs
# - Does NOT assign 14 research paper legal categories
# =============================================================================

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ClauseV2:
    document_id: str
    clause_id: str
    text: str
    start: int
    end: int
    segment_type: str  # legal_clause, section_heading, header, footer, page_number, fragment, unknown
    is_suspicious: bool

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "clause_id": self.clause_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "segment_type": self.segment_type,
            "is_suspicious": self.is_suspicious,
        }


@dataclass
class SegmentedDocumentV2:
    document_id: str
    source_file: str
    total_characters: int
    num_clauses: int
    num_legal_clauses: int
    quality_status: str  # OK, REVIEW, HIGH_RISK
    clauses: List[ClauseV2]

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "source_file": self.source_file,
            "total_characters": self.total_characters,
            "num_clauses": len(self.clauses),
            "num_legal_clauses": self.num_legal_clauses,
            "quality_status": self.quality_status,
            "clauses": [c.to_dict() for c in self.clauses],
        }


class ImprovedSegmenterV2:
    """
    Advanced Rule-Based Segmenter for NDA Text (Module 2.1).
    """

    def __init__(self, min_clause_length: int = 20):
        self.min_clause_length = min_clause_length

    def classify_segment(self, text: str) -> str:
        """Classify a segment into structural types based on heuristics."""
        t = text.strip()
        if not t:
            return "fragment"

        # Check SEC Header / Title / Exhibit tag
        if re.match(r'^(?:EX-\d+[\w\.\-]*|EXHIBIT\s+\d+[\w\.\-]*|EXHIBIT\s+[A-Z]|EX-\d+.*htm)', t, re.IGNORECASE):
            return "header"
        
        # Page numbers
        if re.match(r'^(?:Page\s*\d+(?:\s*of\s*\d+)?|\d+\s*of\s*\d+|-\s*\d+\s*-|\d{1,3})$', t, re.IGNORECASE):
            # If it's just a tiny number (1, 2, 3), check length
            if t.isdigit() and len(t) <= 3:
                return "page_number"
            if "page" in t.lower() or "-" in t:
                return "page_number"

        # Standalone Section Heading (e.g., "1. Definitions" or "SECTION 1. CONFIDENTIALITY")
        if len(t) < 80 and re.match(r'^(?:SECTION|ARTICLE|\d+[\.\d]*)\s+[A-Z\s,]{3,60}$', t, re.IGNORECASE) and not t.endswith('.'):
            return "section_heading"

        # Title line (all caps, short)
        if len(t) < 60 and t.isupper() and not t.endswith('.'):
            return "section_heading"

        # Only punctuation/symbols/numbers
        if all(ch in ".,;:!?-–—\"'()[]{}/*\\&$#@ 0123456789" for ch in t):
            return "fragment"

        # Short fragments (< 20 chars) that are not recognized headings
        if len(t) < self.min_clause_length:
            return "fragment"

        return "legal_clause"

    def _find_boundaries(self, text: str) -> List[int]:
        """Find valid legal clause boundaries while ignoring standalone numbers & headers."""
        boundaries = set()

        # Match legal section numbers followed by capitalized word or text (e.g. "1. Scope", "1.1 Subject", "ARTICLE I", "(a) ")
        # Avoid matching standalone isolated digits on empty lines
        patterns = [
            r'(?m)^\s*(?:ARTICLE|SECTION)\s+[IVXLCDM\d]+[\.\:\s]+[A-Z]',
            r'(?m)^\s*\d{1,2}\.\d{1,2}(?:\.\d{1,2})*\s+[A-Z]',
            r'(?m)^\s*\d{1,2}\.\s+[A-Z]',
            r'(?m)^\s*\([a-z0-9]{1,3}\)\s+[A-Z]',
            r'(?m)^\s*(?:WHEREAS|NOW THEREFORE|IN WITNESS WHEREOF),?',
            r'(?m)^[A-Z\s]{4,50}:(?=\s+[A-Z])'
        ]

        for pat in patterns:
            for m in re.finditer(pat, text):
                boundaries.add(m.start())

        return sorted(boundaries)

    def segment(self, text: str, document_id: str) -> SegmentedDocumentV2:
        if not text or not text.strip():
            return SegmentedDocumentV2(
                document_id=document_id,
                source_file=f"{document_id}.txt",
                total_characters=0,
                num_clauses=0,
                num_legal_clauses=0,
                quality_status="HIGH_RISK",
                clauses=[]
            )

        raw_boundaries = self._find_boundaries(text)
        
        # Split text into initial spans
        spans = []
        if not raw_boundaries:
            spans.append((0, len(text)))
        else:
            if raw_boundaries[0] > 0:
                spans.append((0, raw_boundaries[0]))
            for i in range(len(raw_boundaries)):
                start = raw_boundaries[i]
                end = raw_boundaries[i + 1] if i + 1 < len(raw_boundaries) else len(text)
                spans.append((start, end))

        # Merge short fragments (< 25 chars) into next span if fragment is leading or broken line
        processed_spans = []
        i = 0
        while i < len(spans):
            start, end = spans[i]
            stext = text[start:end].strip()
            
            # If fragment is short and not a distinct title/header, attempt merge with next span if available
            if len(stext) < 25 and i + 1 < len(spans) and not re.match(r'^(?:EX-|EXHIBIT|Page|\d+$)', stext, re.IGNORECASE):
                next_start, next_end = spans[i + 1]
                spans[i + 1] = (start, next_end)
                i += 1
                continue
            
            processed_spans.append((start, end))
            i += 1

        # Create ClauseV2 items
        clauses_v2 = []
        legal_count = 0
        suspicious_count = 0

        for idx, (start, end) in enumerate(processed_spans):
            ctext = text[start:end].strip()
            if not ctext:
                continue

            stype = self.classify_segment(ctext)
            
            # Determine if suspicious (short non-legal or weird formatting)
            is_susp = False
            if len(ctext) < 20 or stype in ["fragment", "page_number", "header"]:
                is_susp = True
                suspicious_count += 1

            if stype == "legal_clause":
                legal_count += 1

            c_obj = ClauseV2(
                document_id=document_id,
                clause_id=f"{document_id}_clause_{idx:03d}",
                text=ctext,
                start=start,
                end=end,
                segment_type=stype,
                is_suspicious=is_susp
            )
            clauses_v2.append(c_obj)

        # Quality Status heuristic
        total_c = len(clauses_v2)
        if total_c <= 3 or suspicious_count > total_c * 0.4:
            q_status = "HIGH_RISK"
        elif suspicious_count > 0 or total_c > 50:
            q_status = "REVIEW"
        else:
            q_status = "OK"

        return SegmentedDocumentV2(
            document_id=document_id,
            source_file=f"{document_id}.txt",
            total_characters=len(text),
            num_clauses=total_c,
            num_legal_clauses=legal_count,
            quality_status=q_status,
            clauses=clauses_v2
        )
