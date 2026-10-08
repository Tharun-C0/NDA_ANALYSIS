import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ClauseV3:
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
class SegmentedDocumentV3:
    document_id: str
    source_file: str
    total_characters: int
    num_clauses: int
    num_legal_clauses: int
    quality_status: str
    clauses: List[ClauseV3]

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

class ImprovedSegmenterV3:
    """
    Advanced Rule-Based Segmenter for NDA Text (Module 2.2).
    """
    def __init__(self, min_clause_length: int = 20):
        self.min_clause_length = min_clause_length
        self.legal_keywords = [
            "shall", "must", "agrees", "acknowledges", "party", "parties",
            "confidential", "information", "agreement", "obligations",
            "disclose", "receive", "terminate", "governed", "rights", "duties"
        ]

    def contains_legal_content(self, text: str) -> bool:
        """Heuristic check for substantive legal content."""
        if len(text) < 50:
            return False
        # Count occurrences of legal keywords
        t_lower = text.lower()
        keyword_count = sum(1 for kw in self.legal_keywords if kw in t_lower)
        # If it has at least 2 keywords or is very long, it's likely legal
        if keyword_count >= 2 or len(text) > 300:
            return True
        return False

    def classify_segment(self, text: str) -> str:
        """Classify a segment into structural types based on heuristics."""
        t = text.strip()
        if not t:
            return "fragment"

        # Check if it has substantive legal content (even if it starts with EX-)
        has_legal = self.contains_legal_content(t)

        # Only punctuation/symbols/numbers
        if all(ch in ".,;:!?-–—\"'()[]{}/*\\&$#@ 0123456789\n\r" for ch in t):
            return "fragment"

        # SEC Header / Title / Exhibit tag
        if re.match(r'^(?:EX-\d+[\w\.\-]*|EXHIBIT\s+\d+[\w\.\-]*|EXHIBIT\s+[A-Z]|EX-\d+.*htm)', t, re.IGNORECASE):
            if has_legal:
                # If it's a giant block that just happens to start with Exhibit, it's a legal_clause.
                return "legal_clause"
            if len(t) < 300:
                return "header"

        # Page numbers
        if re.match(r'^(?:Page\s*\d+(?:\s*of\s*\d+)?|\d+\s*of\s*\d+|-\s*\d+\s*-|\d{1,3})$', t, re.IGNORECASE):
            if t.isdigit() and len(t) <= 3:
                return "page_number"
            if "page" in t.lower() or "-" in t:
                return "page_number"

        # Document titles / Section headings
        # Don't classify as heading if it has substantive legal content.
        if not has_legal:
            if len(t) < 150 and re.match(r'^(?:SECTION|ARTICLE|\d+[\.\d]*)\s+[A-Z\s,]{3,100}$', t, re.IGNORECASE) and not t.endswith('.'):
                return "section_heading"

            if len(t) < 100 and t.isupper() and not t.endswith('.') and len(t.split()) < 15:
                return "section_heading"

        # Short fragments (< 20 chars) that are not recognized headings
        if len(t) < self.min_clause_length and not has_legal:
            return "fragment"

        return "legal_clause"

    def _find_boundaries(self, text: str) -> List[int]:
        """Find valid legal clause boundaries."""
        boundaries = set()

        # V1 baseline had:
        # _NUMBERED_SECTION = r"^\s*(\d{1,3}(?:\.\d{1,3})*)\s*[\.\)]\s+"
        # V2 tried to enforce a capital letter, which missed a lot.
        # Let's use a balanced approach. Match typical numbering, but ignore isolated lines with just a number.
        
        patterns = [
            # 1. Scope  OR  1.1 Scope
            r'(?m)^\s*(?:\d{1,3}(?:\.\d{1,3})*)\s*[\.\)]\s+[A-Z0-9]',
            # (a) text
            r'(?m)^\s*\([a-z0-9]{1,4}\)\s+[A-Z0-9]',
            # ARTICLE I, Section 1
            r'(?m)^\s*(?:ARTICLE|Article|SECTION|Section)\s+[IVXLCDM\d]+[\.\:\s]',
            # ALL CAPS HEADINGS on their own line (at least 3 caps words, preceding a newline)
            r'(?m)^[A-Z][A-Z\s,\-/&]{10,}$',
            # WHEREAS / NOW THEREFORE
            r'(?m)^\s*(?:WHEREAS|NOW THEREFORE|IN WITNESS WHEREOF)',
            # Double newline (paragraph boundary) - to avoid swallowing huge blocks if they lack numbers
            r'\n\s*\n'
        ]

        # Double newline is dangerous if used blindly, let's look for double newlines that separate substantial text.
        for pat in patterns:
            for m in re.finditer(pat, text):
                # For double newlines, just add the boundary after the newlines
                if pat == r'\n\s*\n':
                    boundaries.add(m.end())
                else:
                    boundaries.add(m.start())

        return sorted(boundaries)

    def segment(self, text: str, document_id: str) -> SegmentedDocumentV3:
        if not text or not text.strip():
            return SegmentedDocumentV3(
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

        # Merge extremely short fragments (e.g. < 20 chars) if they are likely just a number or stray text
        # But do NOT over-merge!
        processed_spans = []
        i = 0
        while i < len(spans):
            start, end = spans[i]
            stext = text[start:end].strip()
            
            # If fragment is very short and next span exists, merge
            # Only merge if it's less than min_clause_length and doesn't look like a valid heading
            is_heading = re.match(r'^(?:SECTION|ARTICLE|EX-|EXHIBIT)', stext, re.IGNORECASE) or (stext.isupper() and len(stext)>5)
            
            if len(stext) < self.min_clause_length and i + 1 < len(spans) and not is_heading:
                next_start, next_end = spans[i + 1]
                spans[i + 1] = (start, next_end)
                i += 1
                continue
            
            processed_spans.append((start, end))
            i += 1

        # Create ClauseV3 items
        clauses_v3 = []
        legal_count = 0
        suspicious_count = 0

        for idx, (start, end) in enumerate(processed_spans):
            ctext = text[start:end].strip()
            if not ctext:
                continue

            stype = self.classify_segment(ctext)
            
            # Determine if suspicious (short non-legal or weird formatting, OR suspiciously large header)
            is_susp = False
            if stype in ["fragment", "page_number"] or (stype == "header" and len(ctext) > 500) or len(ctext) < 15:
                is_susp = True
                suspicious_count += 1

            if stype == "legal_clause":
                legal_count += 1

            c_obj = ClauseV3(
                document_id=document_id,
                clause_id=f"{document_id}_clause_{idx:03d}",
                text=ctext,
                start=start,
                end=end,
                segment_type=stype,
                is_suspicious=is_susp
            )
            clauses_v3.append(c_obj)

        # Quality Status heuristic
        total_c = len(clauses_v3)
        if total_c <= 3 or suspicious_count > total_c * 0.4:
            q_status = "HIGH_RISK"
        elif suspicious_count > 0 or total_c > 100:
            q_status = "REVIEW"
        else:
            q_status = "OK"

        return SegmentedDocumentV3(
            document_id=document_id,
            source_file=f"{document_id}.txt",
            total_characters=len(text),
            num_clauses=total_c,
            num_legal_clauses=legal_count,
            quality_status=q_status,
            clauses=clauses_v3
        )
