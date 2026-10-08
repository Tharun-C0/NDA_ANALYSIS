# =============================================================================
# NDA Research Project - PDF Text Extractor
# =============================================================================
# Module 2: Extracts text from NDA PDF documents.
#
# Uses PyMuPDF (fitz) for robust text extraction from PDF files.
# Handles encoding issues, empty pages, and malformed PDFs gracefully.
#
# Research Design Decisions:
# - Text is extracted page-by-page and concatenated with page markers.
# - No summarization or paraphrasing — the full legal text is preserved.
# - Only formatting artifacts (excessive whitespace, control chars) are
#   cleaned; legally meaningful content is never removed.
# - Each extraction produces a quality report for auditing.
# =============================================================================

import re
import logging
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError(
        "PyMuPDF is required for PDF extraction. "
        "Install it with: pip install PyMuPDF"
    )


@dataclass
class ExtractionResult:
    """Result of extracting text from a single PDF document."""
    document_id: str
    source_file: str
    text: str
    character_count: int
    word_count: int
    line_count: int
    page_count: int
    pages_with_text: int
    status: str  # "success", "partial", "failed", "empty"
    error_message: str = ""

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "source_file": self.source_file,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "line_count": self.line_count,
            "page_count": self.page_count,
            "pages_with_text": self.pages_with_text,
            "status": self.status,
            "error_message": self.error_message,
        }


def clean_extracted_text(text: str) -> str:
    """
    Clean text extracted from PDF while preserving legal meaning.

    Operations (in order):
    1. Normalize unicode to NFC form.
    2. Remove null bytes and non-printable control characters.
    3. Replace tabs with spaces.
    4. Collapse runs of spaces (not newlines) into single space.
    5. Remove trailing whitespace from each line.
    6. Collapse 3+ blank lines into 2 (preserve paragraph structure).
    7. Strip leading/trailing whitespace from the entire text.

    Does NOT:
    - Remove any printable characters.
    - Lowercase text.
    - Paraphrase or summarize.
    """
    # 1. Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Remove control characters except \n, \r, \t
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 3. Replace tabs
    text = text.replace("\t", " ")

    # 4. Collapse runs of spaces
    text = re.sub(r"[^\S\n]+", " ", text)

    # 5. Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 6. Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 7. Strip overall
    text = text.strip()

    return text


def extract_text_from_pdf(
    pdf_path: str,
    min_characters: int = 50,
) -> ExtractionResult:
    """
    Extract text from a single PDF file.

    Uses PyMuPDF to extract text page-by-page, then cleans formatting
    artifacts while preserving legal content.

    Args:
        pdf_path: Path to the PDF file.
        min_characters: Minimum characters for a "success" status.

    Returns:
        ExtractionResult with extracted text and quality metrics.
    """
    pdf_path = Path(pdf_path)
    document_id = pdf_path.stem

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        return ExtractionResult(
            document_id=document_id,
            source_file=pdf_path.name,
            text="",
            character_count=0,
            word_count=0,
            line_count=0,
            page_count=0,
            pages_with_text=0,
            status="failed",
            error_message=f"Failed to open PDF: {e}",
        )

    page_texts = []
    pages_with_text = 0

    try:
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                page_text = page.get_text("text")
                if page_text and page_text.strip():
                    pages_with_text += 1
                    page_texts.append(page_text)
                else:
                    page_texts.append("")
            except Exception as e:
                logger.warning(
                    f"[{document_id}] Error extracting page {page_num}: {e}"
                )
                page_texts.append("")
    finally:
        page_count = len(doc)
        doc.close()

    # Combine pages with double newline separator
    raw_text = "\n\n".join(
        t for t in page_texts if t.strip()
    )

    # Clean the extracted text
    cleaned_text = clean_extracted_text(raw_text)

    # Compute quality metrics
    char_count = len(cleaned_text)
    word_count = len(cleaned_text.split()) if cleaned_text else 0
    line_count = cleaned_text.count("\n") + 1 if cleaned_text else 0

    # Determine status
    if char_count == 0:
        status = "empty"
        error_msg = "No text extracted from any page."
    elif char_count < min_characters:
        status = "partial"
        error_msg = f"Only {char_count} characters extracted (minimum: {min_characters})."
    elif pages_with_text < page_count:
        status = "partial"
        error_msg = f"Text found on {pages_with_text}/{page_count} pages."
    else:
        status = "success"
        error_msg = ""

    return ExtractionResult(
        document_id=document_id,
        source_file=pdf_path.name,
        text=cleaned_text,
        character_count=char_count,
        word_count=word_count,
        line_count=line_count,
        page_count=page_count,
        pages_with_text=pages_with_text,
        status=status,
        error_message=error_msg,
    )


def save_extracted_text(
    result: ExtractionResult,
    output_dir: str,
    encoding: str = "utf-8",
) -> Optional[Path]:
    """
    Save extracted text to a .txt file.

    Args:
        result: ExtractionResult from PDF extraction.
        output_dir: Directory to save the text file.
        encoding: Output file encoding.

    Returns:
        Path to saved file, or None if extraction failed.
    """
    if result.status == "failed" or not result.text:
        logger.warning(
            f"[{result.document_id}] Skipping save — "
            f"status={result.status}: {result.error_message}"
        )
        return None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filepath = out_path / f"{result.document_id}.txt"
    with open(filepath, "w", encoding=encoding) as f:
        f.write(result.text)

    logger.debug(f"[{result.document_id}] Saved {result.character_count} chars to {filepath}")
    return filepath


def extract_batch(
    pdf_paths: List[Path],
    output_dir: str,
    min_characters: int = 50,
    encoding: str = "utf-8",
) -> List[ExtractionResult]:
    """
    Extract text from multiple PDF files.

    Args:
        pdf_paths: List of PDF file paths.
        output_dir: Directory to save extracted text files.
        min_characters: Minimum characters for success status.
        encoding: Output file encoding.

    Returns:
        List of ExtractionResult objects.
    """
    results = []

    for i, pdf_path in enumerate(pdf_paths):
        logger.info(
            f"  Extracting [{i+1}/{len(pdf_paths)}]: {pdf_path.name}"
        )
        result = extract_text_from_pdf(str(pdf_path), min_characters)
        save_extracted_text(result, output_dir, encoding)
        results.append(result)

    # Summary
    success = sum(1 for r in results if r.status == "success")
    partial = sum(1 for r in results if r.status == "partial")
    failed = sum(1 for r in results if r.status in ("failed", "empty"))

    logger.info(
        f"Extraction complete: {success} success, {partial} partial, "
        f"{failed} failed out of {len(results)} documents."
    )

    return results
